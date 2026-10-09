#include "fix_rotation_track.h"

#include "atom.h"
#include "domain.h"
#include "math_const.h"
#include "error.h"
#include "utils.h"
#include <cstring>
#include "group.h"
#include <mpi.h>
#include <vector>
#include "math_extra.h"
#include <cmath>
#include <algorithm>
#include <limits>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace FixConst;

/*-- parse fix rot all rotation/track nevery N thetas theta1 theta2 ... file outputfile.dat --*/

FixRotationTrack::FixRotationTrack(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
    nevery = 1;
    theta_T = {MY_PI4}; // pi/4
    output_file = "phitrajectory.dat";

    int iarg = 3; // first argument after fix ID group 

    while (iarg < narg) {
        if (strcmp(arg[iarg], "every") == 0){

            if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix rotation/track every", error);
            nevery = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
            if (nevery <= 0) error->all(FLERR, "Illegal fix rotation/track every value: {}", nevery);
            iarg += 2;

        } else if (strcmp(arg[iarg], "thetas") == 0){

            theta_T.clear();
            iarg += 1;
            while (iarg < narg && strcmp(arg[iarg], "file") != 0 && strcmp(arg[iarg], "every") != 0){
                theta_T.push_back(utils::numeric(FLERR, arg[iarg], false, lmp));
                iarg += 1;
            }
            if (theta_T.empty()) utils::missing_cmd_args(FLERR, "fix rotation/track thetas", error);

        } else if (strcmp(arg[iarg], "file") == 0){

            if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix rotation/track file", error);
            output_file = arg[iarg + 1];
            iarg += 2;

        } else {error->all(FLERR, "Illegal fix rotation/track keyword: {}", arg[iarg]);}
    }

    n_theta = static_cast<int>(theta_T.size());
    for (double theta : theta_T){
        if (theta < 0.0) error->all(FLERR, "Illegal fix rotation/track thetas value: {}", theta);
        if (theta == 0.0) error->warning(FLERR, "thetas value: {} correspond to the integral method", theta);
        if (theta >= MY_PI) error->warning(FLERR, "thetas value: {} is >= pi and correspond to the Euler method", theta);
    }
}

/*-- when to execute --*/

int FixRotationTrack::setmask()
{
    int mask = 0;
    mask |= END_OF_STEP;
    return mask;
}

/*-- verifications before running --*/

void FixRotationTrack::init()
{
    // la simulation a-t-elle des molécules ?
    if (!atom->molecule_flag)
        error->all(FLERR, "Fix rotation/track requires molecule IDs (atom_style with molecules)");

    // la table des atomes existe-t-elle ?
    if (atom->map_style == Atom::MAP_NONE)
        error->all(FLERR, "Fix rotation/track requires an atom map, see atom_modify");

    // le groupe doit contenir tous les atomes
    if (group->count(igroup) != atom->natoms)
        error->all(FLERR, "Fix rotation/track requires a group containing all atoms");

    // n_mol = plus grand ID de molécule
    tagint *molecule = atom->molecule;
    int nlocal = atom->nlocal;

    tagint local_max = 0;
    int local_bad = 0;
    for (int i = 0; i < nlocal; i++) {
        if (molecule[i] < 1) local_bad = 1;
        if (molecule[i] > local_max) local_max = molecule[i];
    }

    int global_bad;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, world);
    if (global_bad)
        error->all(FLERR, "Fix rotation/track: every atom must belong to a molecule (ID >= 1)");

    tagint global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_LMP_TAGINT, MPI_MAX, world); // & donne l'adresse d'une variable
    n_mol = static_cast<int>(global_max);

    // construit table des types 
    std::vector<int> local_count(3*n_mol, 0);
    std::vector<int> global_count(3*n_mol, 0);
    std::vector<tagint> local_tags(3*n_mol, 0);

    int *type = atom->type;
    tagint *tag = atom->tag;

    local_bad = 0;
    for (int i = 0; i < nlocal; i++){
        int t = type[i];
        if (t < 1 || t > 3) { local_bad = 1; continue;} // type hors plage continue

        int idx = 3 * (molecule[i] - 1) + (t - 1);
        local_count[idx] += 1;
        local_tags[idx] = tag[i];
    }

    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, world);
    if (global_bad) error->all(FLERR, "Fix rotation/track: atom types must be 1, 2 or 3");

    MPI_Allreduce(local_count.data(), global_count.data(), 3 * n_mol, MPI_INT, MPI_SUM, world);

    for (int idx = 0; idx < 3 * n_mol; idx++) {
        if (global_count[idx] != 1) {
            int m = idx / 3 + 1; // division entière par 3
            int t = idx % 3 + 1; // reste de la division euclidienne par 3
            error->all(FLERR, "Fix rotation/track: molecule {} does not have type {} atoms", m, t);
        }
    }
    mol_tag.assign(3 * n_mol, 0);
    MPI_Allreduce(local_tags.data(), mol_tag.data(), 3 * n_mol, MPI_LMP_TAGINT, MPI_SUM, world);

}

/*-- at the beginning of each run, declare variables and first call --*/

void FixRotationTrack::setup(int /*vflag*/)
{
    // tcheck if initialization already passed or not, in the case of multiple "run"
    if (!initialized) {
        R_bodyframe.assign(9 * n_mol, 0.0); // 3*3 matrix and one per molecule
        R_ref.assign(9 * n_mol * n_theta, 0.0); // 3*3 matrix and one per molecule per threshold followed
        phi_acc.assign(3 * n_theta * n_mol, 0.0); // vector in 3D one per molecule per threshold (accumulated rotation)
        phi_current.assign(3 * n_theta * n_mol, 0.0); // vector in 3D one per molecule per threshold (current rotation between R_Ref and R_bodyframe)
        phi_total.assign(3 * n_theta * n_mol, 0.0);

        compute_all_bodyframes();

        for (int i = 0; i < n_theta; i++) // i++ = fait passer i à i+1 à la fin de la boucle
            for (int j = 0; j < 9 * n_mol; j++)
                R_ref[i * 9 * n_mol + j] = R_bodyframe[j]; // i * 9 * n_mol --> ligne correspondante au seuil i

        initialized = true;
    }
}

/*-- compute bodyframes at a given time for all molecules --*/

void FixRotationTrack::body_frame(double *e1, double *v, double *R)
{
    double e2[3];
    double e3[3];

    MathExtra::norm3(e1);
    MathExtra::cross3(e1,v,e2);
    MathExtra::norm3(e2);
    MathExtra::cross3(e1,e2,e3);

    for (int c = 0; c < 3; c++) {
        R[c]     = e1[c];     // colonne 0
        R[3 + c] = e2[c];     // colonne 1
        R[6 + c] = e3[c];     // colonne 2
    }
}

void FixRotationTrack::compute_all_bodyframes()
{
    std::vector<double> local_R(9 * n_mol, 0.0);

    double **x = atom->x;
    int nlocal = atom->nlocal; // nombre d'atom sur le rang local

    for (int m = 0; m < n_mol; m++){
        int i1 = atom->map(mol_tag[3 * m]); // définit dans init, donne idx atome de type1 de la molécule m
        if (i1 < 0 || i1 > nlocal - 1) continue;

        int i2 = atom->map(mol_tag[3 * m + 1]);
        int i3 = atom->map(mol_tag[3 * m + 2]);
        if (i2 < 0 || i3 < 0){
            error->one(FLERR,"atoms of molecule {} are not all visible on this rank; try comm_modify cutoff ", m + 1);
        }

        double e1[3];
        double v[3];

        MathExtra::sub3(x[i2], x[i1], e1);
        MathExtra::sub3(x[i3], x[i1], v);
        
        domain->minimum_image(e1[0], e1[1], e1[2]);
        domain->minimum_image(v[0], v[1], v[2]);

        body_frame(e1, v, &local_R[9 * m]);
    }
    MPI_Allreduce(local_R.data(), R_bodyframe.data(), 9 * n_mol, MPI_DOUBLE, MPI_SUM, world);
}

void FixRotationTrack::rotation_vector(const double *dR, double *w)
{
    double cos_theta = std::min(1.0, std::max(-1.0, (dR[0] + dR[4] + dR[8] - 1.0) / 2.0));

    double ax;
    double ay;
    double az;
    ax = dR[5] - dR[7];
    ay = dR[6] - dR[2];
    az = dR[1] - dR[3];
    double sin_theta = std::sqrt(ax*ax + ay*ay + az*az) / 2.0;

    double theta;

    if (sin_theta > std::sqrt(std::numeric_limits<double>::epsilon())){
        theta = std::atan2(sin_theta,cos_theta);
        double inv_2s = theta / (2 * sin_theta);
        w[0] = ax * inv_2s;
        w[1] = ay * inv_2s;
        w[2] = az * inv_2s;
    } else if (cos_theta > 0) {
        w[0] = 0;
        w[1] = 0;
        w[2] = 0;
    } else {
        double n[3];
        if (dR[0] >= dR[4] && dR[0] >= dR[8]){
            double nx = std::sqrt(std::max((dR[0]+1)/2, 0.0));
            n[0] = nx;
            n[1] = (dR[3] + dR[1])/(4 * nx);
            n[2] = (dR[6] + dR[2])/(4 * nx);
        } else if (dR[4] >= dR[8]) {
            double ny = std::sqrt(std::max((dR[4]+1)/2, 0.0));
            n[0] = (dR[3] + dR[1])/(4 * ny);
            n[1] = ny;
            n[2] = (dR[5] + dR[7])/(4 * ny);           
        } else {
            double nz = std::sqrt(std::max((dR[8]+1)/2, 0.0));
            n[0] = (dR[6] + dR[2])/(4 * nz);
            n[1] = (dR[5] + dR[7])/(4 * nz);
            n[2] = nz;
        }
        theta = std::atan2(sin_theta, cos_theta);
        MathExtra::norm3(n);
        w[0] = theta * n[0];
        w[1] = theta * n[1];
        w[2] = theta * n[2];
    }
}

void FixRotationTrack::end_of_step()
{
    compute_all_bodyframes();

    for (int k = 0; k < n_theta; k++){
        for (int m = 0; m < n_mol; m++){
            int idx = k * n_mol + m; // début de la molécule m pour k-ième theta_T 

            double *Rref = &R_ref[9 * idx]; // point vers la matrice 3*3 associé à cette molécule à ce theta_T
            double *Rbody = &R_bodyframe[9 * m];
            double *cur = &phi_current[3 * idx];
            double *acc = &phi_acc[3 * idx];
            double *tot = &phi_total[3 * idx];

            double dR[9];

            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j ++){
                    double S = 0;
                    for (int a = 0; a < 3; a++){
                        S += Rref[a + 3 * i] * Rbody[a + 3 * j]; 
                    }
                    dR[i + 3 * j] = S;
                }
            }
        
            rotation_vector(dR,cur);

            for (int c = 0; c < 3; c++) tot[c] = acc[c] + cur[c];

            if (MathExtra::len3(cur) > theta_T[k]){
                for (int c = 0; c < 3; c++) acc[c] = tot[c];
                for (int c = 0; c < 9; c++) Rref[c] = Rbody[c];
            }
        }
    }
}