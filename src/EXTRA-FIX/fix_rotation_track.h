/* -*- c++ -*- ------ ... ------ */

#ifdef FIX_CLASS // if FIX_CLASS is define

FixStyle(rotation/track,FixRotationTrack);

#else 

#ifndef LMP_FIX_ROTATION_TRACK_H // if LMP_FIX not define define it and ignore next occurence
#define LMP_FIX_ROTATION_TRACK_H

#include "fix.h"
#include <vector>
#include <string>
#include <cstdio>

namespace LAMMPS_NS {

class FixRotationTrack : public Fix {
    public:
    FixRotationTrack(class LAMMPS *, int, char **);
    int setmask() override; // quand lammps doit appeler
    void init() override; // appelé au début de chaque run (vérifications tc)
    void setup(int) override; // appelé juste après calcul des forces (bon endroit pour lire les positions et initialiser R_ref)
    void end_of_step() override; // à chaque nevery demandé
    // void write_restart(FILE *) override;
    // void restart(char *) override;
    ~FixRotationTrack() override;

    protected:
    int n_mol = 0;
    int n_theta = 0;
    bool initialized = false; 
    std::vector<double> theta_T; // longueur n_theta
    std::vector<double> R_ref; // longueur 9*n_theta*n_mol
    std::vector<double> R_bodyframe; // longueur 9*n_mol
    std::vector<double> phi_acc; // longueur 3*n_theta*n_mol
    std::vector<double> phi_current; // longueur 3*n_theta*n_mol
    std::vector<double> phi_total; // longueur 3 * n_theta * n_mol
    std::vector<tagint> mol_tag; // longueur 3*n_mol
    std::string output_file;
    std::vector<FILE *> files;

    // functions for rotations calculations
    void body_frame(double *e1, double *v, double *R);
    void compute_all_bodyframes();
    void rotation_vector(const double *dR, double *w);
    void write_frame();
};

} // namespace LAMMPS_NS

#endif // close #ifndef
#endif // close #ifdef FIX_CLASS