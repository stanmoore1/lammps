#ifdef FIX_CLASS

FixStyle(cluster_switch,FixClusterSwitch)

#else

#ifndef LMP_FIX_CLUSTER_SWITCH_H
#define LMP_FIX_CLUSTER_SWITCH_H

#include "fix.h"
#include <map>

namespace LAMMPS_NS {
  class FixClusterSwitch : public Fix {
  public:
    FixClusterSwitch(class LAMMPS *, int, char **);
    ~FixClusterSwitch();
    int setmask();
    void allocate();
    void init();
    void init_list(int, class NeighList *);
    void pre_exchange();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    double memory_usage();
    double compute_vector(int);


  private:
    int confirm_molecule(tagint); // checks molID state (returns 1 for ON and 0 for off)
    int switch_flag(int); // uses random to decided if this molID should switch state (returns 1 for YES)
    void read_file(char *);
    void read_contacts(char *);
    void attempt_switch(); // where all the MC switching happens
    void check_cluster(); // checks recursively molecules part of central cluster and updates mol_restrict
    void computecross(double *, double *, double *); // auxilliary functions cross product
    void normalize(double *); // auxilliary functions vector normalization
    void check_arrays(); // make sure all mol arrays are properly communicated and have the right information
    void gather_statistics(); //uses newly communicated mol arrays to gather MC statistics

    std::map<tagint,int> *hash; // hash map (key value) to keep track of mols

    int me, nprocs;
    int nlevels_respa;
    int allocate_flag; // flag that controls memory allocation
    int pack_flag; //flag that controls which information is sent via comms
    //    int nlocal, nall; // number of atoms local to this processor, and including ghost atoms
    bigint ngroup; // number of atoms in group

    //int groupbit; //bitmask for group
    
    double probON; // probability of state 1 (ON)
    double probOFF; // probabilty of state 2 (OFF) = 1.0 - probON
    int nSwitchTypes; // number of atom types associated with state switching
    int nSwitchPerMol; // number of atoms associated with state switching per mol
    int *atomtypesON; // atom types associated with ON
    int *atomtypesOFF; // atom types associated with OFF
    int ***contactMap; // atom types associated with successful intermolecule contact
    int nContactTypes; // number of contact conditions to search
    int nAtomsPerContact; // number of atoms within each contact condition
    int switchFreq; // number of timesteps between switching
    double nAttemptsTotal; // number of swap attempts which should equal numMols * (steps/switchFreq)
    double nSuccessTotal; // number of swap successes 
    double nAttemptsON; // number of swap attempts to ON
    double nAttemptsOFF; // number of swap attempts to OFF
    double nSuccessON; // number of swap successes to ON
    double nSuccessOFF; // number of swap successes to OFF
    double nCluster; // number of mols in cluster

    class RanPark *random_equal; //rand num gen (same across procs)
    class RanPark *random_unequal; // rand num gen (diff across procs)
    FILE *fp1; // debug file1
    FILE *fp2; // debug file2


    int nmol; // number of molecules in group that are open to switching
    int maxmol; // maximum mol number of group to size array
    int mol_seed; // molecule ID for 'seed' of cluster - this molecule is always ON
    int mol_offset; // molecule ID (negative shift) to connect molecules for cluster - but outside of switching
    int *mol_restrict; // list of molecule ID tags that are open to switching (mol cluster can consider more mols than mol_restrict)
    //mol_restrict should be updated such that mols part of cluster are turned off to switching
    //is equal to 1 when open to switching, otherwise -1
    class NeighList *list;
    
    tagint **mol_atoms; // 2D list of (mol ID), (internal atom type) = atom ID tag
    int *mol_state; //list of molecules current state (0 = off, 1 = on)
    int *mol_accept; //list of switching decisions for each molecule (-1 initial, 0 fail, 1 switch true
    int *mol_cluster; //list of cluster IDs for each molecule
    double cutsq; // cutoff for contact map

    //    double *atom_state; //per-atom vector to output state of atom, i.e. working copy of vector_atom for the fix
    inline int sbmask(int j) {
      return j >> SBBITS & 3;
    }//similar to pair.h
  protected:
    int nmax;
  };
}
#endif
#endif
