/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Xiang Gu (USF)
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "fix_shock_info.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "force.h"
#include "lattice.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
//#include "pair_rebo.h"
#include "memory.h"
#include "error.h"

#define cpnts_of_rho	1
#define cpnts_of_v	3
#define cpnts_of_epot	1
#define cpnts_of_ekin	4
#define cpnts_of_etot	1
#define cpnts_of_T	4

#define densfactor	1.66053
#define velfactor	0.1
// Pressure factor eV/A^3 ==> GPa
#define pressfactor     160.219


using namespace LAMMPS_NS;
using namespace FixConst;

enum{LOWER,CENTER,UPPER,COORD};
enum{BOX,LATTICE,REDUCED};

/* ---------------------------------------------------------------------- */

FixShockInfo::FixShockInfo(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 13 )
    error->all(FLERR,"Illegal fix shock/info command: not enough arguments");

  nevery = atoi(arg[3]);
  nfreq = atoi(arg[4]);
  nrepeat = atoi(arg[5]);

  if (strcmp(arg[6],"x") == 0)  dim = 0;
  else if (strcmp(arg[6],"y") == 0)  dim = 1;
  else if (strcmp(arg[6],"z") == 0)  dim = 2;
  else error->all(FLERR,"Illegal parameter of dim for fix shock/info command");

  if (strcmp(arg[7],"lower") == 0)  originflag = LOWER;
  if (strcmp(arg[7],"center") == 0)  originflag = CENTER;
  if (strcmp(arg[7],"upper") == 0)  originflag = UPPER;
  else originflag = COORD;
  if (originflag == COORD)  origin = atof(arg[7]);

  delta = atof(arg[8]);
  nmin = atoi(arg[9]);

  int n = strlen(arg[10]) + 1;
  id_compute_pe = new char[n];
  strcpy(id_compute_pe,arg[10]);

  n = strlen(arg[11]) + 1;
  id_compute_stress = new char[n];
  strcpy(id_compute_stress,arg[11]);

  n = strlen(arg[12]) + 1;
  einfo_fileprefix = new char[n];
  strcpy(einfo_fileprefix,arg[12]);

  n = strlen(arg[13]) + 1;
  stress_fileprefix = new char[n];
  strcpy(stress_fileprefix,arg[13]);

  // get datafile ready for write-in

  MPI_Comm_rank(world,&me);

  // parse optional args

  int scaleflag = BOX;

  int iarg = 14;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal setup of units for fix shock/info command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = BOX;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = LATTICE;
      else if (strcmp(arg[iarg+1],"reduced") == 0) scaleflag = REDUCED;
      else error->all(FLERR,"Illegal setup of units for fix shock/info command");
      iarg += 2;
    } else error->all(FLERR,"Illegal setup of units for fix shock/info command");
  }

  // setup scaling

//  int triclinic = domain->triclinic;
//  if (triclinic == 1 && scaleflag != REDUCED)
//    error->all(FLERR,"Fix shock/info for triclinic boxes requires units reduced");

  int triclinic = domain->triclinic;
  if (triclinic == 1 && scaleflag != REDUCED)
    error->all(FLERR,"Fix shock/info for triclinic boxes requires units reduced");

  if (scaleflag == LATTICE && domain->lattice == NULL)
    error->all(FLERR,"Use of fix shock/info with undefined lattice");

  if (scaleflag == LATTICE) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else xscale = yscale = zscale = 1.0;

  // apply scaling factors

  double scale;
  if (dim == 0)  scale = xscale;
  if (dim == 1)  scale = yscale;
  if (dim == 2)  scale = zscale;
  delta *= scale;
  if (originflag == COORD)  origin *= scale;

  // setup and error check

  if (nevery <= 0)  
    error->all(FLERR,"Illegal fix shock/info command: nevery must be positive and non-zero");
  if (nfreq < nevery || nfreq % nevery)
    error->all(FLERR,"Illegal fix shock/info command: nfreq must be a multiple of nevery");
  if (nrepeat <= 0 || nevery * nrepeat > nfreq)
    error->all(FLERR,"Illegal fix shock/info command: nrepeat should not be negative or too big");

  if (delta <= 0.0)
    error->all(FLERR,"Illegal fix shock/info command: delta mus be positive");
  invdelta = 1.0/delta;


  int icompute_pe = modify->find_compute(id_compute_pe);
  if (icompute_pe < 0)
    error->all(FLERR,"Compute ID of atomic epot for fix shock/info does not exist");
  if (modify->compute[icompute_pe]->peratom_flag == 0)
    error->all(FLERR,"Compute ID of atomic epot for fix shock/info does not calculate per-atom info");

  int icompute_stress = modify->find_compute(id_compute_stress);
  if (icompute_stress < 0)
    error->all(FLERR,"Compute ID of atomic stress tensor for fix shock/info does not exist");
  if (modify->compute[icompute_stress]->peratom_flag == 0)
    error->all(FLERR,"Compute ID of atomic stress tensor for fix shock/info does not calculate per-atom info");

  stress_size_peratom = 6;  //modify->compute[icompute_stress]->size_peratom;
  cpnts_noT = cpnts_of_rho + cpnts_of_v + cpnts_of_epot + cpnts_of_ekin + cpnts_of_etot;
  cpnts_all = cpnts_of_T + cpnts_noT;
  nvalues = cpnts_all + 3 * stress_size_peratom;

  nsum = nlayers = maxlayer = 0;
  coord = NULL;
  count_one = count_many = count_total = NULL;
  values_one = values_many = values_total = NULL;
  variable_bin = NULL;
}


/* ---------------------------------------------------------------------- */

FixShockInfo::~FixShockInfo()
{
  memory->sfree(coord);
  memory->sfree(count_one);
  memory->sfree(count_many);
  memory->sfree(count_total);
  memory->destroy(values_one);
  memory->destroy(values_many);
  memory->destroy(values_total);
  memory->sfree(variable_bin);
}

/* ---------------------------------------------------------------------- */

int FixShockInfo::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixShockInfo::init()
{
  int icompute_pe = modify->find_compute(id_compute_pe);
  if (icompute_pe < 0)
    error->all(FLERR,"Compute ID of atomic epot for fix shock/info does not exist");
  compute_pe = modify->compute[icompute_pe];

  int icompute_stress = modify->find_compute(id_compute_stress);
  if (icompute_stress < 0)
    error->all(FLERR,"Compute ID of atomic stress tensor for fix shock/info does not exist");
  compute_stress = modify->compute[icompute_stress];

  if (compute_pe->id) {
    icompute_pe = modify->find_compute(compute_pe->id);
    if (icompute_pe < 0)
      error->all(FLERR,"Fix shock/info needs a Precompute but the corresponding ID does not exist");
    precompute_pe = modify->compute[icompute_pe];
  } else precompute_pe = NULL;

  if (compute_stress->id) {
    icompute_stress = modify->find_compute(compute_stress->id);
    if (icompute_stress < 0)
      error->all(FLERR,"Fix shock/info needs a Precompute but the corresponding ID does not exist");
    precompute_stress = modify->compute[icompute_stress];
  } else precompute_stress = NULL;
}

/* ---------------------------------------------------------------------- */

void FixShockInfo::end_of_step()
{
  int i,j,m,ilayer;
  double lo,hi;

  // skip this procedure if not the right timing to do averaging

  // Fixed by You Lin
  if ( update->ntimestep % nfreq >= nfreq - nevery * nrepeat)  {
    //((PairREBO *) force->pair)->uuflag = 1;
    //((PairREBO *) force->pair)->eeflag = 1;
  }
  else  {
    //((PairREBO *) force->pair)->uuflag = 0;
    //((PairREBO *) force->pair)->eeflag = 0;
  }

  if ( (update->ntimestep - 1) % nfreq < nfreq - nevery * nrepeat )  return;

  if (nsum == 0)  {
    double *boxlo,*boxhi,*prd;
    if (scaleflag == REDUCED) {
      boxlo = domain->boxlo_lamda;
      boxhi = domain->boxhi_lamda;
      prd = domain->prd_lamda;
    } else {
      boxlo = domain->boxlo;
      boxhi = domain->boxhi;
      prd = domain->prd;
    }

    if (originflag == LOWER)  origin = boxlo[dim];
    else if (originflag == UPPER)  origin = boxhi[dim];
    else if (originflag == CENTER)
      origin = 0.5 * (boxlo[dim] + boxhi[dim]);

    if (origin < domain->boxlo[dim])  {
      m = static_cast<int> ((domain->boxlo[dim] - origin) * invdelta);
      lo = origin + m*delta;
    } else {
      m = static_cast<int> ((origin - domain->boxlo[dim]) * invdelta);
      lo = origin - m*delta;
      if (lo > domain->boxlo[dim]) lo -= delta;
    }
    if (origin < domain->boxhi[dim])  {
      m = static_cast<int> ((domain->boxhi[dim] - origin) * invdelta);
      hi = origin + m*delta;
      if (hi < boxhi[dim]) hi += delta;
    } else {
      m = static_cast<int> ((origin - domain->boxhi[dim]) * invdelta);
      hi = origin - m*delta;
    }

    offset = origin + delta;  // lo;  //  + delta;
    nlayers = static_cast<int> ((hi-offset) * invdelta + 0.5);
    double volume = domain->xprd * domain->yprd * domain->zprd;
    layer_volume = delta * volume / prd[dim];

//    printf("domain->xprd = %.8f    domain->yprd = %.8f    delta = %.8f\n",
//           domain->xprd, domain->yprd, delta);

    if (nlayers+1 > maxlayer) {
      maxlayer = nlayers+1;
      coord = (double *) memory->srealloc(coord,(nlayers+1)*sizeof(double),
                      "shock/info:coord");
      count_one = (double *) memory->srealloc(count_one,(nlayers+1)*sizeof(double),
                          "shock/info:count_one");
      count_many = (double *) memory->srealloc(count_many,(nlayers+1)*sizeof(double),
                           "shock/info:count_many");
      count_total = (double *) memory->srealloc(count_total,(nlayers+1)*sizeof(double),
                            "shock/info:count_total");
      values_one = memory->grow(values_one,nlayers+1,nvalues,
                                                "shock/info:values_one");
      values_many = memory->grow(values_many,nlayers+1,nvalues,
                                                 "shock/info:values_many");
      values_total = memory->grow(values_total,nlayers+1,nvalues,
                                                  "shock/info:values_total");
      variable_bin = (double *) memory->srealloc(variable_bin,(nlayers+1)*sizeof(double),
                             "shock/info:variable_bin");
    }

    for (m = 0; m < nlayers+1; m++)  {
      coord[m] = offset + (m+0.5)*delta;
      variable_bin[m] = 0.0;
      count_many[m] = count_total[m] = 0.0;
      for (j = 0; j < nvalues; j++)  values_many[m][j] = values_total[m][j] = 0.0;
    }
  }

  // zero out arrays for one sample

  nsum++;
  for (m = 0; m < nlayers+1; m++)  {
    count_one[m] = 0.0;
    for (j = 0; j < nvalues; j++)  values_one[m][j] = 0.0;
  }

  double mvv2e = force->mvv2e;
  double Smass,KinStressbuf[6];
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;


  if (scaleflag == REDUCED)  domain->x2lamda(nlocal);

  if (precompute_pe)  precompute_pe->compute_peratom();
  compute_pe->compute_peratom();
  double *scalar = compute_pe->vector_atom;

  if (precompute_stress)  precompute_stress->compute_peratom();
  compute_stress->compute_peratom();
  double **vector = compute_stress->array_atom;

  // count_one[ilayer] = total number of atom in the slab
  // values_one[ilayer][0] = total mass of the slab (in atomic unit)
  // values_one[ilayer][1-3] = total velocity of the slab
  // values_one[ilayer][4] = total potential energy of the slab
  // values_one[ilayer][5-7] = total kinetic energy from x,y,z of the slab
  // values_one[ilayer][8] = total kinetic energy of the slab
  // values_one[ilayer][9] = total energy of the slab
  // values_one[ilayer][14-19] = total potential stress of the slab ?
  // values_one[ilayer][20-25] = total kinetic stress of the slab
  // values_one[ilayer][26-31] = total stress of the slab ?
  for (i = 0; i < nlocal; i++)  {
    if (mask[i] & groupbit)  {
      ilayer = static_cast<int> ((x[i][dim] - offset) * invdelta);
      if (ilayer < 0)  continue;  // ilayer = 0;
      if (ilayer > nlayers)  ilayer = nlayers;
      count_one[ilayer] += 1.0;

      if (mass)  values_one[ilayer][0] += mass[type[i]];
      else  values_one[ilayer][0] += rmass[i];

      values_one[ilayer][1] += v[i][0];
      values_one[ilayer][2] += v[i][1];
      values_one[ilayer][3] += v[i][2];

      values_one[ilayer][4] += scalar[i];

      if (mass)  values_one[ilayer][5] += mvv2e * mass[type[i]] * v[i][0] * v[i][0];
      else  values_one[ilayer][5] += mvv2e * rmass[i] * v[i][0] * v[i][0];

      if (mass)  values_one[ilayer][6] += mvv2e * mass[type[i]] * v[i][1] * v[i][1];
      else  values_one[ilayer][6] += mvv2e * rmass[i] * v[i][1] * v[i][1];

      if (mass)  values_one[ilayer][7] += mvv2e * mass[type[i]] * v[i][2] * v[i][2];
      else  values_one[ilayer][7] += mvv2e * rmass[i] * v[i][2] * v[i][2];

      for (j = cpnts_all; j < cpnts_all+stress_size_peratom; j++)
        values_one[ilayer][j] += vector[i][j-cpnts_all];  //0.0;

      if (mass)  Smass = mvv2e * mass[type[i]];
      else  Smass = mvv2e * rmass[i];

      KinStressbuf[0] = Smass * v[i][0]*v[i][0];
      KinStressbuf[1] = Smass * v[i][1]*v[i][1];
      KinStressbuf[2] = Smass * v[i][2]*v[i][2];
      KinStressbuf[3] = Smass * v[i][0]*v[i][1];
      KinStressbuf[4] = Smass * v[i][0]*v[i][2];
      KinStressbuf[5] = Smass * v[i][1]*v[i][2];

      for (j = cpnts_all+stress_size_peratom; j < cpnts_all+2*stress_size_peratom; j++)
        values_one[ilayer][j] -= KinStressbuf[j-cpnts_all-stress_size_peratom];  //0.0;
    }
  }


  for (m = 0; m < nlayers+1; m++)  {
    values_one[m][5] *= 0.5;
    values_one[m][6] *= 0.5;
    values_one[m][7] *= 0.5;
    values_one[m][8] = values_one[m][5] + values_one[m][6] + values_one[m][7];
    values_one[m][9] = values_one[m][4] + values_one[m][8];
  }

  for (m = 0; m < nlayers+1; m++)
    for (j = cpnts_all+2*stress_size_peratom; j < nvalues; j++)
      values_one[m][j] = values_one[m][j-2*stress_size_peratom] + values_one[m][j-stress_size_peratom];  //0.0;


  if (scaleflag == REDUCED)  domain->lamda2x(nlocal);

  // Accumulate count_one to count_many during each step until nfreq steps passed
  for (m = 0; m < nlayers+1; m++)  {
    count_many[m] += count_one[m];
    for (j = 0; j < cpnts_noT; j++)  values_many[m][j] += values_one[m][j];
    for (j = cpnts_all; j < nvalues; j++)  values_many[m][j] += values_one[m][j];
  }


  if (update->ntimestep % nfreq == 0)  {
    // When update, accumulate all count_many to count_total of single processor
    MPI_Allreduce(count_many,count_total,nlayers+1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&values_many[0][0],&values_total[0][0],nvalues*(nlayers+1),MPI_DOUBLE,MPI_SUM,world);

//    if (update->ntimestep == 5000)  {
//      fp = fopen("FinerStructure.data", "w");
//      for (m = 0; m < nlayers+1; m++)  {
//        fprintf(fp, "%.8f\t%14.6f\t%18.10f\n", 
//                coord[m], count_total[m] / nsum, values_total[m][0] * densfactor / (nsum * layer_volume));
//      }
//      fclose(fp);
//    }
    for (m = 0; m < nlayers+1; m++)  {
      count_one[m] = count_many[m] = 0.0;
      for ( j = 0; j < cpnts_noT; j++)  values_one[m][j] = values_many[m][j] = 0.0;
      for (j = cpnts_all; j < nvalues; j++)  values_one[m][j] = values_many[m][j] = 0.0;
    }

    int layer,bin,coordswitch;
    double firstcoord, lastcoord;
    double frac;

    layer = bin = coordswitch = 0;
    firstcoord = lastcoord = 0.0;
    frac = 0.0;

    // Vary bin magic nsum -> number of steps passed since last update
    for (layer = 0; layer < nlayers; layer++)  {
      // count_one[bin] now contains the number of atoms over layers
      count_one[bin] += count_total[layer];
      for (j = 0; j < cpnts_noT; j++)
        values_one[bin][j] += values_total[layer][j];

      for (j = cpnts_all; j < nvalues; j++)
        values_one[bin][j] += values_total[layer][j];

      // skip when the first layer of new bin is empty
      if (!coordswitch)  {    // if (count_one[bin] > 1.e-12 && !coordswitch)  \{
        coordswitch = 1;
        firstcoord = coord[layer];
      }

      if (coordswitch)  variable_bin[bin] += 1.0;

      if ( (count_one[bin]+count_total[layer+1] > nmin * nsum + 1.e-9)
           || (layer == nlayers-1) )  {
        if (bin == 0)  coord[bin] = 0.0;

        if ( (fabs(count_one[bin] - nmin * nsum) < 1.e-9) || (layer == nlayers-1) )  frac = 0.0;
        else  frac = (nmin * nsum - count_one[bin]) / count_total[layer+1];
        // frac contains the fraction of next layer that is needed to fill current bin


        lastcoord = coord[layer];
        coord[bin] += 0.5 * (lastcoord + firstcoord);

        variable_bin[bin] += frac;
        count_one[bin] += frac * count_total[layer+1];
        coord[bin] += 0.5 * frac * delta;

        for (j = 0; j < cpnts_noT; j++)
          values_one[bin][j] += frac * values_total[layer+1][j];

        for (j = cpnts_all; j < nvalues; j++)
          values_one[bin][j] += frac * values_total[layer+1][j];


        variable_bin[bin+1] -= frac;
        count_one[bin+1] -= frac * count_total[layer+1];
        coord[bin+1] = 0.5 * frac * delta;

        for (j = 0; j < cpnts_noT; j++)
          values_one[bin+1][j] -= frac * values_total[layer+1][j];

        for (j = cpnts_all; j < nvalues; j++)
          values_one[bin+1][j] -= frac * values_total[layer+1][j];


        bin++;
        coordswitch = 0;
      }
    }

//    if ( coordswitch == 1 )  {
//      coord[bin] = firstcoord ;
//      bin++;
//    }

    nlayers = bin;


    double boltz = force->boltz;

    // count_one, values_one contain information of sum over steps and processors
  // count_one[ilayer] = total number of atom in the slab
  // values_one[ilayer][0] = total mass of the slab (in atomic unit)
  // values_one[ilayer][1-3] = total velocity of the slab
  // values_one[ilayer][4] = total potential energy of the slab
  // values_one[ilayer][5-7] = total kinetic energy from x,y,z of the slab
  // values_one[ilayer][8] = total kinetic energy of the slab
  // values_one[ilayer][9] = total energy of the slab
  // values_one[ilayer][14-19] = total potential stress of the slab ?
  // values_one[ilayer][20-25] = total kinetic stress of the slab
  // values_one[ilayer][26-31] = total stress of the slab ?
  // values_total[ilayer][10-12] = total kinetic energy from x,y,z of the slab subtracted by the kinetic energy of the center of mass
  // values_total[ilayer][13] = total kinetic energy of the slab subtracted by the kinetic energy of the center of mass

    for (m = 0; m < nlayers; m++)  {
      count_total[m] = count_one[m] / nsum; // average over steps
      if (count_one[m] > 1.e-12)  {
        values_total[m][0] = values_one[m][0] / (nsum * layer_volume * variable_bin[m]);
        values_total[m][0] *= densfactor;
      }
      else  values_total[m][0] = 0.0;

      for (j = 1; j < cpnts_noT; j++)  {
        if (count_one[m] > 1.e-12)  {
          values_total[m][j] = values_one[m][j] / count_one[m];  // energy per atom per step
        }
        else  values_total[m][j] = 0.0;
      }

      for (j = cpnts_all; j < nvalues; j++)  {
        if (count_one[m] > 1.e-12)  {
          values_total[m][j] = values_one[m][j] / (nsum * layer_volume * variable_bin[m]); // stress per volume per step
          values_total[m][j] *= pressfactor;
        }
        else  values_total[m][j] = 0.0;
      }

      if (count_one[m] > 1.e-12)  {
        values_total[m][10] = values_one[m][5]
                            - 0.5 * mvv2e * values_one[m][0] * values_total[m][1] * values_total[m][1];
        values_total[m][10] /= 0.5 * boltz * count_one[m];

        values_total[m][11] = values_one[m][6]
                            - 0.5 * mvv2e * values_one[m][0] * values_total[m][2] * values_total[m][2];
        values_total[m][11] /= 0.5 * boltz * count_one[m];

        values_total[m][12] = values_one[m][7]
                            - 0.5 * mvv2e * values_one[m][0] * values_total[m][3] * values_total[m][3];
        values_total[m][12] /= 0.5 * boltz * count_one[m];

        values_total[m][13] = (values_total[m][10] + values_total[m][11] + values_total[m][12]) / 3.0;
      }
      else
        values_total[m][10] = values_total[m][11] = values_total[m][12] = values_total[m][13] = 0.0;

      for (j = 1; j <= 3; j++)  values_total[m][j] *= velfactor;
    }


    if (me == 0)  {
      char outputfilename[128];
      sprintf(outputfilename, "%s.%d", einfo_fileprefix,update->ntimestep);
      fp = fopen(outputfilename, "w");
      if (fp == NULL)  {
        char str[128];
        sprintf(str,"Cannot open output file %s for shock/info\n",outputfilename);
        error->one(FLERR,str);
      }

      fprintf(fp,"Spatial-averaged (Variable bins) data for Shock Wave Simulation with REBO potential:\n");
      fprintf(fp,"TimeStep \tNumber-of-layers (one per snapshot)\n");
      fprintf(fp,"%d \t\t%d\n",update->ntimestep,nlayers);
      fprintf(fp,"Layer# \t      Coord \t      #Atoms \t      #deltas \t      Dens \t      Vx \t      Vy \t      Vz ");
      fprintf(fp,"\t      Epot \t      Ekinx \t      Ekiny \t      Ekinz \t      Ekin \t      Etot ");
      fprintf(fp,"\t      Tx \t      Ty \t      Tz \t      T \n");

      for (m = 1; m < nlayers; m++)  {
        fprintf(fp," %d \t %10.5f\t %11.5f\t %10.4f\t",m+1,coord[m],count_total[m],variable_bin[m]);
        for (j = 0; j < cpnts_all; j++)  fprintf(fp," %14.8f\t",values_total[m][j]);
        fprintf(fp,"\n");
      }

      fclose(fp);


      sprintf(outputfilename, "%s.%d", stress_fileprefix,update->ntimestep);
      fp = fopen(outputfilename, "w");
      if (fp == NULL)  {
        char str[128];
        sprintf(str,"Cannot open output file %s for shock/stress\n",outputfilename);
        error->one(FLERR,str);
      }

      fprintf(fp,"Spatial-averaged (Variable bins) data for Shock Wave Simulation with REBO potential:\n");
      fprintf(fp,"TimeStep \tNumber-of-layers (one per snapshot)\n");
      fprintf(fp,"%d \t\t%d\n",update->ntimestep,nlayers);
      fprintf(fp,"Layer# \t      Coord \t      #Atoms \t      #deltas ");
      fprintf(fp, "\t      Dens \t    PxxPot \t    PyyPot \t    PzzPot \t    PxyPot \t    PxzPot \t    PyzPot ");
      fprintf(fp, "\t    PxxKin \t    PyyKin \t    PzzKin \t    PxyKin \t    PxzKin \t    PyzKin ");
      fprintf(fp, "\t    PxxTot \t    PyyTot \t    PzzTot \t    PxyTot \t    PxzTot \t    PyzTot \n");

      for (m = 1; m < nlayers; m++)  {
        fprintf(fp," %d \t %10.5f\t %11.5f\t %10.4f\t",m+1,coord[m],count_total[m],variable_bin[m]);
        fprintf(fp," %14.8f\t",values_total[m][0]);
        for (j = cpnts_all; j < nvalues; j++)
          fprintf(fp," %14.8f\t",-values_total[m][j]);
        fprintf(fp,"\n");
      }

      fclose(fp);
    }

    nsum = 0;
  }
}
