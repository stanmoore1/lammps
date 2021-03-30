/*----------------------------------------------------------------------
  Computation of the effective electronegativity for the QTPIE option
  Ref: Helgaker, Jorgensen Olsen (2000) Molecular Electronic -
       Structure Theory, Wiley 
  GTO: Eqs. (9.12.1), (9.2.41)
  GTO restriction: Eq. (9.12.2)
  ----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
  Attention 1: GTO exponents must be given in atomic units, i.e. 1/a_0,
               where a_0 is the Bohr radius. 
  ----------------------------------------------------------------------*/

/*----------------------------------------------------------------------
  Attention 2: system and control structures of reaxc_types do not
               update position and charges to the current timestep
               when qeq method is used. Thus, atom class is needed
               to compute the position dependent properties.
  ----------------------------------------------------------------------*/

#include "reaxc_chi_effective.h"
#include "pair.h"
#include "atom.h"
#include "error.h"
#include "neighbor.h"
#include "comm.h"
#include "lmptype.h"
#include <math.h>
#include <cstring>

#define k 10
#define zero 1.0e-50
#define Ang_to_bohrRad 1.8897259886      // 1 Ang = 1.8897259886 Bohr radius
#define eV_to_Hartree 0.0367493245       // 1 eV = 0.0367493245 Hartree

void calculate_chi_eff(qtpie_parameters *qtpie, LAMMPS_NS::Atom *atom, reax_system *system, double *chi,
                       int ni, int nj, double *lchi_eff)
{
  double R,a_min,OvIntMaxR,Voltage,Overlap,Nominator,Denominator;
  double ea,eb,chia,chib,p,m;
  double phia,phib;
  int i,j,type_i,type_j;
  int ntypes = atom->ntypes;
  int *type = atom->type;
  double **x = atom->x;

  // Use integral pre-screening for overlap calculations
  a_min = find_min(qtpie->gauss_exp,ntypes+1);
  OvIntMaxR = sqrt(pow(a_min,-1.)*log(pow(M_PI/(2.*a_min),3.)*pow(10.,2.*k)));

  if(qtpie->cutghost < OvIntMaxR/Ang_to_bohrRad) {
    char errmsg[256];
    snprintf(errmsg, 256,"qtpie/reax: using limit distance for overlap integral of %f Angstrom when max ghost atom distance is %f Angstrom. "
             "Increase the ghost atom cutoff with comm_modify.",OvIntMaxR/Ang_to_bohrRad,qtpie->cutghost);
    system->error_ptr->all(FLERR,errmsg);
  }

  for (i = 0; i < ni; i++) {

    type_i = type[i];
    ea = qtpie->gauss_exp[type_i];
    chia = chi[type_i];

    Nominator = Denominator = 0.0;

    for (j = 0; j < nj; j++) {

      R = Distance(x[i],x[j])*Ang_to_bohrRad;
      Overlap = Voltage = 0.0;

      if (R < OvIntMaxR)
      {
        type_j = type[j];
        eb = qtpie->gauss_exp[type_j];
        chib = chi[type_j];

        // The expressions below are in atomic units
        // Implementation from Chen Jiahao, Theory and applications of fluctuating-charge models, 2009 (with normalization constants added)
        p = ea + eb;
        m = ea * eb / p;
        Overlap = pow((4. * m / p), 0.75) * exp(-m * R * R);

        // Implementation from T. Halgaker et al., Molecular electronic-structure theory, 2000
//        p = ea + eb;
//        m = ea * eb / p;
//        Overlap = pow((M_PI / p), 1.5) * exp(-m * R * R);

        if (system->pair_ptr->efield_flag) {
          phib = Electric_field_potential(x[j],system);
          Voltage = chia - chib + phib;
        } else {
          Voltage = chia - chib;
        }
        Nominator += Voltage * Overlap;
        Denominator += Overlap;
      }
    }
    if (Denominator != 0.0 && Nominator != 0.0)
      lchi_eff[i] = Nominator / Denominator;
    else
      lchi_eff[i] = zero;

    if (system->pair_ptr->efield_flag) {
      phia = Electric_field_potential(x[i],system);
      lchi_eff[i] -= phia;
    }
  }
}

/* ---------------------------------------------------------------------- */

double find_min(double *array, int array_length)
{
  // since types start from 1, gaussian exponents start from 1
  double smallest = array[1];
  for (int i = 1; i < array_length; i++) 
  {
    if (array[i] < smallest) 
      smallest = array[i];
  }
  return smallest;
}

/* ---------------------------------------------------------------------- */

double Distance(double *Point1, double *Point2)
{
  double x, y, z;
  x = Point2[0] - Point1[0];
  y = Point2[1] - Point1[1];
  z = Point2[2] - Point1[2];
  return sqrt(x*x + y*y + z*z);
}

/* ---------------------------------------------------------------------- */

double Electric_field_potential(double *x, reax_system *system)
{
  double x_efcomp, y_efcomp, z_efcomp;
  x_efcomp = x[0] * system->pair_ptr->ex;
  y_efcomp = x[1] * system->pair_ptr->ey;
  z_efcomp = x[2] * system->pair_ptr->ez;
  return x_efcomp + y_efcomp + z_efcomp;
}

