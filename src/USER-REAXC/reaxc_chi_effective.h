/*----------------------------------------------------------------------
  Computation of the effective electronegativity for the QTPIE method
  ----------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   Additional variables for the QTPIE option: 
     fix qeq/reax ... qtpie file_name
   where file_name is the file containing the Gaussian exponentials with
   the format of: <Atom type> <GTO exponential>
   where <Atom type> corresponds to the atoms used in each simulation

   Attention : GTO exponents must be given in atomic units, i.e. 1/a_0,
               where a_0 is the Bohr radius. 
------------------------------------------------------------------------- */

#ifndef __CHI_EFFECTIVE_H_
#define __CHI_EFFECTIVE_H_

#include "pointers.h"
#include "reaxc_types.h"

void calculate_chi_eff( qtpie_parameters*, LAMMPS_NS::Atom*, reax_system*, double*, int, int, double* );
double find_min( double*, int );
double Distance( double*, double* );

double Electric_field_potential( double*, reax_system* );

#endif
