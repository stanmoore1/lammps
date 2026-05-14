.. index:: fix shock/info
.. index:: fix shock/info/kk

fix shock/info command
======================

Accelerator Variants: *shock/info/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID shock/info nevery nfreq nrepeat dim origin delta nmin compute-pe-ID compute-stress-ID einfo-prefix stress-prefix keyword value ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* shock/info = style name of this fix command
* nevery = compute sample every this many timesteps
* nfreq = output averages every this many timesteps
* nrepeat = number of samples per output
* dim = *x* or *y* or *z* (spatial averaging dimension)
* origin = *lower* or *upper* or *center* or a coordinate value (distance units)
* delta = bin thickness (distance units)
* nmin = minimum number of atoms per variable-width output bin
* compute-pe-ID = ID of a per-atom potential energy :doc:`compute <compute_pe_atom>`
* compute-stress-ID = ID of a per-atom stress :doc:`compute <compute_stress_atom>`
* einfo-prefix = filename prefix for the energy/density/velocity output files
* stress-prefix = filename prefix for the pressure-tensor output files
* zero or more keyword/value pairs may be appended
* keyword = *units*

  .. parsed-literal::

       *units* value = *box* or *lattice* or *reduced*
         *box* = distance units are in simulation box units (default)
         *lattice* = distance units are in lattice spacings
         *reduced* = distance units are in reduced (lamda) coordinates

Examples
""""""""

.. code-block:: LAMMPS

   compute pe all pe/atom
   compute stress all stress/atom NULL
   fix info all shock/info 10 1000 100 x lower 2.0 5 pe stress einfo stress units box

Description
"""""""""""

.. versionadded:: TBD

Compute and output spatially-averaged shock-wave diagnostics.  On each
timestep that is a multiple of *nevery*, the per-atom potential energy
and stress tensor are sampled and binned along the *dim* axis.  After
*nrepeat* samples the per-bin averages are accumulated, and every
*nfreq* timesteps (which must be a multiple of *nevery*) the
accumulated data are reduced across processors, normalized, and written
to two sets of files.

The binning uses uniform layers of thickness *delta* along the *dim*
axis.  The reference point (offset) for the layer positions is set by
the *origin* argument:

* *lower* -- lower box face along *dim*
* *upper* -- upper box face along *dim*
* *center* -- center of the box along *dim*
* a numeric value -- that coordinate

After accumulating over *nfreq* timesteps, the uniform layers are
merged into variable-width bins so that each bin contains at least
*nmin* atom-samples on average.  This removes statistical noise from
sparsely-occupied regions.

**Output files**

Two output files are created per output step:

* ``<einfo-prefix>.<timestep>`` -- contains per-bin density, velocity
  components, potential energy per atom, kinetic energy components, and
  total energy.
* ``<stress-prefix>.<timestep>`` -- contains per-bin density and the
  full 6-component virial pressure tensor (potential and kinetic parts
  and their sum), reported in GPa (for *metal* units).

**Units conversion**

Density values include a factor of 1.66053 (appropriate for converting
:math:`\text{amu}/\text{Å}^3` to :math:`\text{g/cm}^3` in *metal*
units).  Velocity values are scaled by 0.1 (appropriate for converting
:math:`\text{Å/ps}` to :math:`\text{km/s}` in *metal* units).  Stress
values are converted to GPa using a unit-style-dependent factor
(1e-4 for *metal*, 1.01325e-4 for *real*, 1e-9 for *si*).

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.  None of the :doc:`fix_modify <fix_modify>` options are
relevant to this fix.  No global or per-atom quantities are stored by
this fix for access by various :doc:`output commands <Howto_output>`.
No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix style is part of the SHOCK package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

This fix requires a triclinic box to be used with ``units reduced``.

Related commands
""""""""""""""""

:doc:`compute pe/atom <compute_pe_atom>`,
:doc:`compute stress/atom <compute_stress_atom>`,
:doc:`fix ave/chunk <fix_ave_chunk>`

Default
"""""""

units box
