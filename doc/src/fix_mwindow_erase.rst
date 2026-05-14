.. index:: fix mwindow/erase
.. index:: fix mwindow/erase/kk

fix mwindow/erase command
=========================

Accelerator Variants: *mwindow/erase/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID mwindow/erase edge Aerase Slope d_min d_max rate_b dwmax Elast E0 Ewf N N_u_d region-ID compute-ID

* ID, group-ID are documented in :doc:`fix <fix>` command
* mwindow/erase = style name of this fix command
* edge = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*
* Aerase = initial erasing-plane amplitude (distance units)
* Slope = slope of erasing-plane position versus amplitude
* d_min = minimum erasing-plane position (distance units)
* d_max = maximum erasing-plane position (distance units)
* rate_b = damping rate for erasing-plane control (negative enables feedback)
* dwmax = target rate of change of the driving parameter (per timestep)
* Elast = initial (last) energy per atom, or ``N=<value>`` to control by atom count
* E0 = unperturbed (reference) energy per atom
* Ewf = target energy per atom
* N = delete atoms every this many timesteps
* N_u_d = update erasing-plane position every this many timesteps
* region-ID = ID of region from which atoms are candidates for deletion
* compute-ID = ID of a per-atom :doc:`compute pe/atom <compute_pe_atom>` for energy feedback

Examples
""""""""

.. code-block:: LAMMPS

   compute pe all pe/atom
   fix ew all mwindow/erase xhi 5.0 1.0 0.0 20.0 -0.1 0.01 N=500 -4.0 -3.5 100 200 rightedge pe

Description
"""""""""""

.. versionadded:: TBD

Delete atoms from the simulation that have crossed an adaptively
controlled erasing plane.  This fix is designed for use with the
moving window technique in shock-wave simulations, where atoms that
have been processed by the shock front are continuously removed from
the leading edge of the simulation domain so that the computational
cost remains bounded.

The *edge* keyword selects which face of the box the erasing plane is
associated with (*xlo*, *xhi*, *ylo*, *yhi*, *zlo*, or *zhi*).
Atoms on the erasing side of the erasing plane (controlled by *edge*
direction) and inside the specified region are marked for deletion
every *N* timesteps just before neighbor-list rebuilding
(``PRE_EXCHANGE``).

The position of the erasing plane is updated every *N_u_d* timesteps
using a feedback controller that drives the average per-atom potential
energy (or the number of atoms, when ``N=<value>`` is used for
*Elast*) toward the target value *Ewf*.  The control law adjusts the
amplitude *Aerase* and maps it to a plane position via

.. math::

   d_\text{erase} = d_\text{max} - A_\text{erase}
                    \frac{\text{Slope} + 1 - w}{\text{Slope}}

where :math:`w = (E_\text{tot} - E_0) / (E_\text{wish} - E_0)` is the
normalized progress toward the target energy and *Aerase* is updated as

.. math::

   A_\text{erase} \mathrel{+}= R_x

with :math:`R_x` computed from the proportional-integral feedback using
*rate_b* and *dwmax*.  The resulting plane position is clamped to
[*d_min*, *d_max*].

The *compute-ID* argument must refer to a
:doc:`compute pe/atom <compute_pe_atom>` (or equivalent per-atom
potential energy compute).

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Information about the current erasing-plane position *Aerase*,
*d_erase*, and *Elast* is written to :doc:`binary restart files
<restart>` and read back when a simulation is restarted.

This fix computes a global scalar (the current erasing-plane position
*d_erase*) and a global vector of length 8 accessible via
``f_ID[1]``--``f_ID[8]`` on every timestep.  The vector components
are: timestep, *Elast*, *Etot*, *w*, *qq*, *bb*, *Rx*,
*Aerase*, *d_erase*.

No parameter of this fix can be used with the *start/stop* keywords
of the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix style is part of the SHOCK package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

Atoms belonging to the group passed to :doc:`atom_modify first
<atom_modify>` cannot be deleted by this fix.

Related commands
""""""""""""""""

:doc:`fix wall/ylpiston <fix_wall_ylpiston>`,
:doc:`fix append/atoms <fix_append_atoms>`,
:doc:`compute pe/atom <compute_pe_atom>`

Default
"""""""

none
