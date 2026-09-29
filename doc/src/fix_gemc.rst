.. index:: fix gemc

fix gemc command
================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID gemc N M X V T displace maxdlogvolratio seed keyword ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* gemc = style name of this fix command
* N = invoke this fix every N steps
* M = average number of atom translations to attempt every N steps
* X = average number of atom exchanges between the two boxes to attempt every N steps
* V = average number of volume exchanges between the two boxes to attempt every N steps
* T = temperature of the Gibbs ensemble (temperature units)
* displace = maximum Monte Carlo translation distance (distance units)
* maxdlogvolratio = maximum change of ln(V1/V2) in a volume exchange (unitless)
* seed = random # seed (positive integer)
* zero or more keywords may be appended
* keyword = *full_energy*

  .. parsed-literal::

       *full_energy* = compute the full energy of the system for every move

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all gemc 1 100 20 2 0.9 0.3 0.05 29494
   fix mc all gemc 10 1000 200 10 120.0 1.0 0.1 4711 full_energy

Description
"""""""""""

.. versionadded:: 4Jul2026

This fix performs Monte Carlo (MC) moves in the Gibbs ensemble (GEMC)
using two separate simulation boxes that are in thermodynamic contact
without having an interface between them :ref:`(Panagiotopoulos)
<Panagiotopoulos1>`, :ref:`(Frenkel) <Frenkel3>`.  The boxes exchange
volume, so that the pressure is the same in both, and they exchange
atoms, so that the chemical potential is the same in both, while the
total volume, the total number of atoms, and the temperature *T* are
fixed.  The most common application is the calculation of vapor-liquid
coexistence: when started at a total density inside the two-phase
region, one box ends up containing the vapor and the other box the
liquid, and the average densities of the boxes are the two coexisting
densities at temperature *T*.  The :doc:`Gibbs ensemble Howto
<Howto_gemc>` explains how to set up and analyze such a calculation
step by step.

The two boxes are two :doc:`partitions <Run_options>` of LAMMPS, so
LAMMPS must be started with exactly two partitions, for example
``mpirun -np 2 lmp -in in.gemc -partition 1 1``.  Both partitions read
the same input script.  Commands that should differ between the boxes,
like the initial box size or number of atoms, can use
:doc:`world-style variables <variable>` or the :doc:`partition
<partition>` command.  The two partitions can use different numbers of
processors, which is useful because the liquid box usually contains
many more atoms than the vapor box, e.g. ``-partition 1 3`` on 4
processors.  The fix command itself (and the number of atom types)
must be the same in both partitions.

Every *N* timesteps the fix performs a total of *M* + *X* + *V* MC
moves.  The type of each move is chosen randomly with probabilities
proportional to *M*, *X*, and *V*, so these are the average numbers of
the three kinds of moves.  Both boxes always attempt the same kind of
move at the same time:

* A *translation* moves a randomly chosen atom of the fix group in each
  box by a random displacement inside a sphere of radius *displace*.
  The two boxes accept or reject their translations independently with
  the Metropolis criterion :math:`\min[1, \exp(-\Delta U/k_B T)]`.

* An *exchange* removes a randomly chosen atom of the fix group from one
  box, the donor box, and inserts it at a random position in the other
  box.  The donor box is chosen randomly with equal probability.  The
  inserted atom keeps the atom type and group membership of the removed
  atom and gets a velocity drawn from a Maxwell-Boltzmann distribution
  at temperature *T*.  The move is accepted with probability

  .. math::

     \min\left[1, \frac{N_d\,V_r}{(N_r+1)\,V_d}
     \exp\left(-\frac{\Delta U_d + \Delta U_r}{k_B T}\right)\right]

  where :math:`N_d, V_d` and :math:`N_r, V_r` are the number of atoms
  in the fix group and the volume of the donor and receiving box before
  the move.

* A *volume exchange* changes :math:`\ln(V_1/V_2)` by a random amount
  between -*maxdlogvolratio* and +*maxdlogvolratio* while keeping the
  total volume :math:`V_1+V_2` constant.  Each box is scaled
  uniformly with its lower corner kept fixed, and all atom coordinates
  are scaled with it.  The move is accepted with probability

  .. math::

     \min\left[1, \left(\frac{V_1'}{V_1}\right)^{N_1+1}
     \left(\frac{V_2'}{V_2}\right)^{N_2+1}
     \exp\left(-\frac{\Delta U_1 + \Delta U_2}{k_B T}\right)\right]

  where :math:`N_1, N_2` are the total numbers of atoms in the boxes.

Exchange and volume moves are always accepted or rejected by both boxes
together.  The energy *U* is the total potential energy of a box, as
computed by the *thermo_pe* compute, see below.

Atoms that are not in the fix group are never translated or
exchanged, but they interact with the other atoms and are scaled by
volume moves.  Different atom types can be used, e.g. to study the
coexistence of a mixture; the exchange move then transfers the type
of the randomly chosen atom.

Choosing the parameters: *displace* is usually tuned so that roughly
30% to 50% of the translations in the liquid box are accepted, and
*maxdlogvolratio* so that roughly 30% to 50% of the volume exchanges
are accepted.  The acceptance ratio of exchanges is usually low for
dense liquids (a few percent or less), so *X* must be large enough to
exchange every atom several times during the run.  The acceptance
counts are available as output of this fix, see below.

If the fix is used together with time integration, e.g. :doc:`fix nvt
<fix_nh>`, a hybrid MD/MC simulation is performed.  In this case the
thermostat temperature should be the same as *T*, and the temperature
compute used by the thermostat should account for the changing number
of atoms, for example:

.. code-block:: LAMMPS

   compute mdtemp all temp
   compute_modify mdtemp dynamic/dof yes
   fix mdnvt all nvt temp 0.9 0.9 0.5
   fix_modify mdnvt temp mdtemp

In pure MC simulations without time integration, the atom velocities
are irrelevant, and LAMMPS will print a warning that atoms will not
move, which can be ignored.  In both cases, use :doc:`compute_modify
thermo_temp dynamic/dof yes <compute_modify>` so that the thermodynamic
output uses the current number of atoms in each box.

.. versionadded:: TBD

By default, translation and exchange moves compute only the change of
the pair energy of the moved atom, which is much faster than computing
the total energy of the box.  This is only possible when the pair style
supports the *single()* function, is not a many-body potential, does
not use :doc:`pair_modify tail yes <pair_modify>`, and when no fix
contributes to the potential energy and *displace* is not larger than
the neighbor skin distance set by the :doc:`neighbor <neighbor>`
command.  If any of these conditions is not met, a warning is printed
and the total energy is computed for every move.  The same is done for
a move in a box whose edge length is smaller than the pair cutoff.
Volume moves always compute the total energy.  The *full_energy*
keyword requests that the total energy is computed for all moves.

Some fixes have an associated potential energy.  Examples of such fixes
include: :doc:`efield <fix_efield>`, :doc:`gravity <fix_gravity>`,
:doc:`addforce <fix_addforce>`, :doc:`restrain <fix_restrain>`, and
:doc:`wall fixes <fix_wall>`.  For that energy to be included in the
total potential energy of the system (the quantity used by the MC
moves), you MUST enable the :doc:`fix_modify <fix_modify>` *energy*
option for that fix.

Neighbor lists are rebuilt every *N* timesteps that this fix is
invoked, so you should not set *N* too small when combining the fix
with time integration.  In pure MC simulations *N* = 1 with a large
number of moves per invocation is most efficient.

During a run, the fix prints a progress report to the screen and to
the log file of the whole run (not to those of the partitions) every 1%
of the run.  It lists the numbers of accepted and attempted moves since
the last report (translations of the first box, exchanges and volume
changes of both boxes) and the volume, number of atoms, and number
density of each box.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the fix to :doc:`binary restart files
<restart>`.  This includes information about the random number
generator seeds, the next timestep for MC moves, and the numbers of MC
move attempts and successes.  Each partition writes its own restart
file, so the restart file name must be different for the two
partitions, e.g. by using a world-style variable.  See the
:doc:`read_restart <read_restart>` command for info on how to
re-specify a fix in an input script that reads a restart file, so that
the operation of the fix continues in an uninterrupted fashion.

.. note::

   For this to work correctly, the timestep must **not** be changed
   after reading the restart with :doc:`reset_timestep
   <reset_timestep>`.  The fix will try to detect it and stop with an
   error.

None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix.

.. versionchanged:: TBD

This fix computes a global vector of length 6, which can be accessed by
various :doc:`output commands <Howto_output>`, e.g. as *f_ID[1]* in
the :doc:`thermo_style <thermo_style>` command.  The vector values are
the following cumulative counts for the box of the partition
where they are accessed:

  #. translation attempts
  #. translation successes
  #. exchange attempts
  #. exchange successes
  #. volume change attempts
  #. volume change successes

The vector values calculated by this fix are "intensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

This fix requires exactly two partitions, a 3d simulation with an
orthogonal box that is periodic in all three dimensions, and atom IDs.

This fix currently supports only individual atoms.  Molecules, charged
atoms, and long-range solvers (:doc:`kspace_style <kspace_style>`)
are not supported.

Do not set :doc:`neigh_modify once yes <neigh_modify>` or else this fix
will never be called.  Reneighboring is **required**.

Use of multiple *fix gemc* commands in the same input script can be
problematic.

Related commands
""""""""""""""""

:doc:`fix gcmc <fix_gcmc>`,
:doc:`fix widom <fix_widom>`,
:doc:`fix atom/swap <fix_atom_swap>`,
:doc:`partition <partition>`

Defaults
""""""""

By default, single-atom energies are used when possible, i.e. the
*full_energy* keyword is not set.

----------

.. _Panagiotopoulos1:

**(Panagiotopoulos)** Panagiotopoulos, Mol Phys, 61, 813-826 (1987).

.. _Frenkel3:

**(Frenkel)** Frenkel and Smit, Understanding Molecular Simulation,
3rd edition, Academic Press, London, 2023.
