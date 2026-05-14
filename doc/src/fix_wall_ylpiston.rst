.. index:: fix wall/ylpiston
.. index:: fix wall/ylpiston/kk

fix wall/ylpiston command
=========================

Accelerator Variants: *wall/ylpiston/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID wall/ylpiston face coord energy cutoff [fix-mw-ID]

* ID, group-ID are documented in :doc:`fix <fix>` command
* wall/ylpiston = style name of this fix command
* face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*
* coord = position of the piston wall along the specified dimension (distance units)
* energy = energy scale at the wall face (energy units)
* cutoff = interaction cutoff distance from the wall (distance units)
* fix-mw-ID = (optional) ID of a :doc:`fix mwindow/erase <fix_mwindow_erase>` fix; if provided, atoms on the erasing side of the moving window are excluded from the wall interaction

Examples
""""""""

.. code-block:: LAMMPS

   fix pw all wall/ylpiston zhi 50.0 10.0 3.0
   fix pw all wall/ylpiston xhi 80.0 5.0 2.5 mw

Description
"""""""""""

.. versionadded:: TBD

Apply a harmonic piston-wall potential to atoms in the group that come
within *cutoff* distance of the wall at position *coord*.  The
potential has the form

.. math::

   E(d) = E_3 \, d^2

where :math:`d = \text{cutoff} - \delta` is the penetration distance
measured from the cutoff surface, :math:`\delta` is the distance of
the atom from the wall position, and
:math:`E_3 = \text{energy} / \text{cutoff}^2`.

The force on each atom is

.. math::

   F = -2 E_3 \, d \cdot \text{sign}

where *sign* is +1 for *hi* faces and -1 for *lo* faces, so the
force pushes atoms back toward the interior of the simulation.  No
interaction is applied to atoms that are at or beyond the wall
(negative penetration) or farther than *cutoff* from the wall.

The *face* keyword selects which box face the wall is on.  The
wall must be in a non-periodic dimension.

If the optional *fix-mw-ID* argument is given, it must be the ID of a
:doc:`fix mwindow/erase <fix_mwindow_erase>` fix.  In that case, atoms
that have already crossed the erasing plane (i.e. atoms on the
processed side of the moving window) are excluded from the wall
interaction, so the wall only acts on atoms in the active shock region.

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.  None of the :doc:`fix_modify <fix_modify>` options are
relevant to this fix.

This fix computes a global scalar (the total wall energy) and a global
vector of length 3 (force components along x, y, and z).  These can be
accessed via ``f_ID`` and ``f_ID[1]``--``f_ID[3]``.  The scalar and
vector values are "extensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix style is part of the SHOCK package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

The wall must be in a non-periodic dimension.

Related commands
""""""""""""""""

:doc:`fix wall/reflect <fix_wall>`,
:doc:`fix wall/piston <fix_wall_piston>`,
:doc:`fix mwindow/erase <fix_mwindow_erase>`

Default
"""""""

none
