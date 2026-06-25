.. index:: pair_style lj/cut/dispplanar

pair_style lj/cut/dispplanar command
====================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj/cut/dispplanar rcut Delta

* rcut = total (outer) cutoff for the full Lennard-Jones interaction (distance units)
* Delta = width of the smoothstep switching shell, which ramps inward over [rcut-Delta, rcut] (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style lj/cut/dispplanar 3.0 0.6
   pair_coeff 1 1 1.0 1.0

   kspace_style ewald/disp/planar 1.0e-5

Description
"""""""""""

.. versionadded:: TBD

The *lj/cut/dispplanar* style is a variant of :doc:`pair_style lj/cut
<pair_lj>` that computes the standard 12/6 Lennard-Jones potential,

.. math::

   E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
       \left(\frac{\sigma}{r}\right)^6 \right]

It is the matched short-range pair style for the planar long-range
dispersion solvers :doc:`kspace_style ewald/disp/planar and
pppm/disp/planar <kspace_style>`, which compute the long-range
:math:`1/r^6` (van der Waals) interaction for systems whose mean density
varies in only one direction, such as a planar liquid-vapor interface.

This style is a plain :doc:`lj/cut <pair_lj>` evaluated out to the total
cutoff :math:`r_c` (the same cutoff used by the other planar Ewald sums) with no
energy offset; the switching is done entirely by the matched kspace style.  It
exposes the inner cutoff :math:`r_c-\Delta` (where the switch starts), the switch
width :math:`\Delta`, and the dispersion amplitude to the kspace style, which splits
the long-range :math:`1/r^6` dispersion term over the inner shell with a
:math:`C^3`-continuous (septic) smoothstep that ramps from 0 at
:math:`r_c-\Delta` to 1 at :math:`r_c`,

.. math::

   S(r) = t^4 \left( 35 - 84 t + 70 t^2 - 20 t^3 \right), \qquad
   t = \frac{r - r_c}{\Delta}

over the shell :math:`[r_c-\Delta, r_c]`.  The smooth long-range part
:math:`S(r)\,u(r)` (where :math:`u(r) = -4\epsilon\sigma^6/r^6` is the
attractive dispersion term) is summed by the reciprocal-space solver; it
vanishes inside :math:`r_c` and is :math:`C^3`-continuous at
:math:`r_c`, so the *z*-Fourier coefficients of the dispersion-weighted
density decay rapidly and no Gibbs ringing occurs.

The kspace style also applies a real-space "shell correction" that
subtracts the laterally-uniform mean-field part of its reciprocal sum
over the shell.  Because this pair style evaluates the *full*
:math:`1/r^6` dispersion (not the switched part) over
:math:`[r_c-\Delta, r_c]`, the pair and kspace together supply the exact
three-dimensional shell interaction.  For a homogeneous fluid this
reduces exactly to the standard long-range tail correction.

The *lj/cut/dispplanar* style must be used together with a matched
:doc:`kspace_style ewald/disp/planar or pppm/disp/planar
<kspace_style>`.  Using it without one of those kspace styles omits the
long-range part of the interaction (it then behaves as a plain
:doc:`lj/cut <pair_lj>` truncated at :math:`r_c`).

Coefficients
""""""""""""

The following coefficients must be defined for each pair of atom types
via the :doc:`pair_coeff <pair_coeff>` command as in the example above,
or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)

Unlike :doc:`pair_style lj/cut <pair_lj>`, a per-pair cutoff cannot be
specified: the interaction and neighbor cutoff is always
:math:`r_c`, set globally by the *pair_style* command.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the :math:`\epsilon` and
:math:`\sigma` coefficients can be mixed.  The default mix value is
*geometric*.  See the :doc:`pair_modify <pair_modify>` command for
details.

This pair style is **incompatible** with the
:doc:`pair_modify <pair_modify>` *tail yes* option, because the
long-range tail of the dispersion interaction is already handled by the
matched planar kspace style.

This pair style writes its information to :doc:`binary restart files
<restart>`, so pair_style and pair_coeff commands do not need to be
specified in an input script that reads a restart file.

----------

Restrictions
""""""""""""

This pair style is part of the KSPACE package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

This pair style requires a matched
:doc:`kspace_style ewald/disp/planar or pppm/disp/planar
<kspace_style>`.  It is incompatible with the
:doc:`pair_modify <pair_modify>` *tail yes* option.  Like the planar
kspace styles, it requires fully periodic boundaries
(``boundary p p p``), an orthogonal (non-triclinic) simulation box, and
a three-dimensional simulation.

This pair style was written by Stan Moore (SNL).

Related commands
""""""""""""""""

* :doc:`pair_coeff <pair_coeff>`
* :doc:`pair_style lj/cut <pair_lj>`
* :doc:`kspace_style <kspace_style>`
* :doc:`kspace_modify <kspace_modify>`

Default
"""""""

none
