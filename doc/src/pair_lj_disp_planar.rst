.. index:: pair_style lj/disp/planar

pair_style lj/disp/planar command
=================================

.. versionadded:: 30Mar2026

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj/disp/planar rcut Delta

* rcut = inner cutoff for the Lennard-Jones interaction (distance units)
* Delta = width of the dispersion switching shell (optional): 0.1 (default) (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style lj/disp/planar 2.5
   pair_style lj/disp/planar 2.5 0.1
   pair_coeff * * 1.0 1.0

Description
"""""""""""

The *lj/disp/planar* style is the short-range pair style matched to the
planar dispersion long-range solvers :doc:`ewald/disp/planar and
pppm/disp/planar <kspace_style>` :ref:`(Moore) <MooreSB>`.  It computes the
standard 12/6 Lennard-Jones potential

.. math::

   E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
       \left(\frac{\sigma}{r}\right)^6 \right]

in full out to the inner cutoff :math:`r_{\mathrm{cut}}`, but over the thin
shell :math:`[r_{\mathrm{cut}}, r_{\mathrm{cut}}+\Delta]` it fades the
*attractive* :math:`1/r^6` dispersion term off smoothly by multiplying it by
:math:`1-S(r)`, where :math:`S` is the :math:`C^3` septic smoothstep

.. math::

   S(t) = t^4\,(35 - 84\,t + 70\,t^2 - 20\,t^3), \qquad
   t = \frac{r - r_{\mathrm{cut}}}{\Delta} .

The matched reciprocal solver supplies the complementary :math:`S(r)/r^6`
plane-averaged tail, so that the full :math:`1/r^6` dispersion is recovered.
The :math:`1/r^{12}` repulsion is short-ranged and is always evaluated in full
out to :math:`r_{\mathrm{cut}}+\Delta`; only the :math:`1/r^6` dispersion is
split between real and reciprocal space.  Because the split is exact, no energy
shift is applied and no long-range tail correction is added by this pair style
(the reciprocal solver continues the dispersion tail).

This pair style is intended to be used together with a matched
:doc:`ewald/disp/planar or pppm/disp/planar <kspace_style>` kspace style, which
requires the *B* dispersion coefficient, the inner cutoff, and the switch width
:math:`\Delta` from this pair style.  It may also be run on its own as an
ordinary smoothly-truncated Lennard-Jones pair style.

Coefficients
""""""""""""

The following coefficients must be defined for each pair of atom types via the
:doc:`pair_coeff <pair_coeff>` command as in the example above, or in the data
file or restart files read by the :doc:`read_data <read_data>` or
:doc:`read_restart <read_restart>` commands:

* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)

The inner cutoff :math:`r_{\mathrm{cut}}` and switch width :math:`\Delta` are
global and are set by the pair_style command; they cannot be specified per type.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the epsilon and sigma coefficients can be
mixed.  The default mix value is *geometric*; *arithmetic* (Lorentz-Berthelot)
mixing is also supported and is honored by the matched kspace style.  See the
:doc:`pair_modify <pair_modify>` command for details.

This pair style does not support the :doc:`pair_modify <pair_modify>` shift
option (the dispersion tail is continued by the matched kspace style, so no
energy shift is applied).

This pair style is **incompatible** with the :doc:`pair_modify <pair_modify>`
tail option: the long-range dispersion tail is supplied by the matched kspace
style, so an analytic tail correction would double count it.

This pair style writes its information, including the switch width
:math:`\Delta`, to :doc:`binary restart files <restart>`, so pair_style and
pair_coeff commands do not need to be specified in an input script that reads a
restart file.

This pair style does not support the *inner*, *middle*, and *outer* keywords of
the :doc:`run_style respa <run_style>` command.

----------

Restrictions
""""""""""""

This pair style is part of the KSPACE package.  It is only enabled if LAMMPS
was built with that package.  See the :doc:`Build package <Build_package>` doc
page for more info.

This pair style is designed to be paired with the matched
:doc:`ewald/disp/planar or pppm/disp/planar <kspace_style>` kspace style for
systems whose average density varies in one direction (planar interfaces).

Related commands
""""""""""""""""

* :doc:`pair_coeff <pair_coeff>`
* :doc:`pair_style lj/cut <pair_lj>`
* :doc:`kspace_style ewald/disp/planar <kspace_style>`
* :doc:`kspace_style pppm/disp/planar <kspace_style>`

Default
"""""""

none
