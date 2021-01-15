.. index:: fix xlmd/reax

fix xlmd/reax command
====================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID xlmd/reax Nevery cutlo cuthi tolerance params thermo mLatent tauLatent tLatent seed args

* ID, group-ID are documented in :doc:`fix <fix>` command
* xlmd/reax = style name of this fix command
* Nevery = perform QEq every this many steps
* cutlo,cuthi = lo and hi cutoff for Taper radius
* tolerance = precision to which charges will initially be equilibrated
* params = reax/c or a filename
* thermo = integrator style
* mLatent = latent mass
* tauLatent = latent thermostat strength
* tLatent = latent temperature
* seed = random number seed to use for white noise (positive integer) 

* one or more keywords or keyword/value pairs may be appended

  .. parsed-literal::

     keyword = *dual* or *maxiter*
       *maxiter* N = limit the number of iterations to *N*


Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all xlmd/reax 1 0.0 10.0 1e-8 reax/c 1 11.0 1.0e5 3.0e-3 87287

Description
"""""""""""

Perform the charge equilibration (QEq) method using stochastic constrained extended Langrangian molecular dynamics (SC-XLMD) method described in
:ref:`(Tan) <Tan>`.  It is
typically used in conjunction with the ReaxFF force field model as
implemented in the :doc:`pair_style reax/c <pair_reaxc>` command, but
it can be used with any potential in LAMMPS, so long as it defines and
uses charges on each atom. For more technical details about the
charge equilibration performed by fix xlmd/reax, see the
:ref:`(Tan) <xlmd-Tan>` paper.

For information about the other parameters, see the :doc:`fix qeq/reax <fix_qeq_reax>` command.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.  No global scalar or vector or per-atom
quantities are stored by this fix for access by various :doc:`output commands <Howto_output>`.  No parameter of this fix can be used
with the *start/stop* keywords of the :doc:`run <run>` command.

This fix is invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the USER-REAXC package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` doc page for more info.

This fix does not correctly handle interactions
involving multiple periodic images of the same atom. Hence, it should not
be used for periodic cell dimensions less than 10 angstroms.

Related commands
""""""""""""""""

:doc:`pair_style reax/c <pair_reaxc>`

Default
"""""""

maxiter 200

----------

.. _Tan:

**(Tan)** Tan, Leven, An, Lin, Head-Gordon, arXiv:2005.10736 [physics.comp-ph] (2020).
