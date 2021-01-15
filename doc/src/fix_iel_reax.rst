.. index:: fix iel/reax

fix iel/reax command
====================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID iel/reax Nevery cutlo cuthi tolerance/t tolerance/s params thermo tautemp temp/t temp/s args

* ID, group-ID are documented in :doc:`fix <fix>` command
* iel/reax = style name of this fix command
* Nevery = perform QEq every this many steps
* cutlo,cuthi = lo and hi cutoff for Taper radius
* tolerance/t = precision to which charges will be equilibrated for the T matrix
* tolerance/s = precision to which charges will be equilibrated for the S matrix
* params = reax/c or a filename
* thermo = integrator style 
* tautemp = ?
* temp/t = ?
* temp/s = ?

* one or more keywords or keyword/value pairs may be appended

  .. parsed-literal::

     keyword = *dual* or *maxiter*
       *maxiter* N = limit the number of iterations to *N*


Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all iel/reax 1 0.0 10.0 1.0e-6 reax/c 2 10.0 1.0e-6 1.0e-5

Description
"""""""""""

Perform the charge equilibration (QEq) method using the inertial extended Lagrangian/self-consistent field scheme (iEL–SCF) as described in
:ref:`(Leven) <Leven>`.  It is
typically used in conjunction with the ReaxFF force field model as
implemented in the :doc:`pair_style reax/c <pair_reaxc>` command, but
it can be used with any potential in LAMMPS, so long as it defines and
uses charges on each atom. For more technical details about the
charge equilibration performed by fix iel/reax, see the
:ref:`(Leven) <iel-Leven>` paper.

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

.. _Leven:

**(Leven)** Leven, Head-Gordon, Phys. Chem. Chem. Phys., 21, 18652-18659 (2019).
