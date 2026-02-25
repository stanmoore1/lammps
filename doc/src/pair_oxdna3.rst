.. index:: pair_style oxdna3/excv
.. index:: pair_style oxdna3/stk
.. index:: pair_style oxdna3/hbond
.. index:: pair_style oxdna3/xstk
.. index:: pair_style oxdna3/coaxstk
.. index:: pair_style oxdna3/dh

pair_style oxdna3/excv command
==============================

pair_style oxdna3/stk command
=============================

pair_style oxdna3/hbond command
===============================

pair_style oxdna3/xstk command
==============================

pair_style oxdna3/coaxstk command
=================================

pair_style oxdna3/dh command
============================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style1

   pair_coeff * * style2 args

* style1 = *hybrid/overlay oxdna3/excv oxdna3/stk oxdna3/hbond oxdna3/xstk oxdna3/coaxstk oxdna3/dh*

* style2 = *oxdna3/excv* or *oxdna3/stk* or *oxdna3/hbond* or *oxdna3/xstk* or *oxdna3/coaxstk* or *oxdna3/dh*
* args = list of arguments for these particular styles

.. parsed-literal::

     *oxdna3/excv* args = oxdna3_lj.cgdna or oxdna3_real.cgdna
     *oxdna3/stk* args = T oxdna3_lj.cgdna or oxdna3_real.cgdna
       T = temperature (LJ units: 0.1 = 300 K, real units: 300 = 300 K)
     *oxdna3/hbond* args = oxdna3_lj.cgdna or oxdna3_real.cgdna
     *oxdna3/xstk* args = oxdna3_lj.cgdna or oxdna3_real.cgdna
     *oxdna3/coaxstk* args = oxdna3_lj.cgdna or oxdna3_real.cgdna
     *oxdna3/dh* args = T rhos oxdna3_lj.cgdna or oxdna3_real.cgdna
       T = temperature (LJ units: 0.1 = 300 K, real units: 300 = 300 K)
       rhos = salt concentration (mole per litre)

Examples
""""""""

.. code-block:: LAMMPS

   # LJ units
   pair_style hybrid/overlay oxdna3/excv oxdna3/stk oxdna3/hbond oxdna3/xstk oxdna3/coaxstk oxdna3/dh
   pair_coeff * * oxdna3/excv     oxdna3_lj.cgdna
   pair_coeff * * oxdna3/stk      0.1 oxdna3_lj.cgdna
   pair_coeff * * oxdna3/hbond    oxdna3_lj.cgdna
   pair_coeff 1 4 oxdna3/hbond    oxdna3_lj.cgdna
   pair_coeff 2 3 oxdna3/hbond    oxdna3_lj.cgdna
   pair_coeff * * oxdna3/xstk     oxdna3_lj.cgdna
   pair_coeff * * oxdna3/coaxstk  oxdna3_lj.cgdna
   pair_coeff * * oxdna3/dh       0.1 0.2 oxdna3_lj.cgdna

   # Real units
   pair_style hybrid/overlay oxdna3/excv oxdna3/stk oxdna3/hbond oxdna3/xstk oxdna3/coaxstk oxdna3/dh
   pair_coeff * * oxdna3/excv     oxdna3_real.cgdna
   pair_coeff * * oxdna3/stk      300.0 oxdna3_real.cgdna
   pair_coeff * * oxdna3/hbond    oxdna3_real.cgdna
   pair_coeff 1 4 oxdna3/hbond    oxdna3_real.cgdna
   pair_coeff 2 3 oxdna3/hbond    oxdna3_real.cgdna
   pair_coeff * * oxdna3/xstk     oxdna3_real.cgdna
   pair_coeff * * oxdna3/coaxstk  oxdna3_real.cgdna
   pair_coeff * * oxdna3/dh       300.0 0.2 oxdna3_real.cgdna

.. note::

   The coefficients in the above examples are provided in forms
   compatible with both *units lj* and *units real* (see documentation
   of :doc:`units <units>`).  In case of oxDNA3 these have to be read 
   from a potential file with correct unit style by specifying the name 
   of the file. The potential files for each unit style are included in the
   ``potentials`` directory of the LAMMPS distribution.

Description
"""""""""""

The *oxdna3* pair styles compute the pairwise-additive parts of the
oxDNA force field for coarse-grained modelling of DNA. The effective
interaction between the nucleotides consists of potentials for the
excluded volume interaction *oxdna3/excv*, the stacking *oxdna3/stk*,
cross-stacking *oxdna3/xstk* and coaxial stacking interaction
*oxdna3/coaxstk*, electrostatic Debye-Hueckel interaction *oxdna3/dh* as
well as the hydrogen-bonding interaction *oxdna3/hbond* between
complementary pairs of nucleotides on opposite strands.

The exact functional form of the pair styles is rather complex.  The
individual potentials consist of products of modulation factors, which
themselves are constructed from a number of more basic potentials
(Morse, Lennard-Jones, harmonic angle and distance) as well as quadratic
smoothing and modulation terms.  We refer to :ref:`(Bonato) <Bonato2>`
and the original oxDNA publications :ref:`(Ouldridge-DPhil)
<Ouldridge-DPhil4>` and :ref:`(Ouldridge) <Ouldridge4>`
for a detailed description of the oxDNA3 force field.

.. note::

   These pair styles have to be used together with the related oxDNA3
   bond style *oxdna3/fene* for the connectivity of the phosphate
   backbone (see also documentation of :doc:`bond_style oxdna3/fene
   <bond_oxdna>`). Most of the coefficients in the above example have to
   be kept fixed and cannot be changed without reparameterizing the
   entire model.  Exceptions are the first coefficient after
   *oxdna3/stk* (T=0.1 and corresponding *real unit* equivalents in the 
   above examples) and the two coefficients after *oxdna3/dh* (T=0.1 and
   rhos=0.2 in the above example). When using a Langevin
   thermostat e.g. through :doc:`fix langevin <fix_langevin>` or
   :doc:`fix nve/dotc/langevin <fix_nve_dotc_langevin>` the temperature
   coefficients have to be matched to the one used in the fix.

.. note::

   These pair styles have to be used with the *atom_style hybrid bond
   ellipsoid oxdna* (see documentation of :doc:`atom_style
   <atom_style>`). The *atom_style oxdna* stores the 3'-to-5' polarity
   of the nucleotide strand, which is set through the bond topology in
   the data file. The first (second) atom in a bond definition is
   understood to point towards the 3'-end (5'-end) of the strand.

.. warning::

   If data files are produced with :doc:`write_data <write_data>`, then
   the :doc:`newton <newton>` command should be set to *newton on*.
   Otherwise the data files will not have the same 3'-to-5' polarity 
   as the initial data file. This limitation does not apply to
   binary restart files produced with :doc:`write_restart <write_restart>`.

Example input and data files for DNA duplexes can be found in
``examples/PACKAGES/cgdna/examples/oxDNA3/``.  A
simple python setup tool which creates single straight or helical DNA
strands, DNA duplexes or arrays of DNA duplexes can be found in
``examples/PACKAGES/cgdna/util/``.

Please cite :ref:`(Henrich) <Henrich6>` in any publication that uses
this implementation. An updated documentation that contains general
information on the model, its implementation and performance as well as
the structure of the data and input file can be found `here
<PDF/CG-DNA.pdf>`_.

Please cite also the relevant oxDNA3 publication :ref:`(Bonato) <Bonato2>`.

----------

Potential file reading
""""""""""""""""""""""

For each pair style above the first non-modifiable argument can be a
filename (with exception of Debye-Hueckel, for which the effective
charge argument can be a filename), and if it is, no further arguments
should be supplied.  Therefore the following command:

.. code-block:: LAMMPS

   pair_coeff 1 4 oxdna2/hbond   seqdep oxdna_real.cgdna

will be interpreted as a request to read the corresponding hydrogen
bonding potential parameters from the file with the given name.  The
file can define multiple potential parameters for both bonded and pair
interactions, but for the example pair interaction above there must
exist in the file a line of the form:

.. code-block:: LAMMPS

  1 4 hbond     <coefficients>

If potential customization is required, the potential file reading can
be mixed with the manual specification of the potential parameters. For
example, the following command:

.. code-block:: LAMMPS

   pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh
   pair_coeff * * oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32
   pair_coeff * * oxdna2/stk     seqdep 0.1 oxdna2_lj.cgdna
   pair_coeff * * oxdna2/hbond   seqdep oxdna2_lj.cgdna
   pair_coeff 1 4 oxdna2/hbond   seqdep oxdna2_lj.cgdna
   pair_coeff 2 3 oxdna2/hbond   seqdep oxdna2_lj.cgdna
   pair_coeff * * oxdna2/xstk    oxdna2_lj.cgdna
   pair_coeff * * oxdna2/coaxstk oxdna2_lj.cgdna
   pair_coeff * * oxdna2/dh      0.1 0.5 0.815

will read the excluded volume and Debye-Hueckel effective charge *qeff*
parameters from the manual specification and all others from the
potential file *oxdna2_lj.cgdna*.

There are sample potential files for each unit style in the ``potentials``
directory of the LAMMPS distribution. The potential file unit system
must align with the units defined via the :doc:`units <units>`
command. For conversion between different *LJ* and *real* unit systems
for oxDNA, the python tool *lj2real.py* located in the
``examples/PACKAGES/cgdna/util/`` directory can be used. This tool assumes
similar file structure to the examples found in
``examples/PACKAGES/cgdna/examples/``.

----------

Restrictions
""""""""""""

These pair styles can only be used if LAMMPS was built with the
CG-DNA package and the MOLECULE and ASPHERE package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`bond_style oxdna3/fene <bond_oxdna>`,
:doc:`bond_style oxdna/fene <bond_oxdna>`, :doc:`pair_style oxdna/excv <pair_oxdna>`,
:doc:`bond_style oxdna2/fene <bond_oxdna>`, :doc:`pair_style oxdna2/excv <pair_oxdna2>`,
:doc:`bond_style oxrna2/fene <bond_oxdna>`, :doc:`pair_style oxrna2/excv <pair_oxrna2>`,
:doc:`pair_coeff <pair_coeff>`, :doc:`atom_style oxdna <atom_style>`,
:doc:`fix nve/dotc/langevin <fix_nve_dotc_langevin>`

Default
"""""""

none

----------

.. _Bonato2:

**(Bonato)** A. Bonato, T.E. Ouldridge, A.A. Louis, J.P.K. Doye, L. Rovigatti, M. Matthies, O.Henrich, in preparation.

.. _Ouldridge-DPhil4:

**(Ouldridge-DPhil)** T.E. Ouldridge, Coarse-grained modelling of DNA and DNA self-assembly, DPhil. University of Oxford (2011).

.. _Ouldridge4:

**(Ouldridge)** T.E. Ouldridge, A.A. Louis, J.P.K. Doye, J. Chem. Phys. 134, 085101 (2011).

.. _Henrich6:

**(Henrich)** O. Henrich, Y. A. Gutierrez-Fosado, T. Curk, T. E. Ouldridge, Eur. Phys. J. E 41, 57 (2018).

