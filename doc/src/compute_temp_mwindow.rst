.. index:: compute temp/mwindow
.. index:: compute temp/mwindow/kk

compute temp/mwindow command
============================

Accelerator Variants: *temp/mwindow/kk*

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID temp/mwindow vx vy vz

* ID, group-ID are documented in :doc:`compute <compute>` command
* temp/mwindow = style name of this compute command
* vx, vy, vz = components of the bulk (bias) velocity to subtract (velocity units)

Examples
""""""""

.. code-block:: LAMMPS

   compute mytemp all temp/mwindow 1000.0 0.0 0.0
   compute wtemp shock temp/mwindow 500.0 0.0 0.0

Description
"""""""""""

.. versionadded:: TBD

Define a computation that calculates the temperature of a group of
atoms after subtracting a fixed bulk (bias) velocity **(vx, vy, vz)**
from each atom.  This is useful in shock-wave simulations where a
moving window technique is employed and atoms in the window travel at a
known bulk velocity.  The bias velocity represents the overall
center-of-mass motion of the window and is subtracted before the
thermal kinetic energy is computed.

The temperature is calculated by the formula

.. math::

   KE = \frac{\text{dim}}{2} N k_B T

where *KE* is the total kinetic energy of the group of atoms after
subtracting the bias velocity, *dim* is the dimensionality of the
simulation, *N* is the number of atoms in the group, :math:`k_B` is
the Boltzmann constant, and *T* is the resulting temperature.

A 6-component kinetic energy tensor is also calculated by this
compute.  The formula for the *xy* component of the tensor (and
similarly for *xx*, *yy*, *zz*, *xz*, *yz*) is

.. math::

   T_{xy} = \sum_{i \in \text{group}} m_i (v_{ix} - v_{bx})(v_{iy} - v_{by})

where :math:`v_{bx}` and :math:`v_{by}` are the *x* and *y* components
of the bias velocity.

The number of degrees of freedom (DOF) for the temperature calculation
follows the standard formula: :math:`\text{DOF} = \text{dim} \times N -
\text{extra\_dof} - \text{fix\_dof}`, where *extra_dof* defaults to
dim (for a 3-D system) to remove the center-of-mass degrees of freedom,
and *fix_dof* counts constraints from fixes such as rigid bodies.

This compute also implements the
:doc:`velocity bias <Howto_thermostat>` interface so that it can be
used as the *bias* compute with thermostat fixes such as
:doc:`fix nvt <fix_nh>`, :doc:`fix temp/rescale <fix_temp_rescale>`,
and :doc:`fix temp/csvr <fix_temp_csvr>`.
The ``remove_bias`` and ``restore_bias`` methods subtract and add back
the fixed bias velocity **(vx, vy, vz)**.

----------

.. include:: accel_styles.rst

----------

Output info
"""""""""""

This compute calculates a global scalar (the temperature) and a global
vector of length 6 (the symmetric kinetic energy tensor), which can be
accessed by indices 1--6.  These values can be used by any command
that uses global scalar or vector values from a compute as input.  See
the :doc:`Howto output <Howto_output>` page for an overview of LAMMPS
output options.

The scalar value calculated by this compute is "intensive".  The
vector values are "extensive".

The scalar value is in temperature :doc:`units <units>`.  The vector
values are in energy :doc:`units <units>`.

Restrictions
""""""""""""

This compute style is part of the SHOCK package.  It is only enabled
if LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`compute temp <compute_temp>`,
:doc:`fix nvt <fix_nh>`,
:doc:`fix temp/rescale <fix_temp_rescale>`

Default
"""""""

none
