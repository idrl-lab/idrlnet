Welcome to idrlnet's documentation!
===================================

.. toctree::
   :maxdepth: 2

   user/installation
   user/get_started/tutorial
   user/cite_idrlnet
   user/team

Features
--------

IDRLnet is a machine learning library on top of `Pytorch <https://www.tensorflow.org/>`_. Use IDRLnet if you need a machine
learning library that solves both forward and inverse partial differential equations (PDEs) via physics-informed neural
networks (PINN). IDRLnet is a flexible framework inspired by `Nvidia Simnet <https://developer.nvidia.com/simnet>`_.

IDRLnet supports

- complex domain geometries without mesh generation. Provided geometries include interval, triangle, rectangle, polygon,
  circle, sphere... Other geometries can be constructed using three boolean operations: union, difference, and
  intersection;
- sampling in the interior of the defined geometry or on the boundary with given conditions.
- enables the user code to be structured. Data sources, operations, constraints are all represented by ``Node``. The graph
  will be automatically constructed via label symbols of each node. Getting rid of the explicit construction via
  explicit expressions, users model problems more naturally.
- solving variational minimization problem;
- solving integral differential equation;
- adaptive resampling;
- recover unknown parameter of PDEs from noisy measurement data.

API reference
=============
If you are looking for usage of a specific function, class or method, please refer to the following part.

.. toctree::
   :maxdepth: 2


   modules/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
