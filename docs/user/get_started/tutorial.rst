Tutorial
========


To make full use of IDRLnet. We strongly suggest following the following examples:

1. :ref:`Simple Poisson <Solving Simple Poisson Equation>`. This example introduces the primary usage of IDRLnet. Including creating sampling domains, neural
   networks, partial differential equations, training, monitoring, and inference.
2. :ref:`Euler-Bernoulli beam <Euler–Bernoulli beam>`. The example introduces how to use symbols to construct a PDE node efficiently.
3. :ref:`Burgers' Equation <Burgers' Equation>`. The case presents how to include ``time`` in the sampling domains.
4. :ref:`Allen-Cahn Equation <Allen-Cahn Equation>`. The example introduces the representation of periodic boundary conditions.
   ``Receiver`` acting as ``callbacks`` are also introduced, including implementing user-defined algorithms and post-processing during the training.
5. :ref:`Inverse wave equation <Inverse Wave Equation>`. The example introduces how to discover unknown parameters in PDEs.
6. :ref:`Parameterized poisson equation <Parameterized Poisson>`. The example introduces how to train a surrogate with parameters.
7. :ref:`Variational Minimization <Variational Minimization>`. The example introduces how to solve variational minimization problems.
8. :ref:`Volterra integral differential equation <Volterra Integral Differential Equation>`. The example introduces the way to solve IDEs.



.. toctree::
   :maxdepth: 2

   1_simple_poisson
   2_euler_beam
   3_burgers_equation
   4_allen_cahn
   5_inverse_wave_equation
   6_parameterized_poisson
   7_minimal_surface
   8_volterra_ide
