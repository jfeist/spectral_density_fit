Installation
============

Requirements
------------

The package requires Python 3.7 or later, along with the following dependencies:

* numpy ≥ 1.15
* jax ≥ 0.3
* nlopt ≥ 2.7.0

Installing from GitHub
----------------------

To install the latest version directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/jfeist/spectral_density_fit

Development Installation
------------------------

If you want to develop the package, we recommend using `uv <https://docs.astral.sh/uv>`_ to manage the project and dependencies:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/jfeist/spectral_density_fit.git
       cd spectral_density_fit

2. Install dependencies and run tests:

   .. code-block:: bash

       uv run pytest

3. Run linting and formatting:

   .. code-block:: bash

       uv run ruff check
       uv run ruff format

Configuring JAX Precision
--------------------------

.. important::

    The package requires 64-bit precision for accurate fitting. You must configure JAX **before** importing or using the package:

    .. code-block:: python

        import jax
        jax.config.update("jax_enable_x64", True)

    Without this configuration, you will receive a runtime warning, and fitting accuracy may be reduced.
