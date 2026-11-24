Installation
============

pgmpy supports Python 3.10 to 3.15. For most users, the base install from PyPI is sufficient:

.. code-block:: bash

   pip install pgmpy


PyPI
----

Install the latest published release from PyPI:

.. code-block:: bash

   pip install pgmpy

To install optional dependency groups, install pgmpy with extras. Put quotes around the
requirement. Some shells, such as ``zsh``, need the quotes.

.. list-table::
   :header-rows: 1
   :widths: 13 35 52

   * - Extra
     - Command
     - Includes
   * - ``torch``
     - ``pip install "pgmpy[torch]"``
     - PyTorch and Pyro, for the torch backend and for functional and hybrid
       models.
   * - ``optional``
     - ``pip install "pgmpy[optional]"``
     - All packages in ``torch``, and optional packages such as ``daft-pgm``,
       ``xgboost``, ``litellm``, and ``pygraphviz``.
   * - ``tests``
     - ``pip install "pgmpy[tests]"``
     - The test and development tools, for example ``pytest``,
       ``coverage``, ``black``, and ``pre-commit``.
   * - ``docs``
     - ``pip install "pgmpy[docs]"``
     - The documentation tools, for example Sphinx, MyST, nbsphinx,
       sphinx-design, and the pydata-sphinx theme.
   * - ``all``
     - ``pip install "pgmpy[all]"``
     - All optional, test, and documentation dependencies.


Conda
-----

Install the base package from conda-forge:

.. code-block:: bash

   conda install -c conda-forge pgmpy

For the optional dependency groups, create the conda environment first. Then install the
groups with ``pip`` in the same environment.


Development Version
-------------------

To install the latest ``dev`` branch from GitHub:

.. code-block:: bash

   pip install git+https://github.com/pgmpy/pgmpy.git@dev


Editable Install From Source
----------------------------

To change the pgmpy code or documentation, clone the repository and install it in editable
mode:

.. code-block:: bash

   git clone https://github.com/pgmpy/pgmpy.git
   cd pgmpy

Install pgmpy for local development, with the test dependencies:

.. code-block:: bash

   pip install -e .[tests]        # bash
   pip install -e ".[tests]"      # zsh

Use the same command for the other extras:

.. code-block:: bash

   pip install -e ".[docs]"
   pip install -e ".[torch]"
   pip install -e ".[optional]"
   pip install -e ".[all]"
