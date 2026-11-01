Installation
============

pgmpy supports Python 3.10 through 3.14. For most users, the base PyPI install is enough:

.. code-block:: bash

   pip install pgmpy


PyPI
----

Install the latest published release from PyPI:

.. code-block:: bash

   pip install pgmpy

If you also want optional dependency groups, install pgmpy with extras. Quotes
around the requirement are recommended and may be required in shells such as
``zsh``.

.. list-table::
   :header-rows: 1
   :widths: 13 35 52

   * - Extra
     - Command
     - Includes
   * - ``torch``
     - ``pip install "pgmpy[torch]"``
     - PyTorch and Pyro support for the torch backend and functional/hybrid
       workflows.
   * - ``optional``
     - ``pip install "pgmpy[optional]"``
     - Everything in ``torch``, plus optional packages such as ``daft-pgm``,
       ``xgboost``, ``litellm``, and ``pygraphviz``.
   * - ``tests``
     - ``pip install "pgmpy[tests]"``
     - The test and development toolchain, including ``pytest``,
       ``coverage``, ``black``, and ``pre-commit``.
   * - ``docs``
     - ``pip install "pgmpy[docs]"``
     - The documentation toolchain, including Sphinx, MyST, nbsphinx,
       sphinx-design, and the pydata-sphinx theme.
   * - ``all``
     - ``pip install "pgmpy[all]"``
     - The combined optional, testing, and documentation dependencies.


Conda
-----

Install the base package from conda-forge:

.. code-block:: bash

   conda install -c conda-forge pgmpy

If you need the optional dependency groups listed above, install them with
``pip`` inside the same environment after creating it.


Development Version
-------------------

To install the latest ``dev`` branch directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/pgmpy/pgmpy.git@dev


Editable Install From Source
----------------------------

If you are contributing to pgmpy or working on the documentation locally, clone
the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/pgmpy/pgmpy.git
   cd pgmpy

For a local development install with the test dependencies:

.. code-block:: bash

   pip install -e .[tests]        # bash
   pip install -e ".[tests]"      # zsh

The same pattern works for the other extras:

.. code-block:: bash

   pip install -e ".[docs]"
   pip install -e ".[torch]"
   pip install -e ".[optional]"
   pip install -e ".[all]"
