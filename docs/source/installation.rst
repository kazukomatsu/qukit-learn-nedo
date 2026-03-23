Installing qukit-learn
======================

.. note::

    It is recommended to create a virtual environment
    before installing qukit-learn to avoid potential conflicts with other packages.

    ::

        $ python3.9 -m venv qklearn-env
        $ source qklearn-env/bin/activate

Package Installation
--------------------
::

   $ pip install qklearn-0.1.0-cp39-cp39-linux_x86_64.whl

..
    Installation via GitHub
    -----------------------
    Install
    ^^^^^^^
    ::

        $ pip install git+https://github.com/kumagaimasahito/qukit-learn.git

    Clone & Install
    ^^^^^^^^^^^^^^^
    ::

        $ git clone git@github.com:kumagaimasahito/qukit-learn.git
        $ cd qukit-learn
        $ pip install .

    Install & Test
    ^^^^^^^^^^^^^^
    ::

        $ cd qukit-learn
        $ python setup.py test (optionally)
..