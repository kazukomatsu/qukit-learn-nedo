Introduction
============

What is qukit-learn?
--------------------
| qukit-learn is **a quantum machine learning toolkit** with *scikit-learn-like* interface.
| You can easily use qukit-learn just like you use scikit-learn.

Supported Algorithms
--------------------

qukit-learn provides the following algorithms:

- Binary Clustering
- Combinatorial Clustering
- Consensus Clustering
- Constrained Clustering
- Kernel Clustering
- BiDViT
- Linear Regression
- :abbr:`SVC (Support Vector Classification)`

For detailed documentation, please see :doc:`modules`.

Usage Example
----------------
As an example, we demonstrate how to use the :abbr:`SVC (Support Vector Classification)` algorithm.

- In qukit-learn, the :abbr:`SVC (Support Vector Classification)` algorithm can be used as follows:

.. code-block::

    >>> from qklearn.svm import SVC
    >>> from qklearn.utils import read_token
    >>> from amplify import FixstarsClient
    >>> 
    >>> X_train = [[1, 2, 3], [4, 5, 2], [7, 8, 9]]
    >>> t_train = [1, -1, 1]
    >>> X_test = [[3, 4, 2]]
    >>> client = FixstarsClient()
    >>> client.token = read_token("Fixstars")
    >>> 
    >>> model = SVC()
    >>> model.fit(X_train, t_train, client)
    >>> y_pred = model.predict(X_test)

- For your information, in qukit-learn, it can be used as follows:

.. code-block::
    
    >>> from sklearn.svm import SVC
    >>> 
    >>> X_train = [[1, 2, 3], [4, 5, 2], [7, 8, 9]]
    >>> t_train = [1, -1, 1]
    >>> X_test = [[3, 4, 2]]
    >>> 
    >>> model = SVC()
    >>> model.fit(X_train, t_train)
    >>> y_pred = model.predict(X_test)