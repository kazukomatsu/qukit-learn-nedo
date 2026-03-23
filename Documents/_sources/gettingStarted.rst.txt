Getting Started
===============
This page demonstrates the usage of qukit-learn with examples.

qukit-learn is a quantum machine learning library that
supports **clustering**, **regression**, and **classification**.
This allows you to apply quantum approaches to various machine learning problems.

Preparation for Using qukit-learn
---------------------------------
To use qukit-learn, you need to create a solver client of Fixstars Amplify.
Please follow these steps:

#. Place the ``amplify-license.yaml`` in your current directory.
#. Execute the following process.

.. code-block::

    >>> from qklearn.utils import read_token
    >>> from amplify import FixstarsClient
    >>> from datetime import timedelta
    >>> 
    >>> client = FixstarsClient()
    >>> client.token = read_token("Fixstars")
    >>> client.parameters.timeout = timedelta(milliseconds=1000)

Please refer to :ref:`qklearn_utils_read_token` for more details on ``read_token()``.

Clustering
----------
qukit-learn provides six types of clustering modules:

- Binary Clustering
- Combinatorial Clustering
- Consensus Clustering
- Constrained Clustering
- Kernel Clustering
- BiDViT

In this section, we present examples for the following modules:

- Binary Clustering
- Combinatorial Clustering
- Constrained Clustering
- Kernel Clustering
- BiDViT

For details on Consensus Clustering and other modules, please refer to :ref:`qklearn_cluster`.

Binary Clustering
^^^^^^^^^^^^^^^^^
Classify samples into two clusters.

.. code-block::

    >>> from qklearn.cluster import BinaryClustering
    >>> from sklearn.datasets import make_blobs
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> n_clusters = 2
    >>> n_points = 10
    >>> data, labels_origin = make_blobs(random_state=8,
    ...                             n_samples=n_points,
    ...                             n_features=2, 
    ...                             cluster_std=1.5,
    ...                             centers=n_clusters)
    >>> 
    >>> qcl = BinaryClustering()
    >>> labels_qcl = qcl.fit_predict(data, client)
    >>> 
    >>> plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    >>> plt.show()

.. image:: img/binary_clustering.png

Combinatorial Clustering
^^^^^^^^^^^^^^^^^^^^^^^^
Classify samples into the specified number of clusters.

.. code-block::

    >>> from qklearn.cluster import CombinatorialClustering
    >>> from sklearn.datasets import make_blobs
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> n_clusters = 3
    >>> n_points = 9
    >>> data, labels = make_blobs(random_state=8,
    ...                         n_samples=n_points,
    ...                         n_features=2, 
    ...                         cluster_std=1.5,
    ...                         centers=n_clusters)
    >>> 
    >>> qcl = CombinatorialClustering(n_clusters=3)
    >>> labels_qcl = qcl.fit_predict(data, client)
    >>> 
    >>> plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    >>> plt.show()

.. image:: img/combinatorial_clustering/combinatorial_clustering.png

If you want to use a distance matrix insterd of a data matrix, 
please specify ``if_dist = True``.

.. code-block::

    >>> from qklearn.cluster import CombinatorialClustering
    >>> from sklearn.datasets import make_blobs
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial.distance import cdist
    >>> 
    >>> n_clusters = 3
    >>> n_points = 9
    >>> data, labels = make_blobs(random_state=8,
    ...                         n_samples=n_points,
    ...                         n_features=2, 
    ...                         cluster_std=1.5,
    ...                         centers=n_clusters)
    >>> 
    >>> dist = cdist(data, data, metric="euclidean")
    >>> 
    >>> qcl = CombinatorialClustering(n_clusters=3)
    >>> # labels_qcl = qcl.fit_predict(data, client)
    >>> qcl.fit(dist, if_dist=True)  # instead of above
    >>> labels_qcl = qcl.predict(client)
    >>> 
    >>> plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    >>> plt.show()

.. image:: img/combinatorial_clustering/combinatorial_clustering_if_dist.png

Constrained Clustering
^^^^^^^^^^^^^^^^^^^^^^
Classify the input data into the specified number of clusters under constraints.

- Preparing the input data.

.. code-block::

    >>> from qklearn.cluster import ConstrainedClustering
    >>> from sklearn.datasets import make_blobs
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> n_clusters = 3
    >>> n_points = 9
    >>> data, labels = make_blobs(random_state=8,
    ...                         n_samples=n_points,
    ...                         n_features=2, 
    ...                         cluster_std=1.5,
    ...                         centers=n_clusters)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_data.png

This module supports nine types of constraints:

#. ``add_must_link_to_qubo()``  
    Enforces specified pairs of samples to belong to the same cluster.

#. ``set_must_link_by_qbits_reduction()``  
    Enforces specified pairs of samples to belong to the same cluster, with reduced number of instances.

#. ``add_cannot_link_to_qubo()``  
    Enforces specified pairs of samples to belong to different clusters.

#. ``add_partition_level_to_qubo()``  
    Assigns specified samples to specified clusters.

#. ``set_partition_level_by_qbits_reduction()``  
    Assigns specified samples to specified clusters, with reduced number of instances.

#. ``add_non_partition_level_to_qubo()``  
    Prevents specified samples from being assigned to specified clusters.

#. ``set_non_partition_level_by_qbits_reduction()``  
    Prevents specified samples from being assigned to specified clusters, with reduced number of instances.

#. ``add_balanced_sizes_to_qubo()``  
    Equalizes the number of assigned samples per cluster.

#. ``add_limited_sizes_to_qubo()``  
    Limits the number of assigned samples per cluster to the specified value.

Now, let's look at usage examples for the constraints.

1. Setting ``add_must_link_to_qubo()`` and performing clustering.
    The figure below shows each pair of samples (0 and 1, 2 and 3, 4 and 5) belongs to the same cluster.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_must_link_to_qubo([(0,1), (2,3), (4,5)])  # set must-link constraints
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...    plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    >>>     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                            markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    >>>     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_set_must.png

2. Setting ``set_must_link_by_qbits_reduction()`` and performing clustering.
    The figure below shows each pair of samples (0 and 1, 2 and 3, 4 and 5) belongs to the same cluster.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.set_must_link_by_qbits_reduction([(0,1), (2,3), (4,5)])
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_set_must.png

3. Setting ``add_cannot_link_to_qubo()`` and performing clustering.
    The figure below shows each pair of samples (0 and 1, 2 and 3, 4 and 5) belongs to different clusters.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_cannot_link_to_qubo([(0,1), (2,3), (4,5)])
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_add_cannot.png

4. Setting ``add_partition_level_to_qubo()`` and performing clustering.
    The figure below shows samples 0, 2, and 4 are assigned to cluster 0, and samples 1, 3, and 6 are assigned to cluster 1.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_partition_level_to_qubo({0:(0,2,4),1:(1,3,6)})
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_add_partition.png

5. Setting ``set_partition_level_by_qbits_reduction()`` and performing clustering.
    The figure below shows samples 0, 2, and 4 are assigned to cluster 0, and samples 1, 3, and 6 are assigned to cluster 1.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.set_partition_level_by_qbits_reduction({0:(0,2,4),1:(1,3,6)})
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_set_partition.png

6. Setting ``add_non_partition_level_to_qubo()`` and performing clustering.
    The figure below shows samples 0, 2, and 4 are not assigned to cluster 0, and samples 1, 3, and 6 are not assigned to cluster 1.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_non_partition_level_to_qubo({0:(0,2,4),1:(1,3,6)})
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_add_non_partition.png

7. Setting ``set_non_partition_level_by_qbits_reduction()`` and performing clustering.
    The figure below shows samples 0, 2, and 4 are not assigned to cluster 0, and samples 1, 3, and 6 are not assigned to cluster 1.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.set_non_partition_level_by_qbits_reduction({0:(0,2,4),1:(1,3,6)})
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_set_non_partition.png

8. Setting ``add_balanced_sizes_to_qubo()`` and performing clustering.
    The figure below shows the number of samples assigned to each cluster is equalized.

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_balanced_sizes_to_qubo()
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_add_balanced.png

9. Setting ``add_limited_sizes_to_qubo()`` and performing clustering.
    The figure below shows the number of samples assigned to each cluster is limited according to the specified values
    (e.g., cluster 0: 2 samples, cluster 1: 2 samples, cluster 2: 5 samples).

.. code-block::

    >>> ccl = ConstrainedClustering(n_clusters=3)
    >>> ccl.fit(data)
    >>> ccl.add_limited_sizes_to_qubo({0:2, 1:2, 2:5})  # set a constraint
    >>> labels_ccl = ccl.predict(client)
    >>> 
    >>> # plot labels
    >>> scatter = plt.scatter(data[:,0], data[:,1], c=labels_ccl)
    >>> for i, (x, y) in enumerate(data):
    ...     plt.text(x - 0.3, y - 0.3, str(i), fontsize=12, ha='right', va='bottom')
    >>> 
    >>> # show legend for each label
    >>> handles = []
    >>> labels = []
    >>> for label in set(labels_ccl):
    ...     handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
    ...                     markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10))
    ...     labels.append(f'Cluster {label}')
    >>> plt.legend(handles=handles, title="Cluster Labels")
    >>> plt.show()

.. image:: img/constrained_clustering/const_clustering_add_limited.png

Kernel Clustering
^^^^^^^^^^^^^^^^^
Classify samples into the specified number of clusters by using Gaussian kernel.

.. code-block::

    >>> from qklearn.cluster import KernelClustering
    >>> from sklearn.datasets import make_circles
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> data, labels = make_circles(n_samples=64,
    ...                         factor=0.3,
    ...                         noise=0.05,
    ...                         random_state=0)
    >>> 
    >>> qcl = KernelClustering(n_clusters=2, sigma=0.2)
    >>> labels_qcl = qcl.fit_predict(data, client)
    >>> 
    >>> plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    >>> plt.show()

.. image:: img/kernel_clustering.png

BiDViT
^^^^^^
Perform hierarchical clustering on input data.

- Preparing the input data.

.. code-block::

    >>> from qklearn.cluster import BiDViT
    >>> from sklearn.datasets import make_blobs
    >>> import matplotlib.pyplot as plt
    >>>  
    >>> n_clusters = 9
    >>> n_points = 200
    >>> data, label = make_blobs(random_state=10,
    ...                         n_samples=n_points,
    ...                         n_features=2, 
    ...                         cluster_std=1.5,
    ...                         centers=n_clusters)
    >>> 
    >>> plt.scatter(data[:,0], data[:,1], c=label)

.. image:: img/BiDViT/BiDViT_data.png

- Fitting the clustering model and predicting using it.

.. code-block::

    >>> qcl = BiDViT()
    >>> tree = qcl.fit_predict(data, client)
    >>> tree.keys()
    dict_keys([186, 185, 183, 182, 179, 176, 171, 167, 159, 154, 147, 140, 130, 126, 124, 117, 110, 101, 88, 81, 72, 64, 52, 49, 46, 45, 44, 40, 37, 35, 32, 29, 28, 26, 25, 23, 21, 19, 16, 14, 12, 10, 9, 8, 5, 4, 3, 2])

- Plotting the results.

.. code-block::

    >>> plt.scatter(data[:,0], data[:,1], c=tree[101], cmap="prism")

.. image:: img/BiDViT/BiDViT_tree_101.png

.. code-block::

    >>> plt.scatter(data[:,0], data[:,1], c=tree[52], cmap="prism")

.. image:: img/BiDViT/BiDViT_tree_52.png

.. code-block::

    >>> plt.scatter(data[:,0], data[:,1], c=tree[25], cmap="prism")

.. image:: img/BiDViT/BiDViT_tree_25.png

.. code-block::

    >>> plt.scatter(data[:,0], data[:,1], c=tree[10], cmap="prism")

.. image:: img/BiDViT/BiDViT_tree_10.png

.. code-block::

    >>> plt.scatter(data[:,0], data[:,1], c=tree[3], cmap="prism")

.. image:: img/BiDViT/BiDViT_tree_3.png

Regression
----------
qukit-learn provides a linear regression module.
Please refer to :ref:`qklearn_linear` for more details.

Linear Regression
^^^^^^^^^^^^^^^^^
Perform linear regression analysis.

- Preparing the input data.

.. code-block::

    >>> from qklearn.linear_model import LinearRegression as QLR
    >>> from sklearn.linear_model import LinearRegression as SLR
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import mean_squared_error
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Generate dataset for regression
    >>> X, y = make_regression(n_samples=100,
    ...                     n_features=1,
    ...                     n_informative=1,
    ...                     bias=0.0,
    ...                     noise=20.0)
    >>> 
    >>> # Split into training and test data
    >>> X_train, X_test = X[:80], X[-20:]
    >>> y_train, y_test = y[:80], y[-20:]

- Adjusting the parameters.

    - To minimize the :abbr:`MSE (Mean Squared Error)` of the training and prediction results, the following parameters are adjusted.

        - ``num_elements``: Upper limit of exponent of ``basis``.
        - ``exponent_offset``: Parameter that adjusts exponent of ``basis``.
        - For your information, ``basis`` ia a basis for approximating the regression coefficients with binary variables.
    
    - In this example, the minimum MSE is obtained when ``num_elements`` is 9 and ``exponent_offset`` is 5.

.. code-block::

    >>> MSE_min = 10000
    >>> k_min = 0
    >>> e_min = 0
    >>> 
    >>> for k in range(3, 10):
    ...     MSE = np.zeros(10)
    ...     for i, e in enumerate([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]):
    ...         model = QLR()
    ...         model.fit(X_train, y_train, client, num_elements=k, exponent_offset=e)
    ...         y_pred = model.predict(X_test)
    ...         MSE[i] = mean_squared_error(y_test, y_pred)
    ...         if MSE_min >= MSE[i]:
    ...             MSE_min = MSE[i]
    ...             k_min = k
    ...             e_min = e
    ... 
    ...         fig = plt.figure()
    ...         ax = fig.add_subplot(111)
    ...         ax.set_title("the number of elements "+ str(k))
    ...         ax.set_xlabel("the exponent offset")
    ...         ax.set_ylabel("MSE")
    ...         ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ...         ax.scatter([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], MSE, c ='blue')
    ...         ax.plot([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], np.full(10, MSE_sc), c='green')
    >>> 
    >>> print("k_min ", k_min, ", e_min ", e_min, ", MSE_min ", MSE_min)
    k_min  9 , e_min  5 , MSE_min  663.3443227313044

.. image:: img/linear_regression/linear_reg_fitting_params_1.png
.. image:: img/linear_regression/linear_reg_fitting_params_2.png
.. image:: img/linear_regression/linear_reg_fitting_params_3.png
.. image:: img/linear_regression/linear_reg_fitting_params_4.png
.. image:: img/linear_regression/linear_reg_fitting_params_5.png
.. image:: img/linear_regression/linear_reg_fitting_params_6.png
.. image:: img/linear_regression/linear_reg_fitting_params_7.png

- Fitting the linear model and predicting using it.

.. code-block::

    >>> qlr = QLR()
    >>> qlr.fit(X_train, y_train, client, num_elements=k_min, exponent_offset=e_min)
    >>> 
    >>> # Predict using the created model (for training and validation data)
    >>> y_train_pred_qlr = qlr.predict(X_train)
    >>> y_test_pred_qlr = qlr.predict(X_test)
    >>> 
    >>> MSE = mean_squared_error(y_test, y_test_pred_qlr)
    >>> print(MSE)
    663.3443227313044

- Plotting the results.

.. code-block::

    >>> plt.scatter(X_train, y_train, label="Training")
    >>> plt.scatter(X_test, y_test, label="Test")
    >>> plt.scatter(X_train, y_train_pred_qlr, label="QLR (Training prediction)")
    >>> plt.scatter(X_test, y_test_pred_qlr, label="QLR (Test prediction)")
    >>> plt.legend()

.. image:: img/linear_regression/linear_reg.png

Classification
--------------
qukit-learn provides a :abbr:`SVC (Support Vector Classification)` module.
Please refer to :ref:`qklearn_svm` for more details.

SVC
^^^
Perform support vector classification.

- Preparing the input data.

.. code-block::

    >>> from qklearn.svm import SVC as QSVC
    >>> from sklearn.svm import SVC as SSVC
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import train_test_split
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> inputs,targets=make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.7)
    >>> targets[targets == 0] = -1
    >>> X_train, X_test, t_train, t_test = train_test_split(inputs,targets)
    >>> 
    >>> plt.scatter(inputs[:,0],inputs[:,1],c=targets, cmap='winter')
    >>> plt.title('Data points of the 2 classes')
    >>> plt.show()

.. image:: img/svm/svm_data.png

- Adjusting the parameters.

    - To maximize prediction accuracy, the following parameters are adjusted.

        - ``num_elements``: The number of ``basis`` used to approximate the resulting coefficients alpha of the training as binary variables, where alpha means the coefficients obtained by training. 
        - ``exponent_offset``: Parameter to adjust the exponential portion of ``basis``.
        - For your information, ``basis`` is a basis for approximating the resulting coefficients alpha of the training as binary variables.
    - In this example, the maximum accuracy is obtained when ``num_elements`` is 6 and ``exponent_offset`` is 14.

.. code-block::

    >>> acc_max = 0
    >>> k_max = 0
    >>> e_max = 0
    >>> list_exponent_offset = list(range(5, 15))
    >>> 
    >>> for k in range(3, 10):
    ...     acc = np.zeros(10)
    ...     for i, e in enumerate(list_exponent_offset):
    ...         model = QSVC()
    ...         model.fit(X_train, t_train, client, num_elements=k, exponent_offset=e, xi=1)
    ...         y_pred = model.predict(X_test)
    ...         acc[i] = accuracy_score(t_test, y_pred)
    ...         if acc_max <= acc[i]:
    ...             acc_max = acc[i]
    ...             k_max = k
    ...             e_max = e
    >>> 
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.set_title("the number of elements "+ str(k))
    >>> ax.set_xlabel("the exponent offset")
    >>> ax.set_ylabel("accuracy")
    >>> ax.set_xticks(list_exponent_offset)
    >>> ax.scatter(list_exponent_offset, acc, c ='blue')
    >>> ax.plot(list_exponent_offset, np.full(10, acc_sc), c='green')
    >>> 
    >>> print("k_max ", k_max, ", e_max ", e_max, ", acc_max ", acc_max)
    k_max  6 , e_max  14 , acc_max  1.0

.. image:: img/svm/svm_fitting_params_1.png
.. image:: img/svm/svm_fitting_params_2.png
.. image:: img/svm/svm_fitting_params_3.png
.. image:: img/svm/svm_fitting_params_4.png
.. image:: img/svm/svm_fitting_params_5.png
.. image:: img/svm/svm_fitting_params_6.png
.. image:: img/svm/svm_fitting_params_7.png

- Fitting the :abbr:`SVC (Support Vector Classification)` model and predicting using it.

.. code-block::

    >>> qsvc = QSVC()
    >>> qsvc.fit(X_train, t_train, client, num_elements=k_max, exponent_offset=e_max, xi=1)
    >>> 
    >>> # Predict using the created model (for training and validation data)
    >>> y_train_pred_qlr = qsvc.predict(X_train)
    >>> y_test_pred_qlr = qsvc.predict(X_test)

- Plotting the results.

.. code-block::

    >>> plt.scatter(X_train[:,0],X_train[:,1],c=y_train_pred_qlr, cmap='winter')
    >>> plt.title('Data points of the 2 classes')
    >>> plt.show()

.. image:: img/svm/svm_result_1.png

.. code-block::

    >>> plt.scatter(X_test[:,0],X_test[:,1],c=y_test_pred_qlr, cmap='winter')
    >>> plt.title('Data points of the 2 classes')
    >>> plt.show()

.. image:: img/svm/svm_result_2.png