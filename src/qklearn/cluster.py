"""
MIT License

Copyright © 2023-2025 Tohoku University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from scipy.spatial import distance
from .solver import BaseSolver
from .utils import min_max, standardization, measure, solution2labels, array_check
from BiDViT import get_chunk, get_qubo
import numpy as np
import copy
import collections
from .utils import get_kronecker_qubo_euclidean, get_kronecker_qubo_kernel
from .utils import constraint_check
import warnings
import inspect

def _scale_dist_impl(dist, scaling, lam=None):
    M = lam
    if scaling == 'normal' or (lam is None and scaling is None):
        dist = min_max(dist)
    elif scaling == 'divbymax':
        dist = dist / dist.max()
    elif scaling == 'normal_i':
        dist = min_max(dist, axis=1)
    elif scaling == 'standard':
        dist = standardization(dist)
    elif scaling == 'invalid':
        if M is not None:
            M *= dist.max()
    else:
        warnings.warn("scaling failed.")
    if lam is None:
        return dist
    return dist, M

class BinaryClustering(BaseSolver):
    """Classify samples into two clusters.
    """

    def __init__(self):
        self.timing = {}

    @measure
    def set_qubo(self, metric="euclidean", scaling="invalid"):
        self.dist = distance.cdist(self.data, self.data, metric=metric)
        dist = self._scale_dist(scaling=scaling)

        diag = np.diag(np.sum(dist, axis=0))
        triu = np.triu(dist, k=1)
        self.indexed_qubo = -2 * diag + 4 * triu

    def get_qubo(self):
        return self.indexed_qubo

    @measure
    def decode_solution(self):
        self.labels = self.solution

    def _scale_dist(self, scaling=None):
        return _scale_dist_impl(self.dist, scaling)

    def fit(self, X):
        """Generate QUBO for clustering from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of samples.

        Raises
        ------
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        """
        array_check(X, "X", 2)
        self.data = X
        self.n_points = len(self.data)
        self.set_qubo()

    def predict(self, client):
        """Perform clustering.

        Parameters
        ----------
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "indexed_qubo"):
            raise AttributeError("This instance is not fitted yet")
        self.solve(client)
        self.decode_solution()
        return self.labels
    
    def fit_predict(self, X, client):
        """ Generate QUBO for clustering from input data and perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of samples.
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.

        """
        self.fit(X)
        return self.predict(client)

class CombinatorialClustering(BaseSolver):
    """Classify samples into the specified number of clusters.

        Parameters
        ----------
        n_clusters : int
            Number of clusters. Must be greater than or equal to 1.
        lam : float, default=None
            The Lagrange multiplier weight for the one-hot constraint used to uniquely identify each cluster.
            If None, 'n_samples - n_clusters' is used as lam.

        Raises
        ------
        ValueError
            If variable 'n_clusters' was less than 1.
        TypeError
            If an invalid data type was specified for an argument.
    """
    def __init__(self, *, n_clusters, lam=None):
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be int")
        if (n_clusters < 1):
            raise ValueError("n_clusters must be greater than 0")
        self.n_clusters = n_clusters
        if (lam is not None):
            if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)):
                raise TypeError("lam must be float")
        self.lam = lam
        self.timing = {}

    @measure
    def set_qubo(self, metric='euclidean', scaling="normal", lam_rate=1):
        if self.if_dist == False:
            self.dist = distance.cdist(self.data, self.data, metric=metric)
        if self.lam is None:
                self.lam = self.n_points - self.n_clusters
        self.lam_rate = lam_rate
        dist, M = self._scale_dist(scaling=scaling)
        self.lam *= lam_rate

        self.indexed_qubo = get_kronecker_qubo_euclidean(dist, M, self.n_points, self.n_clusters)
            
    def get_qubo(self):
        return self.indexed_qubo
    
    @measure
    def decode_solution(self):
        tmp = {
                self.index2label[i]:q 
                for i,q in enumerate(self.solution)
        }
        self.labels = solution2labels(solution=tmp, n_clusters=self.n_clusters, n_points=self.n_points)
        if -1 in self.labels:
            raise RuntimeError("The cluster labels of the samples that do not satisfy the constraint are set to -1. Please execute the program multiple times or adjust the parameters.")

    def _scale_dist(self, scaling):
        return _scale_dist_impl(self.dist, scaling, lam=self.lam)

    def fit(self, X, if_dist=False):
        """Generate QUBO for clustering from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or array-like of shape(n_samples, n_samples)
            An array of samples, or a distance matrix between samples.
        if_dist : bool, default=False
            A bool indicating whether X is a distance matrix or not.
            If True, 'X' is assumed to be a distance matrix; otherwise, 'X' is treated as an array of samples.

        Raises
        ------
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        """
        array_check(X, "X", 2, if_dist=if_dist)
        X = np.array(X)
        self.if_dist = if_dist
        if self.if_dist:
            self.dist = X
            self.n_points = len(self.dist)
        else:
            self.data = X
            self.n_points = len(self.data)
        self._build_label_index_maps()
        self.set_qubo()

    def predict(self, client):
        """Perform clustering.

        Parameters
        ----------
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from Fixstars Amplify SDK.
        RuntimeError
            If the solution returned by an optimization solver does not satisfy the constraints.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "indexed_qubo"):
            raise AttributeError("This instance is not fitted yet")
        self.solve(client)
        self.decode_solution()
        constraint_check.cluster_number_constraint_check(self.labels, self.n_clusters)
        return self.labels
    
    def fit_predict(self, X, client, if_dist=False):
        """ Generate QUBO for clustering from input data and perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or array-like of shape(n_samples, n_samples)
            An array of samples, or a distance matrix between samples.
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.
        if_dist : bool, default=False
            A bool indicating whether X is a distance matrix or not.
            If True, 'X' is assumed to be a distance matrix; otherwise, 'X' is treated as an array of samples.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        RuntimeError
            If the solution returned by an optimization solver does not satisfy the constraints.
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        """
        self.fit(X, if_dist=if_dist)
        return self.predict(client)

class ConsensusClustering(BaseSolver):
    """Perform clustering by aggregating results from multiple clustering executions.

    Parameters
    ----------
    n_clusters : int
        Number of clusters. Must be greater than or equal to 1.
    lam : float, default=None
        The Lagrange multiplier weight for the one-hot constraint used to uniquely identify each cluster.
        If None, 'n_samples - n_clusters' is used as lam.
    model : {"pairwise_similarity-based", "partition_difference"}, default="pairwise_similarity-based"
        An aggregating model.

    Raises
    ------
    ValueError
        If variable 'n_clusters' was less than 1.
    ValueError
        If an invalid model was specified.
    TypeError
        If an invalid data type was specified for an argument.
    """
    _ACCEPTABLE_MODELS = ["pairwise_similarity-based", "partition_difference"]
    def __init__(self, *, n_clusters, lam=None, model="pairwise_similarity-based"):
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be int")
        if (n_clusters < 1):
            raise ValueError("n_clusters must be greater than 0")
        self.n_clusters = n_clusters
        if (not isinstance(model, str)):
            raise TypeError("model must be str")
        if (model not in ConsensusClustering._ACCEPTABLE_MODELS):
            raise ValueError(f"model {model} is invalid")
        self.model = model
        if (lam is not None):
            if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)):
                raise TypeError("lam must be float")
        self.lam = lam
        self.timing = {}

    @measure
    def set_qubo(self, lam_rate=1):
        if self.lam is None:
                self.lam = self.n_points - self.n_clusters
        self.lam *= lam_rate

        if self.model=="pairwise_similarity-based":
            self.qubo = {
                ((i,a),(j,b)) :
                1-self.similarity_matrix[(i,j)] if (i<j and a==b)
                else (-1)*self.lam if (i==j and a==b)
                else 2*self.lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            self.qubo = self._omit_zero_coefficients(self.qubo) # 非ゼロ要素だけのQUBO
            self.indexed_qubo = self.qubo_to_indexed_qubo()
        elif self.model=="partition_difference":
            self.qubo = {
                ((i,a),(j,b)) :
                1-self.similarity_matrix[(i,j)] if (i<j and a==b)
                else self.similarity_matrix[(i,j)] if  (i<j and a<b)
                else (-1)*self.lam if (i==j and a==b)
                else 2*self.lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            self.qubo = self._omit_zero_coefficients(self.qubo) # 非ゼロ要素だけのQUBO
        else:
            warnings.warn("Please select the Ising model of pairwise_similarity-based or partition_difference")

    def get_qubo(self):
        return self.qubo
    
    @measure
    def decode_solution(self):
        tmp = {
                self.index2label[i]:q 
                for i,q in enumerate(self.solution)
        }
        self.labels = solution2labels(solution=tmp, n_clusters=self.n_clusters, n_points=self.n_points)
        if -1 in self.labels:
            raise RuntimeError("The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1.")

    def fit_predict(self, X, client):
        """Generate QUBO for clustering from input data and perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_result, n_sample)
            An array of clustering results.
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        RuntimeError
            If the solution returned by an optimization solver does not satisfy the constraints.
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        """
        array_check(X, "X", 2)
        self.clusterings = X
        self.n_points = len(self.clusterings[0])
        self.n_clusterings = len(self.clusterings)
        self.lam = self.n_points - self.n_clusters
        self._build_label_index_maps()
        self.similarity_matrix = {
            (i,j) : sum([
                1 for labels_m in self.clusterings if labels_m[i]==labels_m[j]
            ]) / self.n_clusterings
            for i in range(0,self.n_points)
            for j in range(i,self.n_points)
        }

        self.set_qubo()
        self.indexed_qubo = self.qubo_to_indexed_qubo()
        self.solve(client)
        self.decode_solution()
        constraint_check.cluster_number_constraint_check(self.labels, self.n_clusters)
        
        return self.labels
    
class ConstrainedClustering(BaseSolver):
    """Classify the input data into the specified number of clusters under constraints.
    If a solution that satisfies the constraints cannot be found, the cluster to which the sample belongs may become -1.

    Parameters
    ----------
    n_clusters : int
        Number of clusters. Must be greater than or equal to 1.
    lam : float, default=None
        The Lagrange multiplier weight for the one-hot constraint used to uniquely identify each cluster.
        If None, 'n_samples - n_clusters' is used as lam.

    Raises
    ------
    ValueError
        If variable 'n_clusters' was less than 1.
    TypeError
        If an invalid data type was specified for an argument.
    """
    def __init__(self, *, n_clusters, lam=None):
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be int")
        if (n_clusters < 1):
            raise ValueError("n_clusters must be greater than 0")
        self.n_clusters = n_clusters
        if (lam is not None):
            if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)):
                raise TypeError("lam must be float")
        self.lam = lam
        self.timing = {}
        self.deleted_qubits = tuple()

    @measure
    def set_qubo(self, metric='euclidean', scaling="normal", lam_rate=1):
        self.dist = distance.cdist(self.data, self.data, metric=metric)
        if self.lam is None:
                self.lam = self.n_points - self.n_clusters
        self.lam *= lam_rate
        dist, M = self._scale_dist(scaling=scaling)

        self.qubo = {
            ((i,a),(j,b)) :
            dist[i,j] if (i<j and a==b)
            else (-1)*M if (i==j and a==b)
            else 2*M if (i==j and a<b)
            else 0
            for i in range(0,self.n_points) 
            for j in range(i,self.n_points)
            for a in range(0,self.n_clusters)
            for b in range(a,self.n_clusters)
        }
        self.qubo = self._omit_zero_coefficients(self.qubo) # 非ゼロ要素だけのQUBO
            
    def get_qubo(self):
        return self.qubo
    
    def _scale_dist(self, scaling):
        return _scale_dist_impl(self.dist, scaling, lam=self.lam)
    
    def fit(self, X):
        """Generate QUBO for clustering from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of samples.

        Raises
        ------
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        """
        array_check(X, "X", 2)
        self.data = X
        self.n_points = len(self.data)
        self._build_label_index_maps()
        self.set_qubo()
        self.constraints = {
            "must_link_instances": None,
            "cannot_link_instances": None,
            "partition_level_instances": None,
            "non_partition_level_instances": None,
            "limited_sizes": None,
            "balanced_sizes": None,
        }
        self.add_constraint_flag = False
        self.set_constraint_flag = False

    def predict(self, client):
        """Perform clustering.

        Parameters
        ----------
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.

        Returns
        -------
        labels : list of int
            Indices of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        RuntimeError
            If the solution returned by an optimization solver does not satisfy the constraints.
        ValueError
            If any contradictions are found in the set constraints.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        self.survived_label2index = {l: i for l, i in self.label2index.items() if l not in self.deleted_qubits}
        self.survived_label2re_index = {l: rei for rei, l in enumerate(self.survived_label2index.keys())}
        self.index2re_index = {self.label2index[l]: rei for l, rei in self.survived_label2re_index.items()}
        try:
            self.qubo = {(self.survived_label2re_index[ia], self.survived_label2re_index[jb]): coeff for (ia, jb), coeff in self.qubo.items()}
        except Exception as e:
            raise RuntimeError("Please set the instances in the order of partition_level -> non_partition_level -> must_link.") from e
        self.indexed_qubo = np.array(
            [
                [
                    0 if jb < ia
                    else 0 if (ia, jb) not in self.qubo
                    else self.qubo[(ia, jb)]
                    for jb in range(self.n_clusters * self.n_points)
                ]
                for ia in range(self.n_clusters * self.n_points)
            ]
        )

        constraint_check.contradiction_of_constraints_check(self.n_points, self.constraints)
        self.solve(client)
        self.decode_solution()
        reduced_clusters = [
            k for k, v in (self.constraints["limited_sizes"] or {}).items() if v == 0
        ]
        constraint_check.cluster_number_constraint_check(self.labels + reduced_clusters, self.n_clusters)
        constraint_check.instances_level_constraints_check(self.labels, self.constraints)

        return self.labels

    def _link_check(self, link_instances, arg_name):
        if (not isinstance(link_instances, list)):
            raise TypeError(f"{arg_name} must be list")
        for i in link_instances:
            if (not isinstance(i, tuple)):
                raise TypeError(f"{arg_name} must contain tuples")
            if len(i) != 2:
                raise ValueError(f"Length of tuples in {arg_name} must be two")
            for j in i:
                if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
                    raise TypeError(f"A index in {arg_name} must be int")
                if (j < 0 or j >= self.n_points):
                    raise ValueError(f"A index in {arg_name} must be greater than or equal to 0 and less than number of samples.")


    @measure
    def add_must_link_to_qubo(self, must_link_instances, weight_must_link=None):
        """Add a constraint to enforce specified pairs of samples belonging to the same cluster.

        Parameters
        ----------
        must_link_instances : list of tuples of (int, int)
            A list of pairs of sample indices that should belong to the same cluster. Each pair is represented as a tuple of two integers, e.g., `must_link_instances` could be `[(0, 1), (4, 6), (5, 9), ... , (20, 35)]`.
        weight_must_link : float, default=None
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        ValueError
            If tuples in must_link_instances contains an incorrect number of indices.
        ValueError
            If index in must_link_instances was less than 0.
        ValueError
            If index in must_link_instances was greater than number of samples.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_must_link is not None):
            if isinstance(weight_must_link, bool) or not isinstance(weight_must_link, (int, float, np.integer, np.floating)):
                raise TypeError("weight_must_link must be float")
        self._link_check(must_link_instances, "must_link_instances")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)

        self.weight_must_link = self.lam if weight_must_link is None else weight_must_link
        self.must_link_qubo = {
            ((i, k), (i, k)) if inc == 0 else ((j, k), (j, k)) if inc == 1 else ((i, k), (j, k)) : 
            self.weight_must_link if inc != 2 else -2*self.weight_must_link
            for k in range(self.n_clusters) 
            for (i, j) in must_link_instances
            for inc in range(3)
        }
        self.must_link_qubo = self._omit_zero_coefficients(self.must_link_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.must_link_qubo)

        if self.constraints["must_link_instances"] is None:
            self.constraints["must_link_instances"] = must_link_instances
        else:
            self.constraints["must_link_instances"] = list(set(tuple(sorted(link)) for link in (self.constraints["must_link_instances"] + must_link_instances)))

    @measure
    def add_cannot_link_to_qubo(self, cannot_link_instances, weight_cannot_link=None):
        """Add a constraint to enforce specified pairs of samples belonging to the different cluster.

        Parameters
        ----------
        cannot_link_instances : list of tuples of (int, int)
            A list of pairs of sample indices that should belong to the different cluster. Each pair is represented as a tuple of two integers, e.g., `cannot_link_instances` could be `[(0, 1), (4, 6), (5, 9), ... , (20, 35)]`.
        weight_cannot_link : float, default=None
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        ValueError
            If tuples in cannot_link_instances contains an incorrect number of indices.
        ValueError
            If a index in cannot_link_instances was less than 0.
        ValueError
            If a Index in cannot_link_instances was greater than n_samples.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_cannot_link is not None):
            if isinstance(weight_cannot_link, bool) or not isinstance(weight_cannot_link, (int, float, np.integer, np.floating)):
                raise TypeError("weight_cannot_link must be float")
        self._link_check(cannot_link_instances, "cannot_link_instances")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.weight_cannot_link = self.lam if weight_cannot_link is None else weight_cannot_link
        self.cannot_link_qubo = {
            ((i, k), (j, k)) : self.weight_cannot_link
            for k in range(self.n_clusters) 
            for (i, j) in cannot_link_instances
        }
        self.cannot_link_qubo = self._omit_zero_coefficients(self.cannot_link_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.cannot_link_qubo)

        if self.constraints["cannot_link_instances"] is None:
            self.constraints["cannot_link_instances"] = cannot_link_instances
        else:
            self.constraints["cannot_link_instances"] = list(set(tuple(sorted(link)) for link in (self.constraints["cannot_link_instances"] + cannot_link_instances)))

    def _partition_level_check(self, partition_level_instances, arg_name):
        if (not isinstance(partition_level_instances, dict)):
            raise TypeError(f"{arg_name} must be dict")
        for k, v in partition_level_instances.items():
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
                raise TypeError(f"Keys of {arg_name} must be int")
            if (k < 0 or k >= self.n_clusters):
                raise ValueError(f"Keys of {arg_name} must be greater than or equal to 0 and less than number of clusters.")
            if (not isinstance(v, tuple)):
                raise TypeError(f"Values of {arg_name} must be tuples containing indices of samples.")
            for i in v:
                if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
                    raise TypeError(f"Indices in {arg_name} must be int")
                if (i < 0 or i >= self.n_points):
                    raise ValueError(f"Indices of {arg_name} must be greater than or equal to 0 and less than number of samples.")


    @measure
    def add_partition_level_to_qubo(self, partition_level_instances, weight_partition_level=None):
        """Add a constraint to assign specified samples to specified clusters.

        Parameters
        ----------
        partition_level_instances : dict of {int : tuple of (int, ...)}
            Dictionary of samples to assign to each cluster.
            Key is index of cluster, and value is list containing indices of samples belonging to those clusters.
        weight_partition_level : float, default=None
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        ValueError
            If an index in partition_level_instances was les than 0.
        ValueError
            If a cluster index in partition_level_instances was greater than n_clusters.
        ValueError
            If a sample index in partition_level_instances was greater than n_samples.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_partition_level is not None):
            if isinstance(weight_partition_level, bool) or not isinstance(weight_partition_level, (int, float, np.integer, np.floating)):
                raise TypeError("weight_partition_level must be float")
        self._partition_level_check(partition_level_instances,
                                    "partition_level_instances")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.weight_partition_level = self.lam if weight_partition_level is None else weight_partition_level
        self.partition_level_qubo = {
            ((i, k), (i, k)) : self.weight_partition_level
            for k in range(self.n_clusters)
            for m, instances in partition_level_instances.items()
            for i in instances
            if k != m
        }
        self.partition_level_qubo = self._omit_zero_coefficients(self.partition_level_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.partition_level_qubo)

        if self.constraints["partition_level_instances"] is None:
            self.constraints["partition_level_instances"] = partition_level_instances
        else:
            self.constraints["partition_level_instances"] = self.__merge_instances(partition_level_instances, "partition_level_instances")
    
    @measure
    def add_non_partition_level_to_qubo(self, non_partition_level_instances, weight_non_partition_level=None):
        """Add a constraint not to assign specified samples to specified clusters.

        Parameters
        ----------
        non_partition_level_instances : dict of {int : tuple of (int, ...)}
            Dictionary of samples not to assign to each cluster.
            Key is index of cluster, and value is list containing indices of samples which does not belong to those clusters.
        weight_non_partition_level : float, default=None
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        ValueError
            If an index in non_partition_level_instances was les than 0.
        ValueError
            If a cluster index in non_partition_level_instances was greater than n_clusters.
        ValueError
            If a sample index in non_partition_level_instances was greater than n_samples.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_non_partition_level is not None):
            if isinstance(weight_non_partition_level, bool) or not isinstance(weight_non_partition_level, (int, float, np.integer, np.floating)):
                raise TypeError("weight_non_partition_level must be float")
        self._partition_level_check(non_partition_level_instances,
                                    "non_partition_level_instances")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.weight_non_partition_level = self.lam if weight_non_partition_level is None else weight_non_partition_level
        self.non_partition_level_qubo = {
            ((i, m), (i, m)) : self.weight_non_partition_level
            for m, instances in non_partition_level_instances.items()
            for i in instances
        }
        self.non_partition_level_qubo = self._omit_zero_coefficients(self.non_partition_level_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.non_partition_level_qubo)

        if self.constraints["non_partition_level_instances"] is None:
            self.constraints["non_partition_level_instances"] = non_partition_level_instances
        else:
            self.constraints["non_partition_level_instances"] = self.__merge_instances(non_partition_level_instances, "non_partition_level_instances")

    @measure
    def add_balanced_sizes_to_qubo(self, weight_balanced_size=None):
        """Add a constraint to equalize the number of assigned samples per cluster.

        Parameters
        ----------
        weight_balanced_size : float, default=None
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        TypeError
            If an invalid data type was specified for an argument.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        ValueError
            If this method was used in combination with add_limited_sizes_to_qubo().
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_balanced_size is not None):
            if isinstance(weight_balanced_size, bool) or not isinstance(weight_balanced_size, (int, float, np.integer, np.floating)):
                raise TypeError("weight_balanced_size must be float")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        if self.constraints["limited_sizes"] is not None:
            raise ValueError("limited_sizes constraint and balanced_sizes constraint cannot be specified at the same time.")
        
        self.weight_balanced_size = self.lam if weight_balanced_size is None else weight_balanced_size
        self.balanced_sizes_qubo = {
            ((i, k), (j, k)) : self.weight_balanced_size*(1-2*self.n_points/self.n_clusters) if i == j else 2*self.weight_balanced_size
            for i in range(self.n_points)
            for j in range(i, self.n_points)
            for k in range(self.n_clusters)
        }
        self.balanced_sizes_qubo = self._omit_zero_coefficients(self.balanced_sizes_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.balanced_sizes_qubo)

        base = self.n_points // self.n_clusters
        extra = self.n_points % self.n_clusters
        if extra == 0:
            self.constraints["balanced_sizes"] = (base,)
        else:
            self.constraints["balanced_sizes"] = (base + 1, base)

    @measure
    def add_limited_sizes_to_qubo(self, limited_sizes, weight_limited_sizes=None):
        """Add a constraint to enforce the number of assigned samples per cluster does not exceed the specified number.

        Parameters
        ----------
        limited_sizes : dict of {int : int}
            Dictionary specifying the maximum number of samples per cluster.
            Key is index of cluster, and value is Upper limit of number of assigned samples.
            Please specify the limit for all clusters.
        weight_limited_sizes : float, default=None.
            A weight of constraint. If None, 'lam' will be used as weight.

        Raises
        ------
        ValueError
            If an upper limit in limited_sizes was less than 0.
        ValueError
            If a cluster index in limited_sizes was less than 0.
        ValueError
            If a cluster index in limited_sizes was greater than n_clusters.
        ValueError
            If there is a cluster for which no upper limit is specified.
        ValueError
            If this method was used in combination with set_*_by_qbits_reduction().
        ValueError
            If this method was used in combination with add_balanced_sizes_to_qubo().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        if (weight_limited_sizes is not None):
            if isinstance(weight_limited_sizes, bool) or not isinstance(weight_limited_sizes, (int, float, np.integer, np.floating)):
                raise TypeError("weight_limited_sizes must be float")
        if (not isinstance(limited_sizes, dict)):
            raise TypeError("limited_sizes must be dict")
        for k, v in limited_sizes.items():
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError("Cluster indices and upper limit in limited_sizes must be int")
            if (k < 0 or v < 0):
                raise ValueError("Cluster indices and upper limit in limited_sizes must be greater than or equal to 0")
            if (k >= self.n_clusters):
                raise ValueError("A Cluster index exceeds the number of clusters.")
        if (len(list(limited_sizes.keys())) != self.n_clusters):
            raise ValueError("Specify upper limit for all clusters")

        self.add_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        if self.constraints["balanced_sizes"] is not None:
            raise ValueError("limited_sizes constraint and balanced_sizes constraint cannot be specified at the same time.")
        
        self.weight_limited_sizes = self.lam if weight_limited_sizes is None else weight_limited_sizes
        self.limited_sizes_qubo = {
            ((i, k), (j, k)) : self.weight_limited_sizes*(1-2*limited_sizes[k]) if i == j else 2*self.weight_limited_sizes
            for i in range(self.n_points)
            for j in range(i, self.n_points)
            for k in range(self.n_clusters)
        }
        self.limited_sizes_qubo = self._omit_zero_coefficients(self.limited_sizes_qubo) # 非ゼロ要素だけのQUBO
        self.qubo = self._merge_dicts(self.qubo, self.limited_sizes_qubo)
        
        if self.constraints["limited_sizes"] is None:
            self.constraints["limited_sizes"] = limited_sizes
        else:
            self.constraints["limited_sizes"] = {i: min(size_i, limited_sizes[i]) for i, size_i in self.constraints["limited_sizes"].items()}

    @measure
    def set_must_link_by_qbits_reduction(self, must_link_instances):
        """Add a constraint to enforce specified pairs of samples belonging to the same cluster, and reduce the number of instances.
        This method cannot be used in combination with other add_*_to_qubo() or set_*_by_qbits_reduction() methods.
        In addition, it cannot be called consecutively. Doing so may cause unexpected errors.

        Parameters
        ----------
        must_link_instances : list of tuples of (int, int)
            A list of pairs of sample indices that must belong to the same cluster.

        Raises
        ------
        ValueError
            If tuples in must_link_instances contains an incorrect number of indices.
        ValueError
            If index in must_link_instances was less than 0.
        ValueError
            If index in must_link_instances was greater than number of samples.
        ValueError
            If this method was used in combination with add_*_to_qubo().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        self._link_check(must_link_instances, "must_link_instances")

        self.set_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.must_link_instances = must_link_instances
        self.organized_must_link_instances = constraint_check.organize_must_link_instances(must_link_instances)
        for oml in self.organized_must_link_instances: # organized_must_link_instancesから制約グループを一つずつ抽出
            i_surv = oml[-1] # 制約グループのうち生き残る要素のインデックス
            for i_omit in oml[:-1]: # 生き残る要素以外のイテレーション
                for a in range(0, self.n_clusters): # クラスタ方向にイテレーション
                    for b in range(a, self.n_clusters): # ワンホット係数がある場合の処理
                        if ((i_surv, a), (i_surv, b)) in self.qubo.keys() and ((i_omit, a), (i_omit, b)) in self.qubo.keys():
                            self.qubo[((i_surv, a), (i_surv, b))] += self.qubo.pop(((i_omit, a), (i_omit, b))) # 生き残るビット同士の係数にそれ以外を凝集
                    for i in range(self.n_points): # 距離係数に関する処理
                        surv = sorted((i_surv, i)) # QUBOは上三角行列なので，i<jになるようインデックスを並び替え
                        omit = sorted((i_omit, i)) # QUBOは上三角行列なので，i<jになるようインデックスを並び替え
                        if ((omit[0], a), (omit[1], a)) in self.qubo.keys():
                            if ((surv[0], a), (surv[1], a)) in self.qubo.keys():
                                self.qubo[((surv[0], a), (surv[1], a))] += self.qubo.pop(((omit[0], a), (omit[1], a))) # 生き残るビット同士の係数にそれ以外を凝集
                            else:
                                del self.qubo[((omit[0], a), (omit[1], a))] # 生き残るビットが存在していないことがあるので，その時は加算せずにただ消去
                    self.deleted_qubits += ((i_omit, a),)
        self.deleted_qubits = tuple(set(self.deleted_qubits))

        if self.constraints["must_link_instances"] is None:
            self.constraints["must_link_instances"] = must_link_instances
        else:
            self.constraints["must_link_instances"] = list(set(tuple(sorted(link)) for link in (self.constraints["must_link_instances"] + must_link_instances)))

    @measure
    def set_partition_level_by_qbits_reduction(self, partition_level_instances):
        """Add a constraint to assign specified samples to specified clusters, and reduce the number of instances.
        This method cannot be used in combination with other add_*_to_qubo() or set_*_by_qbits_reduction() methods.
        In addition, it cannot be called consecutively. Doing so may cause unexpected errors.

        Parameters
        ----------
        partition_level_instances : dict of {int : tuple of (int, ...)}
            Dictionary of samples to assign to each cluster.
            Key is index of cluster, and value is list containing indices of samples belonging to those clusters.

        Raises
        ------
        ValueError
            If an index in partition_level_instances was les than 0.
        ValueError
            If a cluster index in partition_level_instances was greater than n_clusters.
        ValueError
            If a sample index in partition_level_instances was greater than n_samples.
        ValueError
            If this method was used in combination with add_*_to_qubo().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        self._partition_level_check(partition_level_instances,
                                    "partition_level_instances")

        self.set_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.partition_level_instances = partition_level_instances
        del_list = ()
        for cl, pls in partition_level_instances.items(): # partition_levelから制約グループを一つずつ抽出
            for i_omit in pls: # 制約グループのうち消える要素をイテレーション
                for i in range(self.n_points): # データ点数分のイテレーション
                    if i != i_omit: # i == i_omitの場合のみ，0にはならないが定数項になるので排除
                        omit = sorted((i_omit, i)) # i<jになるような並び替え
                        if ((i, cl), (i, cl)) in self.qubo.keys() and ((omit[0], cl), (omit[1], cl)) in self.qubo.keys():
                            self.qubo[((i, cl), (i, cl))] += self.qubo.pop(((omit[0], cl), (omit[1], cl))) # 生き残る距離係数だけ，対角成分に凝集
                self.deleted_qubits += tuple((i_omit, a) for a in range(self.n_clusters))
            del_list += tuple(pls) # この後，消える係数を一括で消すために，タプルに保持
        self.qubo = {k: v for k, v in self.qubo.items() if k[0][0] not in del_list and k[1][0] not in del_list} # 消える係数を一括で消去
        self.deleted_qubits = tuple(set(self.deleted_qubits))
        
        if self.constraints["partition_level_instances"] is None:
            self.constraints["partition_level_instances"] = partition_level_instances
        else:
            self.constraints["partition_level_instances"] = self.__merge_instances(partition_level_instances, "partition_level_instances")
        
    @measure
    def set_non_partition_level_by_qbits_reduction(self, non_partition_level_instances):
        """Add a constraint not to assign specified samples to specified clusters, and reduce the number of instances.
        This method cannot be used in combination with other add_*_to_qubo() or set_*_by_qbits_reduction() methods.
        In addition, it cannot be called consecutively. Doing so may cause unexpected errors.

        Parameters
        ----------
        non_partition_level_instances : dict of {int : tuple of (int, ...)}
            Dictionary of samples not to assign to each cluster.
            Key is index of cluster, and value is list containing indices of samples which does not belong to those clusters.

        Raises
        ------
        ValueError
            If an index in non_partition_level_instances was les than 0.
        ValueError
            If a cluster index in non_partition_level_instances was greater than n_clusters.
        ValueError
            If a sample index in non_partition_level_instances was greater than n_samples.
        ValueError
            If this method was used in combination with add_*_to_qubo().
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if not hasattr(self, "label2index"):
            raise AttributeError("This instance is not fitted yet")
        self._partition_level_check(non_partition_level_instances,
                                    "non_partition_level_instances")

        self.set_constraint_flag = True
        self.__validate_add_set_exclusivity(inspect.currentframe().f_code.co_name)
        
        self.non_partition_level_instances = non_partition_level_instances
        non_partition_level_instances = {k: non_partition_level_instances[k] if k in non_partition_level_instances else tuple() for k in range(self.n_clusters)} # 辞書内にからのタプルを追加して全ての非分割レベルを付与
        self.qubo = {k: v for k, v in self.qubo.items() if k[0][0] not in non_partition_level_instances[k[0][1]] and k[1][0] not in non_partition_level_instances[k[1][1]]} # 非分割レベル集合内に存在しないqubitだけ残す
        self.deleted_qubits += tuple((i_omit, a_omit) for a_omit, omit_group in non_partition_level_instances.items() for i_omit in omit_group)
        self.deleted_qubits = tuple(set(self.deleted_qubits))

        if self.constraints["non_partition_level_instances"] is None:
            self.constraints["non_partition_level_instances"] = self.non_partition_level_instances
        else:
            self.constraints["non_partition_level_instances"] = self.__merge_instances(self.non_partition_level_instances, "non_partition_level_instances")

    @measure
    def decode_solution(self):
        self.solution = self.__revive_qubits(self.solution)
        tmp = {
                self.index2label[i]:q 
                for i,q in enumerate(self.solution)
        }
        self.labels = solution2labels(solution=tmp, n_clusters=self.n_clusters, n_points=self.n_points)
        if -1 in self.labels:
            raise RuntimeError("The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1.")

    def __revive_qubits(self, incompleted_solutions):
        completed_solutons = [incompleted_solutions[self.index2re_index[index]] if index in self.index2re_index.keys() else None for index in range(self.n_points * self.n_clusters)]
        if hasattr(self, "partition_level_instances"):
            for k_assign, k_assign_group in self.partition_level_instances.items():
                for i in k_assign_group:
                    for a in range(self.n_clusters):
                        completed_solutons[self.label2index[(i, a)]] = 1 if a == k_assign else 0
        if hasattr(self, "non_partition_level_instances"):
            for k_non_assign, k_non_assign_group in self.non_partition_level_instances.items():
                for i in k_non_assign_group:
                    completed_solutons[self.label2index[(i, k_non_assign)]] = 0
        if hasattr(self, "organized_must_link_instances"):
            for oml in self.organized_must_link_instances: # organized_must_link_instancesから制約グループを一つずつ抽出
                i_surv = oml[-1] # 制約グループのうちコピーされる要素のインデックス
                for i_omit in oml[:-1]: # コピーされる要素以外のイテレーション
                    for a in range(0, self.n_clusters): # クラスタ方向にイテレーション
                        completed_solutons[self.label2index[(i_omit, a)]] = completed_solutons[self.label2index[(i_surv, a)]]
        return completed_solutons

    def __validate_add_set_exclusivity(self, c_name):
        if self.add_constraint_flag and self.set_constraint_flag:
            if "add_" in c_name:
                raise ValueError(f"{c_name}() and set_*_by_qbits_reduction() cannot be used at the same time.")
            elif "set_" in c_name:
                raise ValueError(f"{c_name}() and add_*_to_qubo() cannot be used at the same time.")
    
    def __merge_instances(self, instances, instances_name):
        merged = {}
        for key in set(self.constraints[instances_name]) | set(instances):
            vals1 = self.constraints[instances_name].get(key, ())
            vals2 = instances.get(key, ())
            merged[key] = tuple(sorted(set(vals1) | set(vals2)))
        return merged

class KernelClustering(CombinatorialClustering):
    """Classify samples into the specified number of clusters by using Gaussian kernel.
    Depending on the class distribution of input data, it may not be possible to classify into the specified number of clusters.

    Parameters
    ----------
    n_clusters : int
        Number of clusters. Must be greater than or equal to 1.
    sigma : float, default=0.2
        Hyperparameter of Gaussian kernel.
    lam : float, default=None
        The Lagrange multiplier weight for the one-hot constraint used to uniquely identify each cluster.
        If None, '-2 * (Minimum value of the Gram matrix for a Gaussian kernel)' is used as lam.

    Raises
    ------
    ValueError
        If variable 'n_clusters' was less than 1.
    ValueError
        If variable 'sigma' was equal to or less than 0.
    TypeError
        If an invalid data type was specified for an argument.
    """
    def __init__(self, *, n_clusters, sigma=0.2, lam=None):
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be int")
        if (n_clusters < 1):
            raise ValueError("n_clusters must be greater than 0")
        self.n_clusters = n_clusters
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.integer, np.floating)):
            raise TypeError("sigma must be float")
        if (sigma <= 0):
            raise ValueError("sigma must be greater than 0")
        self.sigma = sigma
        if (lam is not None):
            if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)):
                raise TypeError("lam must be float")
        self.lam = lam
        self.timing = {}

    def set_gaussian_matrix(self):
        dists = distance.cdist(self.data, self.data, 'sqeuclidean')
        self.kernel_matrix = np.exp(-dists/(2*self.sigma**2))

    def set_gram_matrix(self):
        mean_row = np.mean(self.kernel_matrix, axis=1).reshape(self.n_points,1)
        mean_col = np.mean(self.kernel_matrix, axis=0).reshape(1,self.n_points)
        mean_all = np.mean(self.kernel_matrix)
        self.gram_matrix = self.kernel_matrix - mean_row - mean_col + mean_all

    @measure
    def set_qubo(self, mode="numpy"):
        self.set_gaussian_matrix()
        self.set_gram_matrix()
        gram_min = self.gram_matrix.min()
        if self.lam is None:
                self.lam = -2*gram_min

        if mode == "numpy":
            self.indexed_qubo = get_kronecker_qubo_kernel(self.gram_matrix, self.lam, self.n_points, self.n_clusters)
        else:
            self.qubo = {
                ((i,a),(j,b)) :
                -(self.gram_matrix[i,j]+self.gram_matrix[j,i]) if (i<j and a==b)
                else -(self.gram_matrix[i,j]+self.lam) if (i==j and a==b)
                else 2*self.lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            self.qubo = self._omit_zero_coefficients(self.qubo) # 非ゼロ要素だけのQUBO
            self.indexed_qubo = self.qubo_to_indexed_qubo()
        
    def get_qubo(self):
        return self.indexed_qubo
    
    def fit(self, X, if_dist=False):
        """Generate QUBO for clustering from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or array-like of shape(n_samples, n_samples)
            An array of samples, or a distance matrix between samples.
        if_dist : bool, default=False
            A bool indicating whether X is a distance matrix or not.
            If True, 'X' is assumed to be a distance matrix; otherwise, 'X' is treated as an array of samples.
            Kernel Clustering only supports False.

        Raises
        ------
        ValueError
            If variable 'X' had invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        TypeError
            If variable 'if_dist' was True.
        """

        if (if_dist == True):
            raise TypeError("Distance matrices are not supported as an input for KernelClustering.")

        array_check(X, "X", 2, if_dist=if_dist)
        X = np.array(X)
       
        self.data = X
        self.n_points = len(self.data)
        self._build_label_index_maps()
        self.set_qubo()
    

class BiDViT(BaseSolver):
    """Perform hierarchical clustering on input data.

    Parameters
    ----------
    kappa : int, default=32
        The maximum number of samples within a chunk utilized for efficient distance calculations.
        A larger kappa improves accuracy but increases computation time, whereas a smaller kappa reduces accuracy but enhances computation speed.
    epsilon : float, defalut=1.0
        The threshold of the Euclidean distance considered for samples to be deemed within the same cluster.
    epsilon_rate : float, defalut=1.05
        Epsilon multiplier
    rev_rate : float, default=1.5
        A weight for the one-hot constraint used to uniquely identify each cluster.

    Raises
    ------
    ValueError
        If variable 'kappa' was less than 1.
    ValueError
        If variable 'epsilon' were less than or equal to 0.
    ValueError
        If variable 'epsilon_rate' were less than or equal to 1.
    TypeError
        If an invalid data type was specified for an argument.
    """
    def __init__(self, kappa=32, epsilon=1.0, epsilon_rate=1.05, rev_rate=1.5):
        if isinstance(kappa, bool) or not isinstance(kappa, (int, np.integer)):
            raise TypeError("kappa must be int")
        if (kappa < 2):
            raise ValueError("kappa must be greater than 1")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating)):
            raise TypeError("epsilon must be float")
        if isinstance(epsilon_rate, bool) or not isinstance(epsilon_rate, (int, float, np.integer, np.floating)):
            raise TypeError("epsilon_rate must be float")
        if isinstance(rev_rate, bool) or not isinstance(rev_rate, (int, float, np.integer, np.floating)):
            raise TypeError("rev_rate must be float")
        if (epsilon <= 0):
            raise ValueError("epsilon must be greater than 0")
        if (epsilon_rate <= 1):
            raise ValueError("epsilon_rate must be greater than 1")
        self.timing = {}
        self.kappa = kappa
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate
        self.rev_rate = rev_rate

    def fit_predict(self, X, client, weight=None):
        """Generate QUBO for clustering from input data and perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of samples.
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.
        weight : list of float, default=None
            Weights applied per chunk. its length must be equal to n_sample.
            If None, a list of ones '[1] * len(X)' is used as weight.

        Returns
        -------
        tree : dict of {int : list of int}
            Clustering results. A dictionary where keys are the number of clusters and values are list of labels of the clusters to which each sample belongs.

        Raises
        ------
        ValueError
            If variable 'X' had invalid shape.
        ValueError
            If length of weight is not equal to n_sample.
        TypeError
            If an invalid data type was specified for an argument.
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        """
        array_check(X, "X", 2)
        if (weight is not None):
            if (not isinstance(weight, list)):
                raise TypeError("weight must be list")
            if (len(weight) != len(X)):
                raise ValueError("The length of weight must be equal to n_sample")
            for i in range(len(weight)):
                if isinstance(weight[i], bool) or not isinstance(weight[i], (int, float, np.integer, np.floating)):
                    raise TypeError("weight must be list of float")
                if (i > len(X)):
                    break

        self.client = client
        self.data = X
        self.weight = weight if weight is not None else [1] * len(self.data)
        
        try:
            result = self.__run_all_level(self.data, self.weight, self.epsilon)
            if result is None:
                raise ValueError(
                    "BiDViT did not converge: the data size did not decrease for many iterations. "
                    "Try increasing kappa or adjusting epsilon/epsilon_rate. "
                    f"(n_samples={len(self.data)}, kappa={self.kappa}, epsilon0={self.epsilon}, epsilon_rate={self.epsilon_rate})"
                )
        except ValueError:
            raise 
        except RuntimeError:
            raise 
        except Exception as e:
            raise RuntimeError(
                f"BiDViT failed due to an unexpected internal error: {type(e).__name__}: {e}"
        ) from e
            
        self.tree = self.__representer(result["tree_list"])
        return self.tree
    
    def __per_level(self, data, chunk, weight, epsilon):
        chunk_num = max(chunk) + 1
        indices_per_chunks = [[] for _ in range(chunk_num)]
        datas_per_chunks = [[] for _ in range(chunk_num)]
        weights_per_chunks = [[] for _ in range(chunk_num)]

        centroids = [[] for _ in range(chunk_num)]
        nearests = [[] for _ in range(chunk_num)]

        centroids_num_per_chunks = [[] for _ in range(chunk_num)]
        weights_of_centroids = [[] for _ in range(chunk_num)]

        for num in range(chunk_num):

            indices_per_chunks[num] = [i for i, x in enumerate(chunk) if x == num]
            datas_per_chunks[num] = [data[i] for i, x in enumerate(chunk) if x == num]
            weights_per_chunks[num] = [weight[i] for i, x in enumerate(chunk) if x == num]

            self.qubo = get_qubo(datas_per_chunks[num], weights_per_chunks[num], epsilon, self.rev_rate)

            self.n_points = len(datas_per_chunks[num])
            if self.n_points <= 1:
                raise ValueError(
                    "Chunk size became too small to build a valid QUBO (chunk_size <= 1). "
                    "Increase kappa or use kappa >= n_samples for small datasets. "
                    f"(chunk_id={num}, chunk_size={self.n_points}, kappa={self.kappa}, epsilon={epsilon})"
                )
                
            self.indexed_qubo = np.array(
                [
                    [
                        0 if j < i
                        else 0 if (i, j) not in self.qubo
                        else self.qubo[(i, j)]
                        for j in range(self.n_points)
                    ]
                    for i in range(self.n_points)
                ]
            )
            if self.indexed_qubo.size == 0 or self.indexed_qubo.shape[0] == 0:
                raise ValueError(
                    "QUBO became empty (0 variables). This is likely because kappa/chunk is too small "
                    "or preprocessing removed all variables. "
                    f"(chunk_id={num}, chunk_size={self.n_points}, kappa={self.kappa}, epsilon={epsilon})"
                )
            
            try:
                self.solve(self.client)
            except ValueError as e:
                if "expected shape is {0}" in str(e) or "different shape" in str(e):
                    raise ValueError(
                        "Failed to set/solve QUBO because the instance is empty (0 variables). "
                        "Increase kappa (preferably close to n_samples for small datasets) "
                        "or adjust epsilon. "
                        f"(chunk_id={num}, chunk_size={self.n_points}, kappa={self.kappa}, epsilon={epsilon})"
                    ) from e
                raise
            except Exception as e:
                # Amplifyなど内部失敗はRuntimeError
                raise RuntimeError(
                    f"Solver (Amplify) failed while solving QUBO: {type(e).__name__}: {e}. "
                    f"(chunk_id={num}, chunk_size={self.n_points}, kappa={self.kappa}, epsilon={epsilon})"
                ) from e

            centroids[num] = [datas_per_chunks[num][i] for i, x in enumerate(self.solution) if x == 1]
            if len(centroids[num]) == 0:
                raise ValueError(
                    "Solver returned no selected centroids for a chunk (all-zero solution). "
                    "This usually indicates an invalid/empty QUBO instance caused by too small kappa, "
                    "or unsuitable epsilon. "
                    f"(chunk_id={num}, chunk_size={self.n_points}, kappa={self.kappa}, epsilon={epsilon})"
                )
            dist_matrix = distance.cdist(centroids[num], datas_per_chunks[num], metric='euclidean')
            nearests[num] = np.argmin(dist_matrix, axis=0)

            centroids_num_per_chunks[num] = len(centroids[num])
            
            prev_nearest = nearests[num]
            prev_weight_of_centroids = [0] * centroids_num_per_chunks[num]
            prev_weights_per_chunk = weights_per_chunks[num]
            for sample_idx, cent_num in enumerate(prev_nearest):
                prev_weight_of_centroids[cent_num] += prev_weights_per_chunk[sample_idx]
            weights_of_centroids[num] = prev_weight_of_centroids

        next_data_size = sum(centroids_num_per_chunks)
        next_data = [[] for _ in range(next_data_size)]
        next_weight = [[] for _ in range(next_data_size)]
        cluster = [[] for _ in range(len(data))]

        temp_sum = 0
        for num in range(chunk_num):
            prev_centroids_num = centroids_num_per_chunks[num]
            prev_centroid = centroids[num]
            prev_weights_of_centroid = weights_of_centroids[num]
            for cent_num in range(prev_centroids_num):
                next_data[cent_num+temp_sum] = prev_centroid[cent_num]
                next_weight[cent_num+temp_sum] = prev_weights_of_centroid[cent_num]

            prev_nearest = nearests[num] + temp_sum
            temp_sum += centroids_num_per_chunks[num]

            prev_chunk = indices_per_chunks[num]
            for cnt in range(len(prev_chunk)):
                cluster[prev_chunk[cnt]] = prev_nearest[cnt]

        return {
            # "indices_per_chunks": indices_per_chunks,
            # "datas_per_chunks": datas_per_chunks,
            # "weights_per_chunks": weights_per_chunks,
            # "centroids": centroids,
            # "nearests": nearests,
            # "centroids_num_per_chunks": centroids_num_per_chunks,
            # "weights_of_centroids": weights_of_centroids,
            "next_data": next_data,
            "next_weight": next_weight,
            "cluster": cluster
        }
    
    def __run_all_level(self, data, weight, epsilon):

        tree_list = []
        # result_list = []
        count = 0
        
        while (len(data) > 2):
            prevlen = len(data)
            chunk = get_chunk(data, self.kappa)
            result = self.__per_level(data, chunk, weight, epsilon)

            data = result["next_data"]
            weight = result["next_weight"]
            cluster = result["cluster"]
            epsilon *= self.epsilon_rate
            
            tree_list.append(cluster)
            # result_list.append(result)
            
            if (prevlen == len(data)):
                count += 1
            else:
                count = 0
            if count > 200:
                raise ValueError(
                    "BiDViT did not make progress (data size did not decrease for 200 levels). "
                    "This can happen when kappa is too small or epsilon settings prevent collapsing. "
                    f"(current_n={len(data)}, kappa={self.kappa}, epsilon={epsilon}, epsilon_rate={self.epsilon_rate})"
                )

        return {
            # "result_list": result_list,
            "tree_list": tree_list
        }
    
    def __representer(self, tree_list):
        represent_list = []
        for k in range(1, len(tree_list)+1):
            represent = copy.deepcopy(tree_list)
            for i in range(len(tree_list)-k, 0, -1):
                for j in range(len(tree_list[i-1])):
                    represent[i-1][j] = represent[i][represent[i-1][j]]
            represent_list.append(represent[0])
        for i in range(len(represent_list) // 2): # reverse the list
            represent_list[i],represent_list[-1-i] = represent_list[-1-i],represent_list[i]
        tree = {
            len(collections.Counter(x)): x
            for x in represent_list
        }
        return tree
