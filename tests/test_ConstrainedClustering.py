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
from qklearn.cluster import ConstrainedClustering
from sklearn.datasets import make_blobs

def test_add_must_link_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_must_link_to_qubo([(0,1), (2,3), (4,5)])

def test_set_must_link_by_qbits_reduction():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.set_must_link_by_qbits_reduction([(0,1), (2,3), (4,5)])

def test_add_cannot_link_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_cannot_link_to_qubo([(0,1), (2,3), (4,5)])

def test_add_partition_level_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_partition_level_to_qubo({0:(0,2,4),1:(1,3,6)})

def test_set_partition_level_by_qbits_reduction():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.set_partition_level_by_qbits_reduction({0:(0,2,4),1:(1,3,6)})

def test_add_non_partition_level_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_non_partition_level_to_qubo({0:(0,2,4),1:(1,3,6)})

def test_set_non_partition_level_by_qbits_reduction():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.set_non_partition_level_by_qbits_reduction({0:(0,2,4),1:(1,3,6)})

def test_add_balanced_sizes_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_balanced_sizes_to_qubo()

def test_add_limited_sizes_to_qubo():
    n_clusters = 3
    n_points = 9
    data, _ = make_blobs(random_state=8,
                                        n_samples=n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=n_clusters)
    qcl = ConstrainedClustering(n_clusters=n_clusters)
    qcl.fit(data)
    qcl.add_limited_sizes_to_qubo({0:2, 1:2, 2:5})