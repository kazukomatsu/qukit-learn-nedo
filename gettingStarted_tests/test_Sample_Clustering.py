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
from qklearn.cluster import CombinatorialClustering, BinaryClustering, KernelClustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient
from datetime import timedelta
from qklearn.utils import read_token
from scipy.spatial.distance import cdist
from sklearn.datasets import make_circles
import pytest

def get_artificial_data(n_clusters, n_points):
    data, labels = make_blobs(random_state=8, n_samples=n_points, n_features=2,
                              cluster_std=1.5, centers=n_clusters)
    return data, labels

@pytest.fixture(scope="function")
def setup_client():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)
    yield client

def test_sample_Combinatorial_Clustering(setup_client):
    cluster_num = 3
    point_num = 9

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    qcl = CombinatorialClustering(n_clusters=cluster_num)
    labels_qcl = qcl.fit_predict(data, setup_client)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_qcl)

    fig.tight_layout()
    plt.savefig("test_sample_Combinatorial_Clustering.png")
    plt.close(fig)

    assert(len(labels_qcl) == point_num)
    assert(len(set(labels_qcl)) == cluster_num)
    assert(type(labels_qcl) is list)

def test_sample_Combinatorial_Clustering_distance_matrix(setup_client):
    cluster_num = 3
    point_num = 9

    data, _ = get_artificial_data(n_clusters=cluster_num, n_points=point_num)
    dist = cdist(data, data, metric="euclidean")

    qcl = CombinatorialClustering(n_clusters=cluster_num)
    qcl.fit(dist, if_dist=True)
    labels_qcl = qcl.predict(setup_client)

    plt.title("labels_qcl")
    plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    plt.savefig("test_sample_Combinatorial_Clustering_distance_matrix.png")
    plt.close()

    assert(len(labels_qcl) == point_num)
    assert(len(set(labels_qcl)) == cluster_num)
    assert(type(labels_qcl) is list)

def test_sample_Binary_Clustering(setup_client):
    cluster_num = 2
    point_num = 10

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    qcl = BinaryClustering()
    labels_qcl = qcl.fit_predict(data, setup_client)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_qcl)

    fig.tight_layout()
    plt.savefig("test_sample_Binary_Clustering.png")
    plt.close(fig)

    assert(len(labels_qcl) == point_num)
    assert(len(set(labels_qcl)) == cluster_num)
    assert(type(labels_qcl) is list)

def test_sample_Kernel_Clustering(setup_client):
    cluster_num = 2
    point_num = 64

    data, _ = make_circles(n_samples=point_num, factor=0.3,
                           noise=0.05, random_state=0)

    qcl = KernelClustering(n_clusters=cluster_num, sigma=0.2)
    labels_qcl = qcl.fit_predict(data, setup_client)

    plt.title("labels_qcl")
    plt.scatter(data[:,0], data[:,1], c=labels_qcl)
    plt.savefig("test_sample_Kernel_Clustering.png")
    plt.close()

    assert(len(labels_qcl) == point_num)
    assert(len(set(labels_qcl)) == cluster_num)
    assert(type(labels_qcl) is list)
