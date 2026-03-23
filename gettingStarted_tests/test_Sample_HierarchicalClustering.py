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
from qklearn.cluster import BiDViT
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient
from datetime import timedelta
from qklearn.utils import read_token
import pytest


def get_artificial_data(n_clusters, n_points):
    data, labels = make_blobs(random_state=8, n_samples=n_points, n_features=2,
                              cluster_std=1.5, centers=n_clusters)
    return data, labels


@pytest.fixture(scope="function")
def setup_client():
    client = FixstarsClient()
    client.parameters.timeout = timedelta(milliseconds=1000)
    client.token = read_token("Fixstars")
    yield client


def test_sample_HierarchicalClustering(setup_client):
    cluster_num = 9
    point_num = 200

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(cluster_num, point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    qcl = BiDViT()
    tree = qcl.fit_predict(data, setup_client)
    print(f"keys : {tree.keys()}")
    index = list(tree.keys())[len(tree)-3]
    print(f"index : {index}")
    print(len(tree))
    ax2.set_title("labels_BiDViT")
    ax2.scatter(data[:,0], data[:,1], c=tree[index])

    fig.tight_layout()
    plt.savefig("test_sample_HierarchicalClustering.png")
    plt.close(fig)

    assert(type(tree) is dict)
    for k, v in tree.items():
        assert(len(v) == point_num)
        assert(len(set(v)) == k)
