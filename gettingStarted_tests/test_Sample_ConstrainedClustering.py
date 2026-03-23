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
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient
from datetime import timedelta
from qklearn.utils import read_token
import pytest, math

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


def test_sample_ConstrainedClustering_add_must_link_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9
    link = [(0,1), (2, 3), (4, 5)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)

    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_must_link_to_qubo(link)
    try:
        labels_ccl = ccl.predict(setup_client)
    except RuntimeError as e:
        err_msg = "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1."
        assert err_msg in str(e)
    else:
        print(f"must_link_to_qubo : {link}")
        print("labels_ccl:")
        print(labels_ccl)
        ax2.set_title("labels_qcl")
        ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

        fig.tight_layout()
        plt.savefig("test_sample_ConstrainedClustering_add_must_link_to_qubo.png")
        plt.close(fig)

        assert(len(labels_ccl) == point_num)
        for i in link:
            assert labels_ccl[i[0]] == labels_ccl[i[1]]
        assert(type(labels_ccl) is list)


def test_sample_ConstrainedClustering_set_must_link_by_qbits_reduction(setup_client):
    cluster_num = 3
    point_num = 9
    link = [(0,1), (2, 3), (4, 5)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)

    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.set_must_link_by_qbits_reduction(link)
    labels_ccl = ccl.predict(setup_client)
    print(f"must_link_to_qubo : {link}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_set_must_link_by_qbits_reduction.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for i in link:
        assert labels_ccl[i[0]] == labels_ccl[i[1]]
    assert(len(set(labels_ccl)) == cluster_num)
    assert(type(labels_ccl) is list)


def test_sample_ConstrainedClustering_add_cannot_link_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9
    nlink = [(0,1), (2,3), (4,5)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)

    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_cannot_link_to_qubo(nlink)
    labels_ccl = ccl.predict(setup_client)
    print(f"cannot_link_to_qubo : {nlink}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_add_cannot_link_to_qubo.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for i in nlink:
        assert labels_ccl[i[0]] != labels_ccl[i[1]]
    assert(type(labels_ccl) is list)
    assert(len(set(labels_ccl)) == cluster_num)


def test_sample_ConstrainedClustering_add_partition_level_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9
    plevel = {0:(0,2,4),1:(1,3,6)}

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_partition_level_to_qubo(plevel)
    labels_ccl = ccl.predict(setup_client)
    print(f"partition level : {plevel}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_add_partition_level_to_qubo.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for k in plevel.keys():
        for v in plevel[k]:
            assert(labels_ccl[v] == k)
    assert(type(labels_ccl) is list)
    assert(len(set(labels_ccl)) == cluster_num)

def test_sample_ConstrainedClustering_set_partition_level_by_qbits_reduction(setup_client):
    cluster_num = 3
    point_num = 9
    plevel = {0:(0,2,4),1:(1,3,6)}

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.set_partition_level_by_qbits_reduction(plevel)
    labels_ccl = ccl.predict(setup_client)
    print(f"partition level : {plevel}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_set_partition_level_by_qbits_reduction.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for k in plevel.keys():
        for v in plevel[k]:
            assert(labels_ccl[v] == k)
    assert(type(labels_ccl) is list)
    assert(len(set(labels_ccl)) == cluster_num)


def test_sample_ConstrainedClustering_add_non_partition_level_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9
    plevel = {0:(0,2,4),1:(1,3,6)}

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_non_partition_level_to_qubo(plevel)
    labels_ccl = ccl.predict(setup_client)
    print(f"non partition level : {plevel}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_add_non_partition_level_to_qubo.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for k in plevel.keys():
        for v in plevel[k]:
            assert(labels_ccl[v] != k)
    assert(type(labels_ccl) is list)
    assert(len(set(labels_ccl)) == cluster_num)

def test_sample_ConstrainedClustering_set_non_partition_level_by_qbits_reduction(setup_client):
    cluster_num = 3
    point_num = 9
    plevel = {0:(0,2,4),1:(1,3,6)}

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.set_non_partition_level_by_qbits_reduction(plevel)
    labels_ccl = ccl.predict(setup_client)
    print(f"non partition level : {plevel}")
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_set_non_partition_level_by_qbits_reduction.png")
    plt.close(fig)

    assert(len(labels_ccl) == point_num)
    for k in plevel.keys():
        for v in plevel[k]:
            assert(labels_ccl[v] != k)
    assert(len(set(labels_ccl)) == cluster_num)
    assert(type(labels_ccl) is list)

def test_sample_ConstrainedClustering_add_balanced_sizes_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_balanced_sizes_to_qubo()
    labels_ccl = ccl.predict(setup_client)
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_add_balanced_sizes_to_qubo.png")
    plt.close(fig)

    assert(len(set(labels_ccl)) == cluster_num)
    assert(len(labels_ccl) == point_num)
    assert(type(labels_ccl) is list)
    for i in set(labels_ccl):
        c = 0
        for j in labels_ccl:
            if (i == j):
                c += 1
        assert(c == math.floor(float(point_num / cluster_num))
               or c == math.ceil(float(point_num / cluster_num)))


def test_sample_ConstrainedClustering_add_limited_sizes_to_qubo(setup_client):
    cluster_num = 3
    point_num = 9
    limitation = {0:2, 1:2, 2:5}

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    data, labels_origin = get_artificial_data(n_clusters=cluster_num,
                                              n_points=point_num)
    print("labels_origin:")
    print(labels_origin)
    ax1.set_title("labels_origin")
    ax1.scatter(data[:,0], data[:,1], c=labels_origin)

    ccl = ConstrainedClustering(n_clusters=cluster_num)
    ccl.fit(data)
    ccl.add_limited_sizes_to_qubo(limitation)
    labels_ccl = ccl.predict(setup_client)
    print("labels_ccl:")
    print(labels_ccl)
    ax2.set_title("labels_qcl")
    ax2.scatter(data[:,0], data[:,1], c=labels_ccl)

    fig.tight_layout()
    plt.savefig("test_sample_ConstrainedClustering_add_limited_sizes_to_qubo.png")
    plt.close(fig)

    assert(len(set(labels_ccl)) == cluster_num)
    assert(len(labels_ccl) == point_num)
    assert(type(labels_ccl) is list)
    for i in set(labels_ccl):
        c = 0
        for j in labels_ccl:
            if (i == j):
                c += 1
        assert(limitation[i] <= c)
