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
from qklearn.cluster import ConsensusClustering

def test_ConsensusClustering():
    labels_one = [0, 2, 0, 1, 1, 2, 1, 0, 2]
    labels_two = [2, 0, 1, 0, 2, 1, 0, 2, 1]
    qcl = ConsensusClustering(n_clusters=3, model="pairwise_similarity-based")
    X=[labels_one, labels_two]

    qcl.clusterings = X
    qcl.n_points = len(qcl.clusterings[0])
    qcl.n_clusterings = len(qcl.clusterings)
    qcl.lam = qcl.n_points - qcl.n_clusters
    qcl.label2index = {
        (i,a):i*qcl.n_clusters+a 
        for i in range(qcl.n_points) 
        for a in range(qcl.n_clusters)
    }
    qcl.index2label = {
        i*qcl.n_clusters+a:(i,a) 
        for i in range(qcl.n_points) 
        for a in range(qcl.n_clusters)
    }
    qcl.similarity_matrix = {
        (i,j) : sum([
            1 for labels_m in qcl.clusterings if labels_m[i]==labels_m[j]
        ]) / qcl.n_clusterings
        for i in range(0,qcl.n_points)
        for j in range(i,qcl.n_points)
    }

    qcl.set_qubo()
    qcl.indexed_qubo = qcl.qubo_to_indexed_qubo()