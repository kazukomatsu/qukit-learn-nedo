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
import numpy as np

def get_kronecker_qubo_euclidean(dist, lagr, n_points, n_clusters):
    # Generate the Objective QUBO 
    dist = np.triu(dist)
    iden = np.identity(n_clusters)
    obje = np.kron(dist, iden)

    # Generate the Constraint QUBO
    lamb = np.diag(np.full(n_points, lagr))
    uden = np.tri(n_clusters, k=-1).T
    coe2 = 2 * uden - iden
    cons = np.kron(lamb, coe2)

    qubo = obje + cons
    return qubo

def get_kronecker_qubo_kernel(gram, lagr, n_points, n_clusters):
    # Generate the Objective QUBO 
    iden = np.identity(n_clusters)
    diag = np.diag(np.diag(gram))
    triu = np.triu(gram, k=1)
    coe1 = - diag - 2 * triu
    obje = np.kron(coe1, iden)

    # Generate the Constraint QUBO
    lamb = np.diag(np.full(n_points, lagr))
    uden = np.tri(n_clusters, k=-1).T
    coe2 = 2 * uden - iden
    cons = np.kron(lamb, coe2)

    qubo = obje + cons
    return qubo