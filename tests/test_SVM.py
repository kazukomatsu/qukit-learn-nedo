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
from qklearn.svm import SVC as QSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def test_SVM():
    inputs,targets=make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.7)
    targets[targets == 0] = -1
    X_train, X_test, t_train, t_test = train_test_split(inputs,targets)

    qsvc = QSVC()
    k_max = 3
    e_max = 11

    qsvc.training_dataset = X_train
    qsvc.num_points = len(X_train)
    qsvc.num_dimensions = len(X_train[0])
    qsvc.training_label = t_train

    qsvc.set_qubo(num_elements=k_max, basis=2, exponent_offset=e_max, xi=1)