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
from qklearn.linear_model import LinearRegression as QLR
from sklearn.datasets import make_regression
import numpy as np

def test_LinearRegression():
    X, y = make_regression(
        n_samples=100,
        n_features=1,
        n_informative=1,
        bias=0.0,
        noise=20.0)
    X_train, X_test = X[:80], X[-20:]
    y_train, y_test = y[:80], y[-20:]

    qlr = QLR()
    k_min = 9
    e_min = 0

    qlr.num_points = len(X)
    qlr.num_dimensions = len(X[0])
    qlr.regression_label = y
    qlr.training_dataset = np.column_stack((X, np.ones(qlr.num_points)))

    qlr.set_qubo(num_elements=k_min, basis=2, exponent_offset=e_min)