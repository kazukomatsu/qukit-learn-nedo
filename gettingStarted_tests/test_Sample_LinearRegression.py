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
from sklearn.linear_model import LinearRegression as SLR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient
from datetime import timedelta
from qklearn.utils import read_token
import pytest

def get_artificial_data(n_points):
    x, y = make_regression(n_samples=n_points, n_features=1, n_informative=1,
                           bias=0.0, noise=20.0)
    return x, y


@pytest.fixture(scope="function")
def setup_client():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)
    yield client

def get_sl_result(x_train, y_train, x_test, y_test):
    slr = SLR()
    slr.fit(x_train, y_train)
    y_train_pred_slr = slr.predict(x_train)
    y_test_pred_slr = slr.predict(x_test)
    MSE_sc = mean_squared_error(y_test, y_test_pred_slr)

    return MSE_sc, y_train_pred_slr, y_test_pred_slr

def test_sample_LinearRegression(setup_client):
    n_points = 100
    x, y = get_artificial_data(n_points)
    x_train, x_test = x[:80], x[-20:]
    y_train, y_test = y[:80], y[-20:]

    MSE_sc, y_train_pred_slr, y_test_pred_slr = get_sl_result(x_train, y_train,
                                                              x_test, y_test)

    MSE_min = 10000
    element_min = 0
    exp_min = 0
    exp = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for element in range(3, 10):
        MSE = np.zeros(10)
        for i, e in enumerate(exp):
            model = QLR()
            model.fit(x_train, y_train, setup_client,
                      num_elements=element, exponent_offset=e)
            y_pred = model.predict(x_test)
            MSE[i] = mean_squared_error(y_test, y_pred)
            if MSE_min >= MSE[i]:
                MSE_min = MSE[i]
                element_min = element
                exp_min = e
            assert(len(y_pred) == len(x_test))
            assert(type(y_pred) is np.ndarray)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("the number of elements "+ str(element))
        ax.set_xlabel("the exponent offset")
        ax.set_ylabel("MSE")
        ax.set_xticks(exp)
        ax.scatter(exp, MSE, c ='blue')
        ax.plot(exp, np.full(10, MSE_sc), c='green')
        plt.savefig("test_sample_LinearRegression_sweep_num_elements_exponent_offset" + str(element) + ".png")
        plt.close(fig)
    print("k_min ", element_min, ", e_min ", exp_min, ", MSE_min ", MSE_min)

    qlr = QLR()
    qlr.fit(x_train, y_train, setup_client,
            num_elements=element_min, exponent_offset=exp_min)
    y_train_pred_qlr = qlr.predict(x_train)
    y_test_pred_qlr = qlr.predict(x_test)
    assert(len(y_train_pred_qlr) == len(x_train))
    assert(len(y_test_pred_qlr) == len(x_test))
    assert(type(y_train_pred_qlr) is np.ndarray)
    assert(type(y_test_pred_qlr) is np.ndarray)

    MSE = mean_squared_error(y_test, y_test_pred_qlr)
    print(MSE)

    fig = plt.figure(figsize=(24, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.scatter(x_train, y_train, label="Training")
    ax1.scatter(x_test, y_test, label="Test")
    ax1.scatter(x_test, y_test_pred_slr, label="SLR")
    ax1.scatter(x_test, y_test_pred_qlr, label="QLR")
    ax1.legend()

    ax2.scatter(x_train, y_train, label="Training")
    ax2.scatter(x_test, y_test, label="Test")
    ax2.scatter(x_train, y_train_pred_qlr, label="QLR (Training prediction)")
    ax2.scatter(x_test, y_test_pred_qlr, label="QLR (Test prediction)")
    ax2.legend()

    ax3.scatter(x_train, y_train, label="Training")
    ax3.scatter(x_test, y_test, label="Test")
    ax3.scatter(x_train, y_train_pred_slr, label="SLR (Training prediction)")
    ax3.scatter(x_test, y_test_pred_slr, label="SLR (Test prediction)")
    ax3.legend()

    fig.tight_layout()
    plt.savefig("test_sample_LinearRegression_result.png")
    plt.close(fig)
