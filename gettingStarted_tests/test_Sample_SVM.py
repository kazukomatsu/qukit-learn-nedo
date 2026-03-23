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
import pytest
from qklearn.svm import SVC as QSVC
from sklearn.svm import SVC as SSVC
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from amplify import FixstarsClient
from datetime import timedelta
from qklearn.utils import read_token

@pytest.fixture(scope="function")
def setup_client():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)
    yield client


def test_sample_SVM(setup_client):
    n_points = 100
    n_clusters = 2

    inputs,targets=make_blobs(n_samples=n_points, centers=n_clusters,
                              random_state=0, cluster_std=0.7)
    plt.scatter(inputs[:,0],inputs[:,1],c=targets, cmap='winter')
    plt.title("Original Data points of the 2 classes")
    plt.savefig("test_sample_SVM_original.png")
    plt.close()

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    x_train, x_test, t_train, t_test = train_test_split(inputs,targets)
    ssvc = SSVC()
    ssvc.fit(x_train, t_train)
    y_train_pred_slr = ssvc.predict(x_train)
    y_test_pred_slr = ssvc.predict(x_test)
    acc_sc = accuracy_score(t_test, y_test_pred_slr)
    print(acc_sc)
    ax1.scatter(x_train[:,0],x_train[:,1],c=y_train_pred_slr, cmap='winter')
    ax1.set_title("Sklearn Train Data points of the 2 classes")
    ax2.scatter(x_test[:,0],x_test[:,1],c=y_test_pred_slr, cmap='winter')
    ax2.set_title("Sklearn Test Data points of the 2 classes")

    fig.tight_layout()
    plt.savefig("test_sample_SVM_sklearn.png")
    plt.close(fig)

    acc_max = 0
    element_max = 0
    exp_max = 0
    list_exponent_offset = list(range(5, 15))
    for element in range(3, 10):
        acc = np.zeros(10)
        for i, exp in enumerate(list_exponent_offset):
            print(f"element : {element}, exp : {exp}")
            model = QSVC()
            model.fit(x_train, t_train, setup_client,
                      num_elements=element, exponent_offset=exp, xi=1)
            y_pred = model.predict(x_test)
            assert(len(y_pred) == len(x_test))
            assert(type(y_pred) is np.ndarray)
            acc[i] = accuracy_score(t_test, y_pred)
            if acc_max <= acc[i]:
                acc_max = acc[i]
                element_max = element
                exp_max = exp

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("the number of elements "+ str(element))
        ax.set_xlabel("the exponent offset")
        ax.set_ylabel("accuracy")
        ax.set_xticks(list_exponent_offset)
        ax.scatter(list_exponent_offset, acc, c ='blue')
        ax.plot(list_exponent_offset, np.full(10, acc_sc), c='green')
        plt.savefig("test_sample_SVM_qklearn_sweep_exponent_offset_num_elements_"+str(element)+".png")
        plt.close(fig)

    print("k_max ", element_max, ", e_max ", exp_max, ", acc_max ", acc_max)

    qsvc = QSVC()
    qsvc.fit(x_train, t_train, setup_client,
             num_elements=element_max, exponent_offset=exp_max, xi=1)
    y_train_pred_qlr = qsvc.predict(x_train)
    y_test_pred_qlr = qsvc.predict(x_test)
    assert(len(y_train_pred_qlr) == len(x_train))
    assert(len(y_test_pred_qlr) == len(x_test))

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title("Qklearn Train Data points of the 2 classes")
    ax1.scatter(x_train[:,0],x_train[:,1],c=y_train_pred_qlr, cmap='winter')
    ax1.set_title("Qklearn Test Data points of the 2 classes")
    ax1.scatter(x_test[:,0],x_test[:,1],c=y_test_pred_qlr, cmap='winter')
    plt.savefig("test_sample_SVM_qklearn.png")
    plt.close(fig)
