"""
MIT License

Copyright c 2025 Tohoku University

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
from datetime import timedelta
from qklearn.cluster import KernelClustering
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_circles
import numpy as np
from scipy.spatial.distance import cdist

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.fixture(scope="module")
def create_err_msg():
    err_msg = {}

    err_msg["55"] = "The cluster labels of the samples that do not satisfy the constraint are set to -1. Please execute the program multiple times or adjust the parameters."
    err_msg["56"] = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["57"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["58"] = err_msg["55"]
    err_msg["59"] = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["60"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["62"] = "Distance matrices are not supported as an input for KernelClustering."

    return err_msg

@pytest.mark.parametrize(
    ["n_clusters", "e", "err_msg"],
    [
        pytest.param(1,   "N", None, id="No.1"),
        pytest.param(2,   "N", None, id="No.2"),
        pytest.param(0,   "V", "n_clusters must be greater than 0", id="No.3"),
        pytest.param(2.5, "T", "n_clusters must be int", id="No.4"),
    ]
)
def test_constructor_n_clusters(mocker, n_clusters, e, err_msg):
    if (e == "N"):
        kcl = KernelClustering(n_clusters=n_clusters)
        data, _ = make_circles(n_samples=64, factor=0.3,
                               noise=0.05, random_state=0)
        client = setup_fixstars()
        kcl.fit(data)
        if n_clusters == 1:
            mocker.patch("qklearn.cluster.solution2labels", return_value=[0 for _ in range(64)])
        labels = kcl.predict(client)

        assert(isinstance(labels, list))
        assert(len(labels) == 64)
        assert(len(set(labels)) == n_clusters)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            kcl = KernelClustering(n_clusters=n_clusters)
        assert(err_msg in str(e.value))
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            kcl = KernelClustering(n_clusters=n_clusters)
        assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["sigma", "e", "err_msg"],
    [
        pytest.param(0.1, "N", None, id="No.5"),
        pytest.param(1,   "N", None, id="No.6"),
        pytest.param(0,   "V", "sigma must be greater than 0", id="No.7"),
        pytest.param("1", "T", "sigma must be float", id="No.8"),
    ]
)
def test_constructor_sigma(sigma, e, err_msg):
    n_clusters = 2
    if (e == "N"):
        kcl = KernelClustering(n_clusters=n_clusters, sigma=sigma)
        data, _ = make_circles(n_samples=64, factor=0.3,
                               noise=0.05, random_state=0)
        client = setup_fixstars()
        kcl.fit(data)
        labels = kcl.predict(client)

        assert(isinstance(labels, list))
        assert(len(labels) == 64)
        assert(len(set(labels)) == n_clusters)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            kcl = KernelClustering(n_clusters=n_clusters, sigma=sigma)
        assert(err_msg in str(e.value))
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            kcl = KernelClustering(n_clusters=n_clusters, sigma=sigma)
        assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["lam", "e", "err_msg"],
    [
        pytest.param(64, "N", None, id="No.9"),
        pytest.param(70.4,   "N", None, id="No.10"),
        pytest.param("1", "T", "lam must be float", id="No.11"),
    ]
)
def test_constructor_lam(lam, e, err_msg):
    n_clusters = 2
    if (e == "N"):
        kcl = KernelClustering(n_clusters=n_clusters, lam=lam)
        data, _ = make_circles(n_samples=64, factor=0.3,
                               noise=0.05, random_state=0)
        client = setup_fixstars()
        kcl.fit(data)
        labels = kcl.predict(client)

        assert(isinstance(labels, list))
        assert(len(labels) == 64)
        assert(len(set(labels)) == n_clusters)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            kcl = KernelClustering(n_clusters=n_clusters, lam=lam)
        assert(err_msg in str(e.value))

@pytest.fixture(scope="module")
def create_fit_data():
    test_data = {}
    dataf2, _ = make_circles(n_samples=64, factor=0.3,
                             noise=0.05, random_state=0)
    distf2 = cdist(dataf2, dataf2, metric="euclidean")
    invalid_listf2 = [[i ,i] for i in range(10)]
    invalid_listf2[9][1] = "h"
    test_data["12"] = dataf2
    test_data["13"] = dataf2.tolist()
    test_data["14"] = distf2
    test_data["15"] = distf2.tolist()
    test_data["16"] = {1: (1, 1)}
    test_data["17"] = np.array(invalid_listf2)
    test_data["18"] = invalid_listf2
    test_data["19"] = [[1,2,3], [1,2]]
    test_data["20"] = [[1,2,3], [1,2]]
    test_data["21"] = np.arange(8).reshape((2, 2, 2))
    test_data["22"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["23"] = np.arange(8).reshape((8,))
    test_data["24"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data["25"] = np.arange(12).reshape((4,3))
    test_data["26"] = [[1, 2, 3],  [4, 5, 6]]
    test_data["33"] = dataf2
    test_data["34"] = dataf2.tolist()
    test_data["35"] = distf2
    test_data["36"] = distf2.tolist()
    test_data["37"] = dataf2
    test_data["38"] = dataf2.tolist()
    test_data["39"] = distf2
    test_data["40"] = distf2.tolist()
    test_data["41"] = {1: (1, 1)}
    test_data["42"] = np.array(invalid_listf2)
    test_data["43"] = invalid_listf2
    test_data["44"] = [[1,2,3], [1,2]]
    test_data["45"] = [[1,2,3], [1,2]]
    test_data["46"] = np.arange(8).reshape((2, 2, 2))
    test_data["47"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["48"] = np.arange(8).reshape((8,))
    test_data["49"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data["50"] = np.arange(12).reshape((4,3))
    test_data["51"] = [[1, 2, 3],  [4, 5, 6]]

    return test_data



@pytest.mark.parametrize(
    ["No", "dist", "e", "err_msg"],
    [
        pytest.param("12", False, "N", None, id="No.12"),
        pytest.param("13", False, "N", None, id="No.13"),
        pytest.param("16", False, "T", "must be array-like object", id="No.16"),
        pytest.param("17", False, "T", "has invalid data type", id="No.17"),
        pytest.param("18", False, "T", "has invalid data type", id="No.18"),
        pytest.param("19", False, "V", "has invalid shape", id="No.19"),
        pytest.param("21", False, "V", "must be 2-d array", id="No.21"),
        pytest.param("22", False, "V", "must be 2-d array", id="No.22"),
        pytest.param("23", False, "V", "must be 2-d array", id="No.23"),
        pytest.param("24", False, "V", "must be 2-d array", id="No.24"),
        pytest.param("61", True,  "T", "Distance matrices are not supported as an input for KernelClustering.", id="No.61")
    ]
)
def test_fit(create_fit_data, dist, No, e, err_msg):
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    indata = create_fit_data.get(No)
    if (e == "N"):
        kcl.fit(indata, if_dist=dist)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            kcl.fit(indata, if_dist=dist)
            assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            kcl.fit(indata, if_dist=dist)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error Type")

@pytest.mark.parametrize(
    ["Client"],
    [
        pytest.param("N", id="No.27"),
        pytest.param("F", id="No.28"),
    ]
)
def test_predict_normal(Client):
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    if (Client == "F"):
        client = setup_fixstars()
    elif (Client == "N"):
        client = None
    else:
        raise Exception("Invalid client")

    kcl.fit(data)
    labels = kcl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == 64)
    assert(len(set(labels)) == n_clusters)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.29"),
    ],
)
@pytest.mark.neal
def test_predict_error_neal(lack_package):
    n_clusters = 2
    client = None
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    kcl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = kcl.predict(client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.30"),
    ],
)
@pytest.mark.dimod
def test_predict_error_dimod(lack_package):
    n_clusters = 2
    client = None
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    kcl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = kcl.predict(client)

@pytest.mark.parametrize(
    ["Case", "err_msg"],
    [
        pytest.param("InvalidToken", "exceptions were raised from solve() of the Fixstars Amplify SDK", id="No.31"),
        pytest.param("NotFitted", "This instance is not fitted yet", id="No.32"),
    ]
)
def test_predict_error(Case, err_msg):
    client = setup_fixstars()
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    if (Case == "InvalidToken"):
        kcl.fit(data)
        client.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            label = kcl.predict(client)
    elif (Case == "NotFitted"):
        with pytest.raises(AttributeError) as e:
            label = kcl.predict(client)
    else:
        raise Exception("Invalid Case")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "client", "e", "if_dist", "err_msg"],
    [
        pytest.param("33", "N", "N", False, None, id="No.33"),
        pytest.param("34", "N", "N", False, None, id="No.34"),
        pytest.param("37", "F", "N", False, None, id="No.37"),
        pytest.param("38", "F", "N", False, None, id="No.38"),
        pytest.param("41", "F", "T", False, "must be array-like object", id="No.41"),
        pytest.param("42", "F", "T", False, "has invalid data type", id="No.42"),
        pytest.param("43", "F", "T", False, "has invalid data type", id="No.43"),
        pytest.param("44", "F", "V", False, "has invalid shape", id="No.44"),
        pytest.param("46", "F", "V", False, "must be 2-d array", id="No.46"),
        pytest.param("47", "F", "V", False, "must be 2-d array", id="No.47"),
        pytest.param("48", "F", "V", False, "must be 2-d array", id="No.48"),
        pytest.param("49", "F", "V", False, "must be 2-d array", id="No.49"),
        pytest.param("62", "F", "T", True, "Distance matrices are not supported as an input for KernelClustering.", id="No.62")
    ],
)
def test_fit_predict(create_fit_data, No, client, e, if_dist, err_msg):
    n_clusters = 2
    indata = create_fit_data.get(No)
    kcl = KernelClustering(n_clusters=n_clusters)
    if (client == "N"):
        client = None
    elif (client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")

    if (e == "N"):
        labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(isinstance(labels, list))
        assert(len(labels) == 64)
        assert(len(set(labels)) == n_clusters)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.52"),
    ]
)
@pytest.mark.neal
def test_fit_predict_err_neal(lack_package):
    client = None
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        labels = kcl.fit_predict(data, client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.53"),
    ]
)
@pytest.mark.dimod
def test_fit_predict_err_dimod(lack_package):
    client = None
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        labels = kcl.fit_predict(data, client)

@pytest.mark.parametrize(
    ["Case", "err_msg"],
    [
        pytest.param("InvalidToken", "exceptions were raised from solve() of the Fixstars Amplify SDK", id="No.54"),
    ],
)
def test_fit_predict_error(Case, err_msg):
    client = setup_fixstars()
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    if (Case == "InvalidToken"):
        client.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            label = kcl.fit_predict(data, client)
    else:
        raise Exception("Invalid Case")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "client", "e", "if_dist", "err_msg"],
    [
        pytest.param("33", "N", "N", False, None, id="additional33"),
        pytest.param("34", "N", "N", False, None, id="additional34"),
        pytest.param("37", "F", "N", False, None, id="additional37"),
        pytest.param("38", "F", "N", False, None, id="additional38"),
        pytest.param("41", "F", "T", False, "must be array-like object", id="additional41"),
        pytest.param("42", "F", "T", False, "has invalid data type", id="additional42"),
        pytest.param("43", "F", "T", False, "has invalid data type", id="additional43"),
        pytest.param("44", "F", "V", False, "has invalid shape", id="additional44"),
        pytest.param("46", "F", "V", False, "must be 2-d array", id="additional46"),
        pytest.param("47", "F", "V", False, "must be 2-d array", id="additional47"),
        pytest.param("48", "F", "V", False, "must be 2-d array", id="additional48"),
        pytest.param("49", "F", "V", False, "must be 2-d array", id="additional49"),
        pytest.param("62", "F", "T", True, "Distance matrices are not supported as an input for KernelClustering.", id="No.additional62")
    ],
)
def test_fit_predict_additional(create_fit_data, No, client, e, if_dist, err_msg):
    n_clusters = 2
    indata = create_fit_data.get(No)
    kcl = KernelClustering(n_clusters=n_clusters, sigma=0.5, lam=64)
    if (client == "N"):
        client = None
    elif (client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")

    if (e == "N"):
        labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(isinstance(labels, list))
        assert(len(labels) == 64)
        assert(len(set(labels)) == n_clusters)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            labels = kcl.fit_predict(indata, client, if_dist=if_dist)
        assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["No", "labels_dummy"],
    [
        pytest.param("55", ([0] * 32 + [1] * 31 + [-1]), id="No.55"),
        pytest.param("56", ([0] * 32 + [1] * 31 + [2]), id="No.56"),
        pytest.param("57", [0] * 64, id="No.57"),
    ]
)
def test_predict_constraint_violation(mocker, No, labels_dummy, create_err_msg):
    client = setup_fixstars()
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    
    kcl.fit(data)
    client = setup_fixstars()
    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        label = kcl.predict(client)

    err_msg = create_err_msg.get(No)
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "labels_dummy"],
    [
        pytest.param("58", ([0] * 32 + [1] * 31 + [-1]), id="No.58"),
        pytest.param("59", ([0] * 32 + [1] * 31 + [2]), id="No.59"),
        pytest.param("60", [0] * 64, id="No.60"),
    ],
)
def test_fit_predict_constraint_violation(mocker, No, labels_dummy, create_err_msg):
    client = setup_fixstars()
    n_clusters = 2
    kcl = KernelClustering(n_clusters=n_clusters)
    data, _ = make_circles(n_samples=64, factor=0.3,
                           noise=0.05, random_state=0)
    
    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        label = kcl.fit_predict(data, client)
    
    err_msg = create_err_msg.get(No)
    assert(err_msg in str(e.value))