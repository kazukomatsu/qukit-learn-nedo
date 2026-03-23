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
from qklearn.cluster import BiDViT
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_blobs
import numpy as np

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.mark.parametrize(
    ["kappa", "e", "err_msg"],
    [
        pytest.param(None, "N", None, id="No.1"),
        pytest.param(1,    "V", "kappa must be greater than 1", id="No.2"),
        pytest.param(2,    "V", "Chunk size became too small to build a valid QUBO (chunk_size <= 1). "+\
                      "Increase kappa or use kappa >= n_samples for small datasets.", id="No.3"),
        pytest.param(0,    "V", "kappa must be greater than 1", id="No.4"),
        pytest.param(1.1,  "T", "kappa must be int", id="No.5"),
    ]
)
def test_constructor_kappa(kappa, e, err_msg):
    client = setup_fixstars()
    n_samples = 30
    if (e == "N"):
        if(kappa is None):
            b = BiDViT()
        else:
            b = BiDViT(kappa=kappa)
        data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                             cluster_std=1.5, centers=2)
        tree = b.fit_predict(data, client)
        assert(isinstance(tree, dict))
        for k, v in tree.items():
            assert(len(v) == n_samples)
            assert(len(set(v)) == k)
    elif(e == "V"):
        with pytest.raises(ValueError) as e:
            b = BiDViT(kappa=kappa)
            data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                             cluster_std=1.5, centers=2)
            tree = b.fit_predict(data, client)
        assert(err_msg in str(e.value))
    elif(e == "T"):
        with pytest.raises(TypeError) as e:
            b = BiDViT(kappa=kappa)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["epsilon", "e", "err_msg"],
    [
        pytest.param(0.1,   "N", None, id="No.6"),
        pytest.param(1,     "N", None, id="No.7"),
        pytest.param(0,     "V", "epsilon must be greater than 0", id="No.8"),
        pytest.param("0.1", "T", "epsilon must be float", id="No.9"),
    ]
)
def test_constructor_epsilon(epsilon, e, err_msg):
    client = setup_fixstars()
    n_samples = 30
    if (e == "N"):
        b = BiDViT(epsilon=epsilon)
        data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                             cluster_std=1.5, centers=2)
        tree = b.fit_predict(data, client)
        assert(isinstance(tree, dict))
        for k, v in tree.items():
            assert(len(v) == n_samples)
            assert(len(set(v)) == k)
    elif(e == "V"):
        with pytest.raises(ValueError) as e:
            b = BiDViT(epsilon=epsilon)
            assert(err_msg in str(e.value))
    elif(e == "T"):
        with pytest.raises(TypeError) as e:
            b = BiDViT(epsilon=epsilon)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["epsilon_rate", "e", "err_msg"],
    [
        pytest.param(1.1,   "N", None, id="No.10"),
        pytest.param(2,     "N", None, id="No.11"),
        pytest.param(1,     "V", "epsilon_rate must be greater than 0", id="No.12"),
        pytest.param("1.1", "T", "epsilon_rate must be float", id="No.13"),
    ]
)
def test_constructor_epsilon_rate(epsilon_rate, e, err_msg):
    client = setup_fixstars()
    n_samples = 30
    if (e == "N"):
        b = BiDViT(epsilon_rate=epsilon_rate)
        data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                             cluster_std=1.5, centers=2)
        tree = b.fit_predict(data, client)
        assert(isinstance(tree, dict))
        for k, v in tree.items():
            assert(len(v) == n_samples)
            assert(len(set(v)) == k)
    elif(e == "V"):
        with pytest.raises(ValueError) as e:
            b = BiDViT(epsilon_rate=epsilon_rate)
            assert(err_msg in str(e.value))
    elif(e == "T"):
        with pytest.raises(TypeError) as e:
            b = BiDViT(epsilon_rate=epsilon_rate)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["rev_rate", "e", "err_msg"],
    [
        pytest.param(2,     "N", None, id="No.14"),
        pytest.param("1.1", "T", "rev_rate must be float", id="No.15"),
    ]
)
def test_constructor_rev_rate(rev_rate, e, err_msg):
    client = setup_fixstars()
    n_samples = 30
    if (e == "N"):
        b = BiDViT(rev_rate=rev_rate)
        data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                             cluster_std=1.5, centers=2)
        tree = b.fit_predict(data, client)
        assert(isinstance(tree, dict))
        for k, v in tree.items():
            assert(len(v) == n_samples)
            assert(len(set(v)) == k)
    elif(e == "T"):
        with pytest.raises(TypeError) as e:
            b = BiDViT(rev_rate=rev_rate)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["X", "client", "additional"],
    [
        pytest.param("N", "N", False, id="No.16"),
        pytest.param("L", "N", False, id="No.17"),
        pytest.param("N", "F", False, id="No.18"),
        pytest.param("L", "F", False, id="No.19"),
        pytest.param("N", "N", True,  id="No.additional16"),
        pytest.param("L", "N", True,  id="No.additional17"),
        pytest.param("N", "F", True,  id="No.additional18"),
        pytest.param("L", "F", True,  id="No.additional19"),
    ]
)
def test_fit_predict_client_normal(X, client, additional):
    if (additional):
        b = BiDViT(kappa=16, epsilon=1.5, epsilon_rate=2, rev_rate=2)
    else:
        b = BiDViT()
    if (client == "N"):
        client = None
    elif (client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")

    n_samples = 30
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    if (X == "L"):
        data = data.tolist()
    elif (X == "N"):
        pass
    else:
        raise Exception("Invalid X")

    tree = b.fit_predict(data, client)
    assert(isinstance(tree, dict))
    for k, v in tree.items():
        assert(len(v) == n_samples)
        assert(len(set(v)) == k)

@pytest.fixture(scope="module")
def create_fit_data():
    test_data = {}

    invalid_listf2 = [[i ,i] for i in range(30)]
    invalid_listf2[9][1] = "h"
    invalid_listf3 = [[i ,i, i] for i in range(30)]
    invalid_listf3[9][2] = "h"

    test_data["20"] = {1: (1, 1)}
    test_data["21"] = np.array(invalid_listf2)
    test_data["22"] = invalid_listf3
    test_data["23"] = [[1,2,3], [1,2]]
    test_data["24"] = np.arange(8).reshape((2, 2, 2))
    test_data["25"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["26"] = np.arange(8).reshape((8,))
    test_data["27"] = [1, 2, 3, 4, 5, 6, 7, 8]

    return test_data

@pytest.mark.parametrize(
    ["No", "e", "err_msg"],
    [
        pytest.param("20", "T", "must be array-like object", id="No.20"),
        pytest.param("21", "T", "has invalid data type", id="No.21"),
        pytest.param("22", "T", "has invalid data type", id="No.22"),
        pytest.param("23", "V", "has invalid shape", id="No.23"),
        pytest.param("24", "V", "must be 2-d array", id="No.24"),
        pytest.param("25", "V", "must be 2-d array", id="No.25"),
        pytest.param("26", "V", "must be 2-d array", id="No.26"),
        pytest.param("27", "V", "must be 2-d array", id="No.27"),
    ]
)
def test_fit_predict_client_X_error(create_fit_data, No, e, err_msg):
    b = BiDViT()
    client = setup_fixstars()
    data = create_fit_data.get(No)

    if (e == "T"):
        with pytest.raises(TypeError) as e:
            b.fit_predict(data, client)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            b.fit_predict(data, client)
    else:
        raise Exception("Invalid error")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.28"),
    ],
)
@pytest.mark.neal
def test_fit_predict_client_error_neal(lack_package):
    b = BiDViT()
    client = None
    n_samples = 30
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        b.fit_predict(data, client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.29"),
    ],
)
@pytest.mark.dimod
def test_fit_predict_client_error_dimod(lack_package):
    b = BiDViT()
    client = None
    n_samples = 30
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        b.fit_predict(data, client)

@pytest.mark.parametrize(
    ["Case", "err_msg"],
    [
        pytest.param("InvalidToken", "exceptions were raised from solve() of the Fixstars Amplify SDK", id="No.30"),
    ]
)
def test_fit_predict_client_error_f(Case, err_msg):
    b = BiDViT()
    client = setup_fixstars()
    n_samples = 30
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    if (Case == "InvalidToken"):
        client.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            b.fit_predict(data, client)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["weight"],
    [
        pytest.param([1.5]*30, id="No.31"),
        pytest.param([2]*30,   id="No.32"),
    ]
)
def test_fit_predict_weight_normal(weight):
    kappa = 15
    b = BiDViT(kappa=kappa)
    client = setup_fixstars()
    n_samples = 30
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    tree = b.fit_predict(data, client, weight=weight)
    assert(isinstance(tree, dict))
    for k, v in tree.items():
        assert(len(v) == n_samples)
        assert(len(set(v)) == k)
    
@pytest.mark.parametrize(
    ["weight", "e", "err_msg"],
    [
        pytest.param([2]*29,   "V", "The length of weight must be equal to n_sample", id="No.33"),
        pytest.param(2,        "T", "weight must be list", id="No.34"),
        pytest.param(["2"]*30, "T", "weight must be list of float", id="No.35"),
    ]
)
def test_fit_predict_weight_error(weight, e, err_msg):
    kappa = 15
    b = BiDViT(kappa=kappa)
    n_samples = 30
    client = setup_fixstars()
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=2)
    if (e == "V"):
        with pytest.raises(ValueError) as e:
            tree = b.fit_predict(data, client, weight=weight)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            tree = b.fit_predict(data, client, weight=weight)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))









