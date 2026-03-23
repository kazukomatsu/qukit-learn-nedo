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
from qklearn.cluster import BinaryClustering
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_blobs
import numpy as np

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.fixture(scope="module")
def create_fit_data():
    # {"case No." : input}
    test_data = {}
    dataf2, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    dataf3, _ = make_blobs(random_state=8, n_samples=10, n_features=3,
                         cluster_std=1.5, centers=2)
    invalid_listf2 = [[i ,i] for i in range(10)]
    invalid_listf2[9][1] = "h"
    invalid_listf3 = [[i ,i, i] for i in range(10)]
    invalid_listf3[9][2] = "h"

    test_data["1"] = dataf2
    test_data["2"] = dataf3.tolist()
    test_data["3"] = {1: (1, 1)}
    test_data["4"] = np.array(invalid_listf2)
    test_data["5"] = invalid_listf3
    test_data["6"] = [[1,2,3], [1,2]]
    test_data["7"] = np.arange(8).reshape((2, 2, 2))
    test_data["8"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["9"] = np.arange(8).reshape((8,))
    test_data["10"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data["17"] = dataf2
    test_data["18"] = dataf3.tolist()
    test_data["19"] = dataf2
    test_data["20"] = dataf3.tolist()
    test_data["21"] = {1: (1, 1)}
    test_data["22"] = np.array(invalid_listf2)
    test_data["23"] = invalid_listf3
    test_data["24"] = [[1,2,3], [1,2]]
    test_data["25"] = np.arange(8).reshape((2, 2, 2))
    test_data["26"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["27"] = np.arange(8).reshape((8,))
    test_data["28"] = [1, 2, 3, 4, 5, 6, 7, 8]

    return test_data

@pytest.fixture(scope="module")
def create_err_msg():
    err_msg = {}

    err_msg["3"]  = "must be array-like object"
    err_msg["4"]  = "has invalid data type"
    err_msg["5"]  = "has invalid data type"
    err_msg["6"]  = "has invalid shape"
    err_msg["7"]  = "must be 2-d array"
    err_msg["8"]  = "must be 2-d array"
    err_msg["9"]  = "must be 2-d array"
    err_msg["10"] = "must be 2-d array"
    err_msg["15"] = "exceptions were raised from solve() of the Fixstars Amplify SDK"
    err_msg["16"] = "This instance is not fitted yet"
    err_msg["21"]  = "must be array-like object"
    err_msg["22"]  = "has invalid data type"
    err_msg["23"]  = "has invalid data type"
    err_msg["24"]  = "has invalid shape"
    err_msg["25"]  = "must be 2-d array"
    err_msg["26"]  = "must be 2-d array"
    err_msg["27"]  = "must be 2-d array"
    err_msg["28"] = "must be 2-d array"
    err_msg["31"] = "exceptions were raised from solve() of the Fixstars Amplify SDK"

    return err_msg

@pytest.mark.parametrize(
    ["No"],
    [
        pytest.param("1", id="No.1"),
        pytest.param("2", id="No.2"),
    ],
)
def test_fit_normal(create_fit_data, No):
    bcl = BinaryClustering()
    input = create_fit_data.get(No)
    bcl.fit(input)

@pytest.mark.parametrize(
    ["No", "e"],
    [
        pytest.param("3",  "T", id="No.3"),
        pytest.param("4",  "T", id="No.4"),
        pytest.param("5",  "T", id="No.5"),
        pytest.param("6",  "V", id="No.6"),
        pytest.param("7",  "V", id="No.7"),
        pytest.param("8",  "V", id="No.8"),
        pytest.param("9",  "V", id="No.9"),
        pytest.param("10", "V", id="No.10"),
    ],
)
def test_fit_error(create_fit_data, create_err_msg, No, e):
    bcl = BinaryClustering()
    err_msg = create_err_msg.get(No)
    input = create_fit_data.get(No)
    if (e == "T"):
        with pytest.raises(TypeError) as e:
            bcl.fit(input)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            bcl.fit(input)
    else:
        raise Exception("Invalid Error Type")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["Client"],
    [
        pytest.param("N", id="No.11"),
        pytest.param("F", id="No.12"),
    ],
)
def test_predict_normal(Client):
    bcl = BinaryClustering()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    bcl.fit(data)

    if (Client == "F"):
        client = setup_fixstars()
    elif (Client == "N"):
        client = None
    else:
        raise Exception("Invalid Client Type")

    label = bcl.predict(client)
    assert(isinstance(label, list))
    assert(len(label) == 10)
    assert(len(set(label)) == 2)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.13"),
    ],
)
@pytest.mark.neal
def test_predict_error_neal(lack_package):
    client = None
    bcl = BinaryClustering()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    bcl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = bcl.predict(client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.14"),
    ],
)
@pytest.mark.dimod
def test_predict_error_dimod(lack_package):
    client = None
    bcl = BinaryClustering()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    bcl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = bcl.predict(client)

@pytest.mark.parametrize(
    ["Case"],
    [
        pytest.param("Invalid_Token", id="No.15"),
        pytest.param("Not_Fitted", id="No.16"),
    ]
)
def test_predict_error_fixstars(Case, create_err_msg):
    client = setup_fixstars()
    if (Case == "Invalid_Token"):
        err_msg = create_err_msg.get("15")
        client.token = "token"
        data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                             cluster_std=1.5, centers=2)
        bcl = BinaryClustering()
        bcl.fit(data)
        with pytest.raises(RuntimeError) as e:
            label = bcl.predict(client)
    elif (Case == "Not_Fitted"):
        err_msg = create_err_msg.get("16")
        bcl = BinaryClustering()
        with pytest.raises(AttributeError) as e:
            label = bcl.predict(client)
    else:
        raise Exception("Invalid test case")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "e", "C"],
    [
        pytest.param("17",  "N", "F", id="No.17"),
        pytest.param("18",  "N", "N", id="No.18"),
        pytest.param("19",  "N", "N", id="No.19"),
        pytest.param("20",  "N", "F", id="No.20"),
        pytest.param("21",  "T", "F", id="No.21"),
        pytest.param("22",  "T", "F", id="No.22"),
        pytest.param("23",  "T", "F", id="No.23"),
        pytest.param("24",  "V", "F", id="No.24"),
        pytest.param("25",  "V", "F", id="No.25"),
        pytest.param("26",  "V", "F", id="No.26"),
        pytest.param("27",  "V", "F", id="No.27"),
        pytest.param("28",  "V", "F", id="No.28"),
    ],
)
def test_fit_predict(create_fit_data, create_err_msg, No, e, C):
    if (C == "F"):
        client = setup_fixstars()
    elif (C == "N"):
        client = None
    else:
        raise Exception("Invalid Client")

    err_msg = create_err_msg.get(No)
    input = create_fit_data.get(No)
    bcl = BinaryClustering()

    if (e == "N"):
        label = bcl.fit_predict(input, client)
        assert(isinstance(label, list))
        assert(len(label) == 10)
        assert(len(set(label)) == 2)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            label = bcl.fit_predict(input, client)
        assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            label = bcl.fit_predict(input, client)
        assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error Type")


@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.29"),
    ],
)
@pytest.mark.neal
def test_fit_predict_error_neal(lack_package):
    client = None
    bcl = BinaryClustering()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = bcl.fit_predict(data, client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.30"),
    ],
)
@pytest.mark.dimod
def test_fit_predict_error_dimod(lack_package):
    client = None
    bcl = BinaryClustering()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = bcl.fit_predict(data, client)

@pytest.mark.parametrize(
    ["Case"],
    [
        pytest.param("Invalid_Token", id="No.31"),
    ],
)
def test_fit_predict_error_fixstars(Case, create_err_msg):
    err_msg = create_err_msg.get("31")
    client = setup_fixstars()
    client.token = "token"
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    bcl = BinaryClustering()
    with pytest.raises(RuntimeError) as e:
        bcl.fit_predict(data, client)
    assert(err_msg in str(e.value))
