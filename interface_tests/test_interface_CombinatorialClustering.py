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
from qklearn.cluster import CombinatorialClustering
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_blobs
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

    err_msg["3"]  = "n_clusters must be greater than 0"
    err_msg["4"]  = "n_clusters must be int"
    err_msg["7"]  = "lam must be float"
    err_msg["12"]  = "must be array-like object"
    err_msg["13"]  = "has invalid data type"
    err_msg["14"]  = "has invalid data type"
    err_msg["15"]  = "has invalid shape"
    err_msg["16"]  = "has invalid shape"
    err_msg["17"]  = "must be 2-d array"
    err_msg["18"]  = "must be 2-d array"
    err_msg["19"]  = "must be 2-d array"
    err_msg["20"]  = "must be 2-d array"
    err_msg["21"]  = "must be square matrix"
    err_msg["22"]  = "must be square matrix"
    err_msg["27"]  = "exceptions were raised from solve() of the Fixstars Amplify SDK"
    err_msg["28"]  = "This instance is not fitted yet"
    err_msg["37"]  = "must be array-like object"
    err_msg["38"]  = "has invalid data type"
    err_msg["39"]  = "has invalid data type"
    err_msg["40"]  = "has invalid shape"
    err_msg["41"]  = "has invalid shape"
    err_msg["42"]  = "must be 2-d array"
    err_msg["43"]  = "must be 2-d array"
    err_msg["44"]  = "must be 2-d array"
    err_msg["45"]  = "must be 2-d array"
    err_msg["46"]  = "must be square matrix"
    err_msg["47"]  = "must be square matrix"
    err_msg["50"]  = "exceptions were raised from solve() of the Fixstars Amplify SDK"
    err_msg["51"]  = "The cluster labels of the samples that do not satisfy the constraint are set to -1. Please execute the program multiple times or adjust the parameters."
    err_msg["52"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["53"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["54"]  = err_msg["51"]
    err_msg["55"]  = err_msg["51"]
    err_msg["56"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["57"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["58"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters." 
    err_msg["59"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters." 

    return err_msg

@pytest.mark.parametrize(
    ["n_clusters", "lam", "e", "No"],
    [
        pytest.param(1,   None, "N", "1", id="No.1"),
        pytest.param(2,   None, "N", "2", id="No.2"),
        pytest.param(0,   None, "V", "3", id="No.3"),
        pytest.param(2.5, None, "T", "4", id="No.4"),
        pytest.param(3,   10,   "N", "5", id="No.5"),
        pytest.param(3,   11.1, "N", "6", id="No.6"),
        pytest.param(3,   "12", "T", "7", id="No.7"),
    ]
)
def test_constructor(n_clusters, lam, e, No, create_err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    client = setup_fixstars()
    err_msg = create_err_msg.get(No)
    if (e == "N"):
        if (lam is None):
            ccl = CombinatorialClustering(n_clusters=n_clusters)
        else:
            ccl = CombinatorialClustering(n_clusters=n_clusters, lam=lam)
        ccl.fit(data)
        labels = ccl.predict(client)
        assert(isinstance(labels, list))
        assert(len(labels) == 10)
        assert(len(set(labels)) == n_clusters)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            if (lam is None):
                ccl = CombinatorialClustering(n_clusters=n_clusters)
            else:
                ccl = CombinatorialClustering(n_clusters=n_clusters, lam=lam)
            assert(err_msg in str(e.value))
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            if (lam is None):
                ccl = CombinatorialClustering(n_clusters=n_clusters)
            else:
                ccl = CombinatorialClustering(n_clusters=n_clusters, lam=lam)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error Type")

@pytest.fixture(scope="module")
def create_fit_data():
    # {"case No." : input}
    test_data = {}
    dataf2, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                           cluster_std=1.5, centers=3)
    dataf3, _ = make_blobs(random_state=8, n_samples=10, n_features=3,
                           cluster_std=1.5, centers=3)
    distf2 = cdist(dataf2, dataf2, metric="euclidean")
    distf3 = cdist(dataf3, dataf3, metric="euclidean")
    invalid_listf2 = [[i ,i] for i in range(10)]
    invalid_listf2[9][1] = "h"
    invalid_listf3 = [[i ,i, i] for i in range(10)]
    invalid_listf3[9][2] = "h"

    test_data["8"]  = dataf2
    test_data["9"]  = dataf3.tolist()
    test_data["10"] = distf2
    test_data["11"] = distf3.tolist()
    test_data["12"] = {1: (1, 1)}
    test_data["13"] = np.array(invalid_listf2)
    test_data["14"] = invalid_listf3
    test_data["15"] = [[1,2,3], [1,2]]
    test_data["16"] = [[1,2,3], [1,2]]
    test_data["17"] = np.arange(8).reshape((2, 2, 2))
    test_data["18"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["19"] = np.arange(8).reshape((8,))
    test_data["20"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data["21"] = np.arange(12).reshape((4,3))
    test_data["22"] = [[1, 2, 3],  [4, 5, 6]]
    test_data["29"]  = dataf2
    test_data["30"]  = dataf3.tolist()
    test_data["31"] = distf2
    test_data["32"] = distf3.tolist()
    test_data["33"]  = dataf2
    test_data["34"]  = dataf3.tolist()
    test_data["35"] = distf2
    test_data["36"] = distf3.tolist()
    test_data["37"] = {1: (1, 1)}
    test_data["38"] = np.array(invalid_listf2)
    test_data["39"] = invalid_listf3
    test_data["40"] = [[1,2,3], [1,2]]
    test_data["41"] = [[1,2,3], [1,2]]
    test_data["42"] = np.arange(8).reshape((2, 2, 2))
    test_data["43"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["44"] = np.arange(8).reshape((8,))
    test_data["45"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data["46"] = X=np.arange(12).reshape((4,3))
    test_data["47"] = [[1, 2, 3],  [4, 5, 6]]

    return test_data

@pytest.mark.parametrize(
    ["No", "dist", "e"],
    [
        pytest.param("8",  False, "N", id="No.8"),
        pytest.param("9",  False, "N", id="No.9"),
        pytest.param("10", True,  "N", id="No.10"),
        pytest.param("11", True,  "N", id="No.11"),
        pytest.param("12", False, "T", id="No.12"),
        pytest.param("13", False, "T", id="No.13"),
        pytest.param("14", False, "T", id="No.14"),
        pytest.param("15", False, "V", id="No.15"),
        pytest.param("16", True,  "V", id="No.16"),
        pytest.param("17", False, "V", id="No.17"),
        pytest.param("18", False, "V", id="No.18"),
        pytest.param("19", False, "V", id="No.19"),
        pytest.param("20", False, "V", id="No.20"),
        pytest.param("21", True,  "V", id="No.21"),
        pytest.param("22", True,  "V", id="No.22"),
    ],
)
def test_fit(create_fit_data, dist, No, e, create_err_msg):
    ccl = CombinatorialClustering(n_clusters = 3)
    indata = create_fit_data.get(No)
    err_mesg = create_err_msg.get(No)
    if (e == "N"):
        ccl.fit(indata, if_dist=dist)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            ccl.fit(indata, if_dist=dist)
            assert(err_mesg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            ccl.fit(indata, if_dist=dist)
            assert(err_mesg in str(e.value))
    else:
        raise Exception("Invalid Error Type")

@pytest.mark.parametrize(
    ["Client"],
    [
        pytest.param("N", id="No.23"),
        pytest.param("F", id="No.24"),
    ]
)
def test_predict_normal(Client):
    sample_num = 10
    n_clusters = 3
    ccl = CombinatorialClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=sample_num, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    if (Client == "N"):
        client = None
    elif (Client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid client")

    labels = ccl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == sample_num)
    assert(len(set(labels)) == n_clusters)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.25"),
    ],
)
@pytest.mark.neal
def test_predict_error_neal(lack_package):
    client = None
    ccl = CombinatorialClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = ccl.predict(client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.26"),
    ],
)
@pytest.mark.dimod
def test_predict_error_dimod(lack_package):
    client = None
    ccl = CombinatorialClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = ccl.predict(client)

@pytest.mark.parametrize(
    ["Case", "No"],
    [
        pytest.param("InvalidToken", "27", id="No.27"),
        pytest.param("NotFitted", "28", id="No.28"),
    ],
)
def test_predict_error(Case, No, create_err_msg):
    client = setup_fixstars()
    ccl = CombinatorialClustering(n_clusters=3)
    err_msg = create_err_msg.get(No)
    if (Case == "InvalidToken"):
        client.token = "hoge"
        data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                             cluster_std=1.5, centers=3)
        ccl.fit(data)
        with pytest.raises(RuntimeError) as e:
            label = ccl.predict(client)
    elif (Case == "NotFitted"):
        with pytest.raises(AttributeError) as e:
            label = ccl.predict(client)
    else:
        raise Exception("Invalid Case")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "client", "e", "if_dist"],
    [
        pytest.param("29", "N", "N", False, id="No.29"),
        pytest.param("30", "N", "N", False, id="No.30"),
        pytest.param("31", "N", "N", True,  id="No.31"),
        pytest.param("32", "N", "N", True,  id="No.32"),
        pytest.param("33", "F", "N", False, id="No.33"),
        pytest.param("34", "F", "N", False, id="No.34"),
        pytest.param("35", "F", "N", True,  id="No.35"),
        pytest.param("36", "F", "N", True,  id="No.36"),
        pytest.param("37", "F", "T", False, id="No.37"),
        pytest.param("38", "F", "T", False, id="No.38"),
        pytest.param("39", "F", "T", False, id="No.39"),
        pytest.param("40", "F", "V", False, id="No.40"),
        pytest.param("41", "F", "V", True,  id="No.41"),
        pytest.param("42", "F", "V", False, id="No.42"),
        pytest.param("43", "F", "V", False, id="No.43"),
        pytest.param("44", "F", "V", False, id="No.44"),
        pytest.param("45", "F", "V", False, id="No.45"),
        pytest.param("46", "F", "V", True,  id="No.46"),
        pytest.param("47", "F", "V", True,  id="No.47"),
    ],
)
def test_fit_predict(create_fit_data, No, client, e, if_dist, create_err_msg):
    n_clusters = 3
    indata = create_fit_data.get(No)
    ccl = CombinatorialClustering(n_clusters=n_clusters)
    err_msg = create_err_msg.get(No)

    if (client == "N"):
        client = None
    elif (client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")
    if (e == "N"):
        labels = ccl.fit_predict(indata, client, if_dist=if_dist)  
        assert(isinstance(labels, list))
        assert(len(labels) == 10)
        assert(len(set(labels)) == n_clusters)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            labels = ccl.fit_predict(indata, client, if_dist=if_dist)
            assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            labels = ccl.fit_predict(indata, client, if_dist=if_dist)
            assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.48"),
    ],
)
@pytest.mark.neal
def test_fit_predict_err_neal(lack_package):
    client = None
    n_clusters = 3
    ccl = CombinatorialClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=10,
                         n_features=2, cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        labels = ccl.fit_predict(data, client)


@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.49"),
    ],
)
@pytest.mark.dimod
def test_fit_predict_error_dimod(lack_package):
    client = None
    n_clusters = 3
    ccl = CombinatorialClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=10,
                         n_features=2, cluster_std=1.5, centers=2)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        labels = ccl.fit_predict(data, client)

@pytest.mark.parametrize(
    ["Case", "No"],
    [
        pytest.param("InvalidToken", "50", id="No.50"),
    ],
)
def test_fit_predict_error(Case, No, create_err_msg):
    client = setup_fixstars()
    ccl = CombinatorialClustering(n_clusters=3)
    err_msg = create_err_msg.get(No)
    if (Case == "InvalidToken"):
        client.token = "hoge"
        data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                             cluster_std=1.5, centers=3)
        with pytest.raises(RuntimeError) as e:
            label = ccl.fit_predict(data, client)
    else:
        raise Exception("Invalid Case")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "client", "e", "if_dist"],
    [
        pytest.param("29", "N", "N", False, id="additional29"),
        pytest.param("30", "N", "N", False, id="additional30"),
        pytest.param("31", "N", "N", True,  id="additional31"),
        pytest.param("32", "N", "N", True,  id="additional32"),
        pytest.param("33", "F", "N", False, id="additional33"),
        pytest.param("34", "F", "N", False, id="additional34"),
        pytest.param("35", "F", "N", True,  id="additional35"),
        pytest.param("36", "F", "N", True,  id="additional36"),
        pytest.param("37", "F", "T", False, id="additional37"),
        pytest.param("38", "F", "T", False, id="additional38"),
        pytest.param("39", "F", "T", False, id="additional39"),
        pytest.param("40", "F", "V", False, id="additional40"),
        pytest.param("41", "F", "V", True,  id="additional41"),
        pytest.param("42", "F", "V", False, id="additional42"),
        pytest.param("43", "F", "V", False, id="additional43"),
        pytest.param("44", "F", "V", False, id="additional44"),
        pytest.param("45", "F", "V", False, id="additional45"),
        pytest.param("46", "F", "V", True,  id="additional46"),
        pytest.param("47", "F", "V", True,  id="additional47"),
    ],
)
def test_fit_predict_additional(create_fit_data, No, client, e, if_dist, create_err_msg):
    n_clusters = 3
    indata = create_fit_data.get(No)
    ccl = CombinatorialClustering(n_clusters=n_clusters, lam=10)
    err_msg = create_err_msg.get(No)

    if (client == "N"):
        client = None
    elif (client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")
    if (e == "N"):
        labels = ccl.fit_predict(indata, client, if_dist=if_dist)  
        assert(isinstance(labels, list))
        assert(len(labels) == 10)
        assert(len(set(labels)) == n_clusters)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            labels = ccl.fit_predict(indata, client, if_dist=if_dist)
            assert(err_msg in str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            labels = ccl.fit_predict(indata, client, if_dist=if_dist)
            assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["No", "labels_dummy"],
    [
        pytest.param("51", [0, 1, 2, 0, 1, 2, 0, 1, 2, -1], id="No.51"),
        pytest.param("52", [0, 0, 1, 1, 2, 2, 3, 3, 3, 4], id="No.52"),
        pytest.param("53", [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], id="No.53"),
    ],
)
def test_predict_constraint_violation(mocker, No, labels_dummy, create_err_msg):
    ccl = CombinatorialClustering(n_clusters = 3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                        cluster_std=1.5, centers=3)
    err_msg = create_err_msg.get(No)

    ccl.fit(data)
    client = setup_fixstars()
    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        labels = ccl.predict(client)
    assert err_msg in str(e.value)
    print(str(e.value))

@pytest.mark.parametrize(
    ["No", "if_dist", "labels_dummy"],
    [
        pytest.param("54", True, [0, 1, 2, 0, 1, 2, 0, 1, 2, -1], id="No.54"),
        pytest.param("55", False, [0, 1, 2, 0, 1, 2, 0, 1, 2, -1], id="No.55"),
        pytest.param("56", True, [0, 0, 1, 1, 2, 2, 3, 3, 3, 4], id="No.56"),
        pytest.param("57", False, [0, 0, 1, 1, 2, 2, 3, 3, 3, 4], id="No.57"),
        pytest.param("58", True, [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], id="No.58"),
        pytest.param("59", False, [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], id="No.59"),
    ],
)
def test_fit_predict_constraint_violation(mocker, No, if_dist, labels_dummy, create_err_msg):
    ccl = CombinatorialClustering(n_clusters = 3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                        cluster_std=1.5, centers=3)
    dist = cdist(data, data, metric="euclidean")
    err_msg = create_err_msg.get(No)

    client = setup_fixstars()
    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        if if_dist:
            labels = ccl.fit_predict(dist, client, if_dist)
        else:
            labels = ccl.fit_predict(data, client)
    assert err_msg in str(e.value)
