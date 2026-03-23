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
from qklearn.cluster import ConsensusClustering
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.fixture(scope="module")
def create_err_msg():
    err_msg = {}

    err_msg["4"] = "n_clusters must be greater than 0"
    err_msg["5"] = "n_clusters must be int"
    err_msg["6"] = "lam must be float"
    err_msg["7"] = "model hoge is invalid"
    err_msg["8"] = "model must be str"
    err_msg["9"] = "must be array-like object"
    err_msg["10"] = "has invalid data type"
    err_msg["11"] = "has invalid data type"
    err_msg["12"] = "has invalid shape"
    err_msg["13"] = "must be 2-d array"
    err_msg["14"] = "must be 2-d array"
    err_msg["15"] = "must be 2-d array"
    err_msg["16"] = "must be 2-d array"
    err_msg["20"]  = "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1."
    err_msg["21"]  = "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1."
    err_msg["22"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["23"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["24"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["25"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters."

    return err_msg

@pytest.fixture(scope="module")
def create_fit_data():
    # {"case No." : input}
    test_data = {}
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = CombinatorialClustering(n_clusters=3)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)
    ccl_label[0] = "c"

    scl = KMeans(n_clusters=3)
    scl_label = scl.fit_predict(data).tolist()
    scl_label[0] = "s"

    result = [ccl_label, scl_label]

    test_data["9"] = {1: (1, 1)}
    test_data["10"] = np.array(result)
    test_data["11"] = result
    test_data["12"] = [[1,2,3], [1,2]]
    test_data["13"] = np.arange(8).reshape((2, 2, 2))
    test_data["14"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["15"] = np.arange(8).reshape((8,))
    test_data["16"] = [1, 2, 3, 4, 5, 6, 7, 8]

    return test_data

@pytest.mark.parametrize(
    ["n_clusters", "lam", "model", "client", "rtype"],
    [
        pytest.param(1,   None, None, "F", "L", id="No.1"),
        pytest.param(2,   10, "pairwise_similarity-based", "N", "N", id="No.2"),
        pytest.param(2,   10.5, "partition_difference", "F", "N", id="No.3"),
    ]
)
def test_constructor_normal(n_clusters, lam, model, client, rtype):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=n_clusters)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)

    scl = KMeans(n_clusters=n_clusters)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    if (rtype == "N"):
        result = np.array(result)

    if (lam is None and model is None):
        concl = ConsensusClustering(n_clusters=n_clusters)
    else:
        concl = ConsensusClustering(n_clusters=n_clusters,
                                    lam=lam, model=model)

    if (client == "F"):
        con_label = concl.fit_predict(result, fclient)
    elif(client == "N"):
        con_label = concl.fit_predict(result, None)
    else:
        raise Exception("Invalid Client")

    assert(isinstance(con_label, list))
    assert(len(con_label) == 10)
    assert(len(set(con_label)) == n_clusters)

@pytest.mark.parametrize(
    ["n_clusters", "lam", "model", "No"],
    [
        pytest.param(0,   None, None,   "4", id="No.4"),
        pytest.param(2.5, None, None,   "5", id="No.5"),
        pytest.param(3,   "12", None,   "6", id="No.6"),
        pytest.param(3,   None, "hoge", "7", id="No.7"),
        pytest.param(3,   None, 3,      "8", id="No.8"),
    ]
)
def test_constructor_error(n_clusters, lam, model, No, create_err_msg):
    err_msg = create_err_msg.get(No)
    if (No == "4"):
        with pytest.raises(ValueError) as e:
            concl = ConsensusClustering(n_clusters=n_clusters)
    elif(No == "5"):
        with pytest.raises(TypeError) as e:
            concl = ConsensusClustering(n_clusters=n_clusters)
    elif(No == "6"):
        with pytest.raises(TypeError) as e:
            concl = ConsensusClustering(n_clusters=n_clusters, lam=lam)
    elif(No == "7"):
        with pytest.raises(ValueError) as e:
            concl = ConsensusClustering(n_clusters=n_clusters, model=model)
    elif(No == "8"):
        with pytest.raises(TypeError) as e:
            concl = ConsensusClustering(n_clusters=n_clusters, model=model)

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["e", "No"],
    [
        pytest.param("T", "9",  id="No.9"),
        pytest.param("T", "10", id="No.10"),
        pytest.param("T", "11", id="No.11"),
        pytest.param("V", "12", id="No.12"),
        pytest.param("V", "13", id="No.13"),
        pytest.param("V", "14", id="No.14"),
        pytest.param("V", "15", id="No.15"),
        pytest.param("V", "16", id="No.16"),
    ]
)
def test_fit_predict_X_error(create_fit_data, e, No, create_err_msg):
    indata = create_fit_data.get(No)
    concl = ConsensusClustering(n_clusters=3)
    err_msg = create_err_msg.get(No)
    if (e == "T"):
        with pytest.raises(TypeError) as e:
            concl.fit_predict(indata, None)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            concl.fit_predict(indata, None)

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.17"),
    ],
)
@pytest.mark.neal
def test_fit_predict__error_neal(lack_package):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=2)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)

    scl = KMeans(n_clusters=2)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    result = np.array(result)

    concl = ConsensusClustering(n_clusters=2,
                                lam=10.5, model="partition_difference")

    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        con_label = concl.fit_predict(result, None)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.18"),
    ],
)
@pytest.mark.dimod
def test_fit_predict__error_dimod(lack_package):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=2)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)

    scl = KMeans(n_clusters=2)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    result = np.array(result)

    concl = ConsensusClustering(n_clusters=2,
                                lam=10.5, model="partition_difference")

    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        con_label = concl.fit_predict(result, None)

@pytest.mark.parametrize(
    ["Case"],
    [
        pytest.param("InvalidToken", id="No.19"),
    ]
)
def test_fit_predict_client_error(Case):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=2)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)

    scl = KMeans(n_clusters=2)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    result = np.array(result)

    concl = ConsensusClustering(n_clusters=2,
                                lam=10.5, model="partition_difference")
    if (Case == "InvalidToken"):
        fclient.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            con_label = concl.fit_predict(result, fclient)
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["model", "X", "client"],
    [
        pytest.param("pairwise_similarity-based", "L", "F", id="additional"),
        pytest.param("pairwise_similarity-based", "L", "N", id="additional"),
        pytest.param("pairwise_similarity-based", "N", "F", id="additional"),
        pytest.param("pairwise_similarity-based", "N", "N", id="additional"),
        pytest.param("partition_difference", "L", "F", id="additional"),
        pytest.param("partition_difference", "L", "N", id="additional"),
        pytest.param("partition_difference", "N", "F", id="additional"),
        pytest.param("partition_difference", "N", "N", id="additional"),
    ],
)
def test_fit_predict_additional(model, X, client):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=2)
    fclient = setup_fixstars()
    ccl_label = ccl.fit_predict(data, fclient)

    scl = KMeans(n_clusters=2)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    if (X == "N"):
        result = np.array(result)

    concl = ConsensusClustering(n_clusters=2, lam=10, model=model)

    if (client == "F"):
        con_label = concl.fit_predict(result, fclient)
    elif(client == "N"):
        con_label = concl.fit_predict(result, None)
    else:
        raise Exception("Invalid Client")

    assert(isinstance(con_label, list))
    assert(len(con_label) == 10)
    assert(len(set(con_label)) == 2)


@pytest.mark.parametrize(
    ["No", "lam", "model", "labels_dummy"],
    [
        pytest.param("20", None, "pairwise_similarity-based", [0, 1, 2, 0, 1, 2, 0, 1, 2, -1], id="No.20"),
        pytest.param("21", None, "partition_difference", [0, 1, 2, 0, 1, 2, 0, 1, 2, -1], id="No.21"),
        pytest.param("22", 10, "pairwise_similarity-based", [0, 0, 1, 1, 2, 2, 3, 3, 3, 4], id="No.22"),
        pytest.param("23", 10, "partition_difference", [0, 0, 1, 1, 2, 2, 3, 3, 3, 4], id="No.23"),
        pytest.param("24", 10.5, "pairwise_similarity-based", [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], id="No.24"),
        pytest.param("25", 10.5, "partition_difference", [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], id="No.25"),
    ]
)
def test_fit_predict_constraint_violation(mocker, No, lam, model, labels_dummy, create_err_msg):
    n_clusters = 3
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                            cluster_std=1.5, centers=2)

    ccl = CombinatorialClustering(n_clusters=n_clusters)
    client = setup_fixstars()
    ccl_label = ccl.fit_predict(data, client)

    scl = KMeans(n_clusters=n_clusters)
    scl_label = scl.fit_predict(data)

    result = [ccl_label, scl_label]
    result = np.array(result)

    if (lam is None and model is None):
        concl = ConsensusClustering(n_clusters=n_clusters)
    else:
        concl = ConsensusClustering(n_clusters=n_clusters,
                                    lam=lam, model=model)

    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        con_label = concl.fit_predict(result, client)
    
    err_msg = create_err_msg.get(No)





