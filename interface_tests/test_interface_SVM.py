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
import sys
import os
from qklearn.svm import SVC
from qklearn.utils import read_token
from amplify import FixstarsClient
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.fixture(scope="module")
def create_fit_data():
    # {"case No." : input}
    test_data = {}
    
    test_data["1"] = {
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "t": [1, -1]
    }
    test_data["2"] = {
        "X": np.array([[1, 2], [3, 4]]),
        "t": [1, -1]
    }
    test_data["3"] = {
        "X": ([1.0, 2.0], [3.0, 4.0]),
        "t": [1, -1]
    }
    test_data["4"] = {
        "X": [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]],
        "t": [1, 1, -1]
    }
    test_data["5"] = {
        "X": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        "t": [1, 1, -1, -1]
    }
    test_data["6"] = {
        "X": [[], []],
        "t": [1, -1]
    }
    test_data["7"] = {
        "X": [[1.0, "2.0"], [3.0, 4.0]],
        "t": [1, -1]
    }
    test_data["8"] = {
        "X": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        "t": [1.0, 1.0, 1.0, -1.0, -1.0]
    }
    test_data["9"] = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        "t": np.array([1, 1, 1, -1, -1])
    }
    test_data["10"] = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        "t": {0:1, 1:1, 2:1, 3:-1, 4:-1}
    }
    test_data["11"] = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "t": [[1, 1], [1, -1], [-1, -1]]
    }
    test_data["12"] = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "t": []
    }
    test_data["13"] = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        "t": [1, "1", 1, -1, -1]
    }
    test_data["14"] = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "t": np.array([1, 1, -1, -1])
    }
    test_data["15"] = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "t": np.array([1, 1, -1, -1]),
        "client": "Fixstars"
    }
    test_data["16"] = {
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "t": [1, -1],
        "num": 1.0
    }
    test_data["17"] = {"num": "1"}
    test_data["18"] = {"num": 0}
    test_data["19"] = {
        "X": [[3.0, 0.0], [2.5, 1.5]],
        "t": [1, -1],
        "num": 1
    }
    test_data["20"] = {"num": 2}
    test_data["21"] = {"basis": 2.0}
    test_data["22"] = {"basis": 2.0}
    test_data["23"] = {"basis": 1}
    test_data["24"] = {"basis": 2}
    test_data["25"] = {"basis": 3}
    test_data["26"] = {"exp": 0.0}
    test_data["27"] = {"exp": "0"}
    test_data["28"] = {"exp": 1}
    test_data["29"] = {"xi": "1"}
    test_data["30"] = {"xi": -1}
    test_data["31"] = {"xi": -0.1}
    test_data["32"] = {"xi": 0}
    test_data["33"] = {"xi": 0.0}
    test_data["34"] = {"xi": 2}
    test_data["35"] = {"xi": 0.1}
    test_data["36"] = {
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "t": [1, 1, -1]
    }
    test_data["40"] = {
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "t": [1, 1]
    }
    test_data["41"] = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "t": [1, 2, -1]
    }

    return test_data

@pytest.fixture(scope="module")
def create_predict_data():
    # {"case No." : input}
    test_data = {}
    
    test_data["1"] = [[1.0, 2.0], [3.0, 4.0]]
    test_data["2"] = [[1.0, 2.0], [3.0, 4.0]]
    test_data["3"] = np.array([[1, 2], [3, 4]])
    test_data["4"] = ([1.0, 2.0], [3.0, 4.0])
    test_data["5"] = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]]
    test_data["6"] = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    test_data["7"] = [[], []]
    test_data["8"] = [[1.0, "2.0"], [3.0, 4.0]]
    test_data["9"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    return test_data

@pytest.mark.parametrize(
    ["No"],
    [
        pytest.param("1", id="No.1"),
        pytest.param("2", id="No.2"),
        pytest.param("8", id="No.8"),
        pytest.param("9", id="No.9"),
        pytest.param("14", id="No.14"),
        pytest.param("15", id="No.15"),
        pytest.param("19", id="No.19"),
        pytest.param("20", id="No.20"),
        pytest.param("24", id="No.24"),
        pytest.param("25", id="No.25"),
        pytest.param("28", id="No.28"),
        pytest.param("32", id="No.32"),
        pytest.param("33", id="No.33"),
        pytest.param("34", id="No.34"),
        pytest.param("35", id="No.35"),
    ],
)
def test_fit_normal(create_fit_data, No):
    svc = SVC()
    input = create_fit_data.get(No)
    
    if "client" in input:
        cl = setup_fixstars()
    else:
        cl = None
    
    try:
        svc.fit(X=input.get("X", [[1.0, 2.0], [3.0, 4.0]]),
                t=input.get("t", [1, -1]),
                client=cl,
                num_elements=input.get("num", 3),
                basis=input.get("basis", 2),
                exponent_offset=input.get("exp", 0),
                xi=input.get("xi", 1))
    except RuntimeError as e:
        if No == "25":
            print("An expected error occurred in Test No.25.")
            msg = "The slope of the decision boundary obtained by the optimization solver is an invalid value."
            assert msg in str(e)
            return
        else:
            raise Exception(f"Test No.{No} raised an unexpected error.")
    
    X = [[5.0, 2.0], [3.0, 7.0]]
    label = svc.predict(X)
    assert isinstance(label, np.ndarray) is True
    assert len(label) == len(X)
    for l in label:
        assert l == 1 or l == -1

@pytest.mark.parametrize(
    ["No", "e"],
    [
        pytest.param("3", "T", id="No.3"),
        pytest.param("4", "V", id="No.4"),
        pytest.param("5", "V", id="No.5"),
        pytest.param("6", "V", id="No.6"),
        pytest.param("7", "T", id="No.7"),
        pytest.param("10", "T", id="No.10"),
        pytest.param("11", "V", id="No.11"),
        pytest.param("12", "V", id="No.12"),
        pytest.param("13", "T", id="No.13"),
        pytest.param("16", "T", id="No.16"),
        pytest.param("17", "T", id="No.17"),
        pytest.param("18", "V", id="No.18"),
        pytest.param("21", "T", id="No.21"),
        pytest.param("22", "T", id="No.22"),
        pytest.param("23", "V", id="No.23"),
        pytest.param("26", "T", id="No.26"),
        pytest.param("27", "T", id="No.27"),
        pytest.param("29", "T", id="No.29"),
        pytest.param("30", "V", id="No.30"),
        pytest.param("31", "V", id="No.31"),
        pytest.param("36", "I", id="No.36"),
        pytest.param("40", "V", id="No.40"),
        pytest.param("41", "V", id="No.41"),
    ],
)
def test_fit_error(create_fit_data, No, e):
    svc = SVC()
    input = create_fit_data.get(No)
    if (e == "T"):
        with pytest.raises(TypeError) as e:
            svc.fit(X=input.get("X", [[1.0, 2.0], [3.0, 4.0]]),
                    t=input.get("t", [1, 1]),
                    client=None,
                    num_elements=input.get("num", 3),
                    basis=input.get("basis", 2),
                    exponent_offset=input.get("exp", 0),
                    xi=input.get("xi", 1))
        print(str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            svc.fit(X=input.get("X", [[1.0, 2.0], [3.0, 4.0]]),
                    t=input.get("t", [1, 1]),
                    client=None,
                    num_elements=input.get("num", 3),
                    basis=input.get("basis", 2),
                    exponent_offset=input.get("exp", 0),
                    xi=input.get("xi", 1))
        print(str(e.value))
    elif (e == "I"):
        with pytest.raises(IndexError) as e:
            svc.fit(X=input.get("X", [[1.0, 2.0], [3.0, 4.0]]),
                    t=input.get("t", [1, 1]),
                    client=None,
                    num_elements=input.get("num", 3),
                    basis=input.get("basis", 2),
                    exponent_offset=input.get("exp", 0),
                    xi=input.get("xi", 1))
        print(str(e.value))
    else:
        raise Exception("Invalid Error Type")

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.37"),
    ],
)
@pytest.mark.neal
def test_fit_error_neal(lack_package):
    svc = SVC()
    X = [[1.0, 2.0], [3.0, 4.0]]
    t = [1, -1]
    client = None
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        svc.fit(X, t, client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.38"),
    ],
)
@pytest.mark.dimod
def test_fit_predict_error_neal(lack_package):
    svc = SVC()
    X = [[1.0, 2.0], [3.0, 4.0]]
    t = [1, -1]
    client = None
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        svc.fit(X, t, client)

@pytest.mark.parametrize(
    ["Case"],
    [
        pytest.param("Invalid_Token", id="No.39"),
    ],
)
def test_fit_predict_error_fixstars(Case):
    svc = SVC()
    X = [[1.0, 2.0], [3.0, 4.0]]
    t = [1, -1]
    client = setup_fixstars()
    client.token = "token"
    
    with pytest.raises(RuntimeError) as e:
        svc.fit(X, t, client)

@pytest.mark.parametrize(
    ["No"],
    [
        pytest.param("2", id="No.2"),
        pytest.param("3", id="No.3"),
    ],
)
def test_predict_normal(create_predict_data, No):
    svc = SVC()
    X = [[1.0, 2.0], [3.0, 4.0]]
    t = [1, -1]
    svc.fit(X, t, client=None)

    input = create_predict_data.get(No)
    label = svc.predict(input)
    assert isinstance(label, np.ndarray) is True
    assert len(label) == len(X)
    for l in label:
        assert l == 1 or l == -1

@pytest.mark.parametrize(
    ["No", "e"],
    [
        pytest.param("1", "A", id="No.1"),
        pytest.param("4", "T", id="No.4"),
        pytest.param("5", "V", id="No.5"),
        pytest.param("6", "V", id="No.6"),
        pytest.param("7", "V", id="No.7"),
        pytest.param("8", "T", id="No.8"),
        pytest.param("9", "V", id="No.9"),
    ],
)
def test_predict_error(create_predict_data, No, e):
    svc = SVC()
    X = [[1.0, 2.0], [3.0, 4.0]]
    t = [1, -1]
    if No != "1":
        svc.fit(X, t, client=None)
    input = create_predict_data.get(No)

    if (e == "T"):
        with pytest.raises(TypeError) as e:
            label = svc.predict(input)
        print(str(e.value))
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            label = svc.predict(input)
        print(str(e.value))
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            label = svc.predict(input)
        print(str(e.value))
    else:
        raise Exception("Invalid Error Type")


@pytest.mark.parametrize(
    ["X", "t", "Client"],
    [
        pytest.param("N", "N", "N", id="additional"),
        pytest.param("N", "L", "N", id="additional"),
        pytest.param("L", "N", "F", id="additional"),
        pytest.param("L", "L", "F", id="additional"),
    ],
)
def test_predict_additional(X, t, Client):
    svc = SVC()
    inputs,targets=make_blobs(n_samples=100, centers=2,
                              random_state=0, cluster_std=0.7)
    targets[targets == 0] = -1
    x_train, x_test, t_train, t_test = train_test_split(inputs,targets)
    if (X == "L"):
        x_train = x_train.tolist()
        x_test = x_test.tolist()
    if (t == "L"):
        t_train = t_train.tolist()
    if (Client == "N"):
        client = None
    elif (Client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")
    svc.fit(x_train, t_train, client, num_elements=4, basis=4, exponent_offset=-1, xi=1.1)
    label = svc.predict(x_test)
    assert isinstance(label, np.ndarray) is True
    assert len(label) == len(x_test)
    for l in label:
        assert l == 1 or l == -1
