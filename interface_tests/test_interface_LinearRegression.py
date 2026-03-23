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
from qklearn.linear_model import LinearRegression
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_regression
import numpy as np

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.mark.parametrize(
    ["X", "Y", "Client"],
    [
        pytest.param("N", "N", "N", id="No.1"),
        pytest.param("N", "L", "N", id="No.2"),
        pytest.param("L", "N", "F", id="No.3"),
        pytest.param("L", "L", "F", id="No.4"),
    ],
)
def test_fit_normal(X, Y, Client):
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    if (X == "L"):
        x = x.tolist()
    if (Y == "L"):
        y = y.tolist()

    if (Client == "N"):
        client = None
    elif (Client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")

    lr.fit(x, y, client)

@pytest.fixture(scope="module")
def test_fit_data():
    test_data_X = {}
    test_data_Y = {}

    valid_X = [[i ,i] for i in range(30)]
    invalid_X = [[i ,i] for i in range(30)]
    invalid_X[29][0] = "X"
    valid_Y = [i for i in range(30)]
    invalid_Y = [i for i in range(30)]
    invalid_Y[29] = "Y"

    test_data_X["5"] = [[1,2,3], [1,2]]
    test_data_Y["5"] = [1, 2]
    test_data_X["6"] = np.arange(8).reshape((2, 2, 2))
    test_data_Y["6"] = [1, 2]
    test_data_X["7"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data_Y["7"] = [1, 2]
    test_data_X["8"] = np.arange(8).reshape((8,))
    test_data_Y["8"] = [1, 2]
    test_data_X["9"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data_Y["9"] = [1, 2]
    test_data_X["10"] = {0: (1, 2)}
    test_data_Y["10"] = [1, 2]
    test_data_X["11"] = invalid_X
    test_data_Y["11"] = valid_Y
    test_data_X["12"] = [[1,2], [1,2]]
    test_data_Y["12"] = [1, [1, 2]]
    test_data_X["13"] = [[1,2], [1,2]]
    test_data_Y["13"] = np.arange(4).reshape((2, 2))
    test_data_X["14"] = [[1,2], [1,2]]
    test_data_Y["14"] = [[1,2], [1,2]]
    test_data_X["15"] = [[1,2], [1,2]]
    test_data_Y["15"] = {0: 1}
    test_data_X["16"] = valid_X
    test_data_Y["16"] = invalid_Y
    test_data_X["17"] = valid_X
    test_data_Y["17"] = valid_Y[:15]

    test_data_X["30"] = [[1,2,3], [1,2]]
    test_data_X["31"] = np.arange(8).reshape((2, 2, 2))
    test_data_X["32"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data_X["33"] = np.arange(8).reshape((8,))
    test_data_X["34"] = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data_X["35"] = {0: (1, 2)}
    test_data_X["36"] = invalid_X

    return [test_data_X, test_data_Y]

@pytest.mark.parametrize(
    ["No", "e", "err_msg"],
    [
        pytest.param("5",  "V", "has invalid shape", id="No.5"),
        pytest.param("6",  "V", "must be 2-d array", id="No.6"),
        pytest.param("7",  "V", "must be 2-d array", id="No.7"),
        pytest.param("8",  "V", "must be 2-d array", id="No.8"),
        pytest.param("9",  "V", "must be 2-d array", id="No.9"),
        pytest.param("10", "T", "must be array-like object", id="No.10"),
        pytest.param("11", "T", "has invalid data type", id="No.11"),
        pytest.param("12", "V", "has invalid shape", id="No.12"),
        pytest.param("13", "V", "must be 1-d array", id="No.13"),
        pytest.param("14", "V", "must be 1-d array", id="No.14"),
        pytest.param("15", "T", "must be array-like object", id="No.15"),
        pytest.param("16", "T", "has invalid data type", id="No.16"),
        pytest.param("17", "I", "The lengths of 'X' and 'y' do not match", id="No.17"),
    ]
)
def test_fit_error(test_fit_data, No, e, err_msg):
    lr = LinearRegression()
    x = test_fit_data[0].get(No)
    y = test_fit_data[1].get(No)
    client = None

    if (e == "V"):
        with pytest.raises(ValueError) as e:
            lr.fit(x, y, client)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            lr.fit(x, y, client)
    elif (e == "I"):
        with pytest.raises(IndexError) as e:
            lr.fit(x, y, client)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["num_element", "e", "err_msg"],
    [
        pytest.param(1,   "N", "", id="No.18"),
        pytest.param(0,   "V", "num_elements must be greater than or equal to 1", id="No.19"),
        pytest.param(1.5, "T", "num_elements must be int", id="No.20"),
    ],
)
def test_fit_num_element(num_element, e, err_msg):
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    client = setup_fixstars()
    if (e == "N"):
        lr.fit(x, y, client, num_elements=num_element)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            lr.fit(x, y, client, num_elements=num_element)
            assert(err_msg in str(e.value))
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            lr.fit(x, y, client, num_elements=num_element)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["basis", "e", "err_msg"],
    [
        pytest.param(4,   "N", "", id="No.21"),
        pytest.param(4.5, "T", "basis must be int", id="No.22"),
    ],
)
def test_fit_basis(basis, e, err_msg):
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    client = setup_fixstars()
    if (e == "N"):
        lr.fit(x, y, client, basis=basis)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            lr.fit(x, y, client, basis=basis)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["exponent_offset", "e", "err_msg"],
    [
        pytest.param(-1,  "N", "", id="No.23"),
        pytest.param(1.5, "T", "exponent_offset must be int", id="No.24"),
    ]
)
def test_fit_exponent_offset(exponent_offset, e, err_msg):
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    client = setup_fixstars()
    if (e == "N"):
        lr.fit(x, y, client, exponent_offset=exponent_offset)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            lr.fit(x, y, client, exponent_offset=exponent_offset)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.25"),
    ]
)
@pytest.mark.neal
def test_fit_error_neal(lack_package):
    client = None
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        lr.fit(x, y, client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.26"),
    ]
)
@pytest.mark.dimod
def test_fit_error_dimod(lack_package):
    client = None
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        lr.fit(x, y, client)

@pytest.mark.parametrize(
    ["Case", "err_msg"],
    [
        pytest.param("InvalidToken", "exceptions were raised from solve() of the Fixstars Amplify SDK", id="No.27"),
    ],
)
def test_fit_error_client(Case, err_msg):
    client = setup_fixstars()
    if (Case == "InvalidToken"):
        lr = LinearRegression()
        x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                               bias=0.0, noise=20.0)
        client.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            lr.fit(x, y, client)
            assert(err_msg in str(e.value))
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["X"],
    [
        pytest.param("N", id="No.28"),
        pytest.param("L", id="No.29"),
    ]
)
def test_predict_normal(X):
    client = None
    lr = LinearRegression()
    x, y = make_regression(n_samples=90, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    x_t = x[:60]
    y_t = y[:60]
    x_p = x[-30:]

    lr.fit(x_t, y_t, client)

    if (X == "N"):
        y_p = lr.predict(x_p)
    elif (X == "L"):
        y_p = lr.predict(x_p.tolist())
    else:
        raise Exception("Invalid X")

    assert(len(y_p) == len(x_p))
    assert(type(y_p) is np.ndarray)
    assert(y_p.shape == (len(x_p), ))

@pytest.mark.parametrize(
    ["No", "e", "err_msg"],
    [
        pytest.param("30",  "V", "has invalid shape", id="No.30"),
        pytest.param("31",  "V", "must be 2-d array", id="No.31"),
        pytest.param("32",  "V", "must be 2-d array", id="No.32"),
        pytest.param("33",  "V", "must be 2-d array", id="No.33"),
        pytest.param("34",  "V", "must be 2-d array", id="No.34"),
        pytest.param("35",  "T", "must be array-like object", id="No.35"),
        pytest.param("36",  "T", "has invalid data type", id="No.36"),
        pytest.param("37",  "A", "This instance is not fitted yet", id="No.37"),
    ]
)
def test_predict_error(test_fit_data, No, e, err_msg):
    lr = LinearRegression()
    x_t, y_t = make_regression(n_samples=30, n_features=2, n_informative=1,
                               bias=0.0, noise=20.0)
    x_p = test_fit_data[0].get(No)
    client = None

    if (e == "V"):
        lr.fit(x_t, y_t, client)
        with pytest.raises(ValueError) as e:
            y_p = lr.predict(x_p)
    elif (e == "T"):
        lr.fit(x_t, y_t, client)
        with pytest.raises(TypeError) as e:
            y_p = lr.predict(x_p)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            y_p = lr.predict(x_p)
    else:
        raise Exception("Invalid Error")

@pytest.mark.parametrize(
    ["X", "Y", "Client"],
    [
        pytest.param("N", "N", "N", id="No.additional1"),
        pytest.param("N", "L", "N", id="No.additional2"),
        pytest.param("L", "N", "F", id="No.additional3"),
        pytest.param("L", "L", "F", id="No.additional4"),
    ],
)
def test_fit_predict_normal(X, Y, Client):
    lr = LinearRegression()
    x, y = make_regression(n_samples=30, n_features=2, n_informative=1,
                           bias=0.0, noise=20.0)
    if (X == "L"):
        x = x.tolist()
    if (Y == "L"):
        y = y.tolist()

    if (Client == "N"):
        client = None
    elif (Client == "F"):
        client = setup_fixstars()
    else:
        raise Exception("Invalid Client")

    lr.fit(x, y, client, num_elements=5, basis=4, exponent_offset=-1)

    y_p = lr.predict(x)
    assert(len(y_p) == len(x))
    assert(type(y_p) is np.ndarray)
    assert(y_p.shape == (len(x), ))
