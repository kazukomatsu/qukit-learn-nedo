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
from qklearn.utils import read_token, cost
from amplify import FixstarsClient
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import csv,os
import numpy as np
import clustbench

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

def load_dataset_list(nos=None, labels_specified=None):
    # nos means the number of the benchmark dataset to be used.
    # labels_specified means the  label number of the benchmark dataset to be used.
    
    benchmark_tests_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(benchmark_tests_dir, 'benchmarks.csv')

    with open(path) as f:
        reader = csv.DictReader(f)
        items = [(int(row["no"]), row["battery"], row["dataset"]) for row in reader]
    if nos is not None:
        items = [items[no-1] for no in nos]
    
    params = []
    for i, item in enumerate(items):
        params.append(
            pytest.param(
                str(i+1),
                item[1],
                item[2],
                0 if labels_specified is None else labels_specified[i],
                id=f"No.{i+1}"
            )
        )
        print(f"No.{i+1}: {params[i]}")
    return params

def setup_benchmark_datasets(battery_name, dataset_name, label_num):
    data_url = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"
    b = clustbench.load_dataset(battery_name, dataset_name, url=data_url)
    
    return b.data, b.labels[label_num]

def calc_eval_indices(X, labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    c = cost(X, labels_pred)
    return ari, nmi, c

@pytest.mark.parametrize(
    ["no", "battery_name", "dataset_name", "label_num"],
    load_dataset_list(
        nos=[7, 12, 15, 31, 32, 43, 44, 45, 48, 50, 51, 52, 53, 53, 53, 54, 55, 56, 57, 58, 69],
        labels_specified=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 3, 4, 0, 0, 0, 0, 1, 0]
    )
)
def test_fit_predict_benchmark(request, no, battery_name, dataset_name, label_num):
    request.node.no = no
    request.node.name = battery_name + "." + dataset_name
    
    client = setup_fixstars()
    
    X, labels_true = setup_benchmark_datasets(battery_name, dataset_name, label_num)
    n_samples = X.shape[0]

    bcl = BinaryClustering()
    try:
        labels_bcl = bcl.fit_predict(X, client)
    except Exception as e:
        request.node.qklearn_ANI = np.nan
        request.node.qklearn_NMI = np.nan
        request.node.qklearn_cost = np.nan
        raise e
    
    assert(isinstance(labels_bcl, list))
    assert(len(labels_bcl) == n_samples)
    assert(len(set(labels_bcl)) == 2)

    result_bcl = calc_eval_indices(X, labels_true, labels_bcl)
    print(result_bcl)

    # Aggregate the result data to output a test summary.
    request.node.qklearn_ANI = result_bcl[0]
    request.node.qklearn_NMI = result_bcl[1]
    request.node.qklearn_cost = result_bcl[2]

    # Run sklearn using the same benchmark dataset.
    kmeans = KMeans(n_clusters=2)
    try:
        labels_kmeans = kmeans.fit_predict(X)
    except Exception as e:
        request.node.sklearn_ANI = np.nan
        request.node.sklearn_NMI = np.nan
        request.node.sklearn_cost = np.nan
        raise e

    result_kmeans = calc_eval_indices(X, labels_true, labels_kmeans)
    print(result_kmeans)
    
    # Aggregate the result data to output a test summary.
    request.node.sklearn_ANI = result_kmeans[0]
    request.node.sklearn_NMI = result_kmeans[1]
    request.node.sklearn_cost = result_kmeans[2]