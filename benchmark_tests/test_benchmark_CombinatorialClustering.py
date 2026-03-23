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
from qklearn.utils import read_token, cost
from amplify import FixstarsClient
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import csv, os, random
import numpy as np
from collections import defaultdict
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
    load_dataset_list()
)
def test_fit_predict_benchmark(request, no, battery_name, dataset_name, label_num):
    request.node.no = no
    request.node.name = battery_name + "." + dataset_name

    client = setup_fixstars()

    X_tmp, labels_true_tmp = setup_benchmark_datasets(battery_name, dataset_name, label_num)
    
    # Adjust the dataset based on the given test number.
    X, labels_true = adjust_benchmark_datasets(no, X_tmp, labels_true_tmp)
    n_samples = X.shape[0]
    n_clusters = len(set(labels_true))

    ccl = CombinatorialClustering(n_clusters=n_clusters)
    try:
        labels_ccl = ccl.fit_predict(X, client)
    except Exception as e:
        request.node.qklearn_ANI = np.nan
        request.node.qklearn_NMI = np.nan
        request.node.qklearn_cost = np.nan
        raise e
    
    assert(isinstance(labels_ccl, list))
    assert(len(labels_ccl) == n_samples)
    assert(len(set(labels_ccl)) == n_clusters)

    result_ccl = calc_eval_indices(X, labels_true, labels_ccl)
    print(result_ccl)

    # Aggregate the result data to output a test summary.
    request.node.qklearn_ANI = result_ccl[0]
    request.node.qklearn_NMI = result_ccl[1]
    request.node.qklearn_cost = result_ccl[2]

    # Run sklearn using the same benchmark dataset.
    kmeans = KMeans(n_clusters=n_clusters)
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


def adjust_benchmark_datasets(target_no, X, labels_true):
    # Setting the extraction count for each target_no
    special_slices = {
        "4": 8340,
        "27": (500, 50),
        "28": (500, 50),
        "40": (3120, 8),
        "41": (700, 35),
    }
    cluster_sample_nums = {
        "13": 1562, "23": 62, "24": 20, "25": 10, "30": 26,
        "35": 111, "36": 111, "37": 111, "38": 111, "40": 390, "41": 20, "42": 40,
        "62": 510, "63": 510, "64": 250, "65": 308
    }

    if target_no == "4":
        n = special_slices[target_no]
        ret_X = X[:n]
        ret_labels_true = labels_true[:n]
    elif target_no in ["27", "28", "40", "41"]:
        num_out, num_classes = special_slices[target_no]
        ret_X, ret_labels_true = select_balanced_data(X, labels_true, num_out, num_classes)
    elif target_no in cluster_sample_nums:
        n_clusters = len(set(labels_true))
        num = cluster_sample_nums[target_no]
        ret_X = select_per_cluster(X, n_clusters, num)
        ret_labels_true = select_per_cluster(labels_true, n_clusters, num)
    else:
        ret_X = X
        ret_labels_true = labels_true
    
    return ret_X, ret_labels_true

def select_per_cluster(data, n_clusters, num):
    # Note: This method assumes that the dataset being used has an equal number of samples in each cluster.
    cluster_size = len(data) // n_clusters
    result = []
    for i in range(n_clusters):
        start = cluster_size * i
        end = start + num
        result.extend(data[start:end])
    return np.array(result)

def select_balanced_data(X, label, num_out, num_classes):
    # Index extraction for each class
    class_indices = defaultdict(list)
    for idx, val in enumerate(label):
        class_indices[val].append(idx)

    # Extract so that each class is balanced.
    selected_indices = []
    for class_val in range(1, num_classes + 1):
        candidates = class_indices[class_val]
        pick_num = min((num_out//num_classes), len(candidates))
        selected_indices.extend(random.sample(candidates, pick_num))

    num_selected = len(selected_indices)  # the number of indices already selected
    
    # If additional indexes are needed, distribute the rest randomly.
    if num_selected < num_out:
        additional = num_out - num_selected
        additional_candidates = []
        for class_val in range(1, num_classes + 1):
            used = set(selected_indices)  # already used indexes
            remain = [idx for idx in class_indices[class_val] if idx not in used]
            additional_candidates.extend(remain)
        
        # Extract missing parts
        selected_indices.extend(random.sample(additional_candidates, additional))
    
    new_label = [label[i] for i in selected_indices]
    new_X = X[selected_indices]
    return new_X, new_label