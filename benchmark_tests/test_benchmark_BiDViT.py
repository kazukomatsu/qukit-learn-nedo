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
from qklearn.utils import read_token, cost
from amplify import FixstarsClient
from sklearn.cluster import AgglomerativeClustering
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
    load_dataset_list(
        nos=[53, 34, 8, 14, 33, 57, 46, 49, 31, 7, 9, 48, 10, 59, 43, 14],
        labels_specified=[2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    )
)
def test_fit_predict_benchmark(request, no, battery_name, dataset_name, label_num):
    request.node.no = no
    request.node.name = battery_name + "." + dataset_name

    client = setup_fixstars()

    X_tmp, labels_true_tmp = setup_benchmark_datasets(battery_name, dataset_name, label_num)
    
    # Adjust the dataset based on the given test number.
    X, labels_true = adjust_benchmark_datasets(no, X_tmp, labels_true_tmp)
    n_samples = X.shape[0]
    
    b = BiDViT()
    try:
        tree = b.fit_predict(X, client)
    except Exception as e:
        request.node.qklearn_results = [
            (np.nan, np.nan, np.nan, np.nan)
        ]
        request.node.sklearn_results = [
            (np.nan, np.nan, np.nan, np.nan)
        ]
        raise e

    assert(isinstance(tree, dict))
    for k, v in tree.items():
        assert(len(v) == n_samples)
        assert(len(set(v)) == k)
    
    qklearn_results = []
    sklearn_results = []
    for n_clusters, labels_bidvit in tree.items():
        result_bidvit = calc_eval_indices(X, labels_true, labels_bidvit)
        print(result_bidvit)
        
        # Run sklearn using the same benchmark dataset.
        acl = AgglomerativeClustering(n_clusters=n_clusters)
        try:
            labels_acl = acl.fit_predict(X)
        except Exception as e:
            qklearn_results.append(
                (
                    result_bidvit[0],
                    result_bidvit[1],
                    result_bidvit[2],
                    n_clusters
                )
            )
            sklearn_results.append(
                (
                    np.nan,
                    np.nan,
                    np.nan,
                    n_clusters
                )
            )
            request.node.qklearn_results = qklearn_results
            request.node.sklearn_results = sklearn_results
            raise e

        result_acl = calc_eval_indices(X, labels_true, labels_acl)
        print(result_acl)
        
        # Aggregate the results data to output a test summary.
        qklearn_results.append(
            (
                result_bidvit[0],
                result_bidvit[1],
                result_bidvit[2],
                n_clusters
            )
        )
        sklearn_results.append(
            (
                result_acl[0],
                result_acl[1],
                result_acl[2],
                n_clusters
            )
        )
    request.node.qklearn_results = qklearn_results
    request.node.sklearn_results = sklearn_results

def adjust_benchmark_datasets(target_no, X, labels_true):
    # Setting the extraction count for each target_no
    special_slices = {
        "4": (1000, 3),
        "6": (200, 3),
        "11": (1000, 3),
        "13": (200, 2), 
        "15": (200, 3),
        "16": (200, 3)
    }

    if target_no in special_slices:
        num_out, num_classes = special_slices[target_no]
        ret_X, ret_labels_true = select_balanced_data(X, labels_true, num_out, num_classes) 
    elif target_no == "12":
        ret_X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        ret_labels_true = labels_true
    else:
        ret_X = X
        ret_labels_true = labels_true
    
    return ret_X, ret_labels_true

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