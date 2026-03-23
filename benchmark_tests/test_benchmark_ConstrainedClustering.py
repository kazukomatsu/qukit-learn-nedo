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
from qklearn.cluster import ConstrainedClustering
from qklearn.utils import read_token, cost
from amplify import FixstarsClient
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import csv, os
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
        nos=[
             6, 7, 8, 9,11,
            15,17,18,19,44,
            21,22,59,29,31,
            32,33,34,39,43,
            20,46,47,48,49,
            50,56,57,67,69,
             6, 7, 8, 9,11,
            21,22,59,29,31,
            32,33,34,39,43,
        ]
    )
)
def test_fit_predict_benchmark(request, no, battery_name, dataset_name, label_num):
    request.node.no = no
    request.node.name = battery_name + "." + dataset_name

    client = setup_fixstars()

    X, labels_tmp = setup_benchmark_datasets(battery_name, dataset_name, label_num)

    labels_true = [int(l - 1) if no != "28" else int(l) for l in labels_tmp]
    
    n_samples = X.shape[0]
    n_clusters = len(set(labels_true))
    
    constaint_name, constaint, constaint_dict = set_constarint_for_datasets(no, labels_true)

    if constaint_name != "balanced-sizes":
        if len(constaint) != 0:
            print(f"constaint = {constaint}")
        else:
            print(f"constaint = {constaint_dict}")
    request.node.constraint = constaint_name

    constcl = ConstrainedClustering(n_clusters=n_clusters)
    try:
        constcl.fit(X)
        
        # Set the specified constaints.
        if constaint_name == "must-link":
            constcl.add_must_link_to_qubo(constaint)
        elif constaint_name == "cannot-link":
            constcl.add_cannot_link_to_qubo(constaint)
        elif constaint_name == "partition-level":
            constcl.add_partition_level_to_qubo(constaint_dict)
        elif constaint_name == "non-partition-level":
            constcl.add_non_partition_level_to_qubo(constaint_dict)
        elif constaint_name == "limited-sizes":
            constcl.add_limited_sizes_to_qubo(constaint_dict)
        elif constaint_name == "balanced-sizes":
            constcl.add_balanced_sizes_to_qubo()
        elif constaint_name == "set-must-link":
            constcl.set_must_link_by_qbits_reduction(constaint)
        elif constaint_name == "set-partition-level":
            constcl.set_partition_level_by_qbits_reduction(constaint_dict)
        elif constaint_name == "set-non-partition-level":
            constcl.set_non_partition_level_by_qbits_reduction(constaint_dict)
        else:
            pytest.fail("Unexpected constarint name.")

        labels_constcl = constcl.predict(client)
    except Exception as e:
        request.node.const_ANI = np.nan
        request.node.const_NMI = np.nan
        request.node.const_cost = np.nan
        raise e

    assert(isinstance(labels_constcl, list))
    assert(len(labels_constcl) == n_samples)
    assert(len(set(labels_constcl)) == n_clusters)

    result_constcl = calc_eval_indices(X, labels_true, labels_constcl)
    print(result_constcl)

    # Aggregate the result data to output a test summary.
    request.node.const_ANI = result_constcl[0]
    request.node.const_NMI = result_constcl[1]
    request.node.const_cost = result_constcl[2]

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


def set_constarint_for_datasets(no, labels):
    constraint_name_map = {
        "must-link": range(1, 6),
        "cannot-link": range(6, 11),
        "partition-level": list(range(11, 16)),
        "non-partition-level": list(range(16, 21)),
        "limited-sizes": range(21, 26),
        "balanced-sizes": range(26, 31),
        "set-must-link": range(31, 36),
        "set-partition-level": list(range(36, 41)),
        "set-non-partition-level": list(range(41, 46)),
    }
    constraint_name_dict = {
        str(i): name
        for name, nums in constraint_name_map.items()
        for i in nums
    }
    
    constraint = []
    constraint_dict = {}
    
    try:
        constraint_name = constraint_name_dict[no]
    except KeyError as e:
        raise KeyError("Unexpected Test No.") from e

    if "must-link" in constraint_name:
        # Note: This assumes that all samples from index 0 to (span - 1) have the same correct label.
        if no == "3" or no == "33":
            span = 200
        else:
            span = 100
        n_samples_limited = len(labels[:span])
        for i in range(n_samples_limited):
            constraint.append((i % n_samples_limited, (i+1) % n_samples_limited))
    
    elif constraint_name == "cannot-link":
        # Note: This assumes that the samples from index 0 to (span - 1) and the samples from index span to (2*span - 1) belong to different clusters.
        if no == "6":
            span = 200
        elif no == "10":
            span = 500
        else:
            span = 40
        for i in range(len(labels[:span])):
            constraint.append((i, (i+span)))
        
    elif constraint_name == "non-partition-level" or constraint_name == "set-non-partition-level":
        indices = tuple(idx for idx, val in enumerate(labels) if val != labels[0])
        target_cluster_num = labels[0]
        constraint_dict[target_cluster_num] = tuple(indices[:100])

    elif constraint_name == "partition-level" or constraint_name == "set-partition-level":
        indices = tuple(idx for idx, val in enumerate(labels) if val == labels[0])
        constraint_dict[int(labels[0])] = tuple(indices[:100])

    elif constraint_name == "limited-sizes":
        for n in range(len(set(labels))):
            constraint_dict[n] = (len(labels) // len(set(labels))) + 1

    return constraint_name, constraint, constraint_dict