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
from qklearn.cluster import ConsensusClustering, CombinatorialClustering, KernelClustering
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
    client.parameters.timeout = timedelta(milliseconds=3500)
    
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
    test_config_dict = {
         "1": ("Combinatorial", 64),
         "2": ("Combinatorial", 64),
         "3": ("Combinatorial", 64),
         "4": ("Combinatorial", 64),
         "5": ("Combinatorial", 64),
         "6": ("Kernel", 4),
         "8": ("Kernel", 64),
         "9": ("Combinatorial", 64),
        "10": ("Combinatorial", 64),
        "11": ("Combinatorial", 64),
        "13": ("Combinatorial", 64),
        "14": ("Combinatorial", 64),
        "16": ("Combinatorial", 64),
        "17": ("Kernel", 16),
        "18": ("Kernel", 64),
        "19": ("Combinatorial", 64),
        "20": ("Kernel", 64),
        "21": ("Combinatorial", 64),
        "22": ("Combinatorial", 64),
        "23": ("Combinatorial", 64),
        "24": ("Combinatorial", 64),
        "25": ("Combinatorial", 64),
        "26": ("Combinatorial", 64),
        "27": ("Combinatorial", 64),
        "28": ("Combinatorial", 64),
        "29": ("Combinatorial", 64),
        "30": ("Combinatorial", 64),
        "33": ("Kernel", 64),
        "34": ("Combinatorial", 64),
        "35": ("Combinatorial", 64),
        "36": ("Combinatorial", 64),
        "37": ("Combinatorial", 64),
        "38": ("Combinatorial", 64),
        "39": ("Kernel", 64),
        "40": ("Combinatorial", 64),
        "41": ("Combinatorial", 64),
        "42": ("Combinatorial", 64),
        "46": ("Combinatorial", 64),
        "47": ("Kernel", 64),
        "48": ("Combinatorial", 64),
        "49": ("Kernel", 64),
        "53": ("Combinatorial", 64),
        "56": ("Kernel", 16),
        "57": ("Combinatorial", 64),
        "58": ("Combinatorial", 64),
        "59": ("Kernel", 16),
        "60": ("Kernel", 64),
        "61": ("Kernel", 64),
        "62": ("Combinatorial", 64),
        "63": ("Combinatorial", 64),
        "64": ("Combinatorial", 64),
        "65": ("Combinatorial", 64),
        "66": ("Combinatorial", 64),
        "67": ("Kernel", 64),
        "68": ("Kernel", 16),
    }
    
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
        ccl.fit(X)
        lam_ccl_default = ccl.lam
        labels_ccl = ccl.predict(client)
        
        if no in test_config_dict.keys():
            if test_config_dict[no][0] == "Combinatorial":
                lam_ccl = lam_ccl_default*test_config_dict[no][1]
                ccl = CombinatorialClustering(n_clusters=n_clusters, lam=lam_ccl)
                labels_the_other = ccl.fit_predict(X, client)
                del ccl
                
                request.node.qklearn_lam_default = lam_ccl_default
                request.node.qklearn_lam_success = lam_ccl
            else:
                kcl_tmp = KernelClustering(n_clusters=n_clusters)
                kcl_tmp.fit(X)
                lam_kcl_default = kcl_tmp.lam
                del kcl_tmp
                
                lam_kcl = lam_kcl_default*test_config_dict[no][1]
                kcl = KernelClustering(n_clusters=n_clusters, lam=lam_kcl)
                labels_the_other = kcl.fit_predict(X, client)
                del kcl
                
                request.node.qklearn_lam_default = lam_kcl_default
                request.node.qklearn_lam_success = lam_kcl
        else:
            kcl = KernelClustering(n_clusters=n_clusters)
            kcl.fit(X)
            lam_kcl_default = kcl.lam
            labels_the_other = kcl.predict(client)
            del kcl
            
            request.node.qklearn_lam_default = lam_kcl_default
            request.node.qklearn_lam_success = lam_kcl_default
    except Exception as e:
        request.node.qklearn_ANI = np.nan
        request.node.qklearn_NMI = np.nan
        request.node.qklearn_cost = np.nan
        raise e
    
    labels_X = [labels_ccl, labels_the_other]

    concl = ConsensusClustering(n_clusters=n_clusters)
    try:
        labels_concl = concl.fit_predict(labels_X, client)
    except Exception as e:
        request.node.qklearn_ANI = np.nan
        request.node.qklearn_NMI = np.nan
        request.node.qklearn_cost = np.nan
        raise e
    
    assert(isinstance(labels_concl, list))
    assert(len(labels_concl) == n_samples)
    assert(len(set(labels_concl)) == n_clusters)

    result_concl = calc_eval_indices(X, labels_true, labels_concl)
    print(result_concl)

    # Aggregate the result data to output a test summary.
    request.node.qklearn_ANI = result_concl[0]
    request.node.qklearn_NMI = result_concl[1]
    request.node.qklearn_cost = result_concl[2]
    
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
        "3": (1600, 10),
        "4": (3330, 3),
        "5": (2660, 6),
        "10": (2000, 5),
        "13": (2500, 4),
        "23": (800, 20),
        "24": (450, 35),
        "25": (320, 50),
        "27": (320, 50),
        "28": (320, 50),
        "30": (510, 31),
        "35": (1060, 15),
        "36": (1060, 15),
        "37": (1060, 15),
        "38": (1060, 15),
        "40": (2000, 8),
        "41": (450, 35),
        "42": (640, 25),
        "62": (2280, 7),
        "63": (2280, 7),
        "64": (1600, 10),
        "65": (1770, 9),
    }

    if target_no in special_slices:
        num_out, num_classes = special_slices[target_no]
        ret_X, ret_labels_true = select_balanced_data(X, labels_true, num_out, num_classes)
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