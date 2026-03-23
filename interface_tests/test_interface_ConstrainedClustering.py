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
from qklearn.utils import read_token
from amplify import FixstarsClient
from sklearn.datasets import make_blobs
import numpy as np
import traceback

def setup_fixstars():
    client = FixstarsClient()
    client.token = read_token("Fixstars")
    client.parameters.timeout = timedelta(milliseconds=1000)

    return client

@pytest.fixture(scope="module")
def create_err_msg():
    err_msg = {}

    err_msg["1"] = "n_clusters must be greater than 0"
    err_msg["2"] = "n_clusters must be int"
    err_msg["3"] = "lam must be float"

    err_msg["180"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["182"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["184"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["186"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["188"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["190"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["192"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["194"] = ["conflict", "specified " + "balanced-sizes", "and " + "partition-level"]
    err_msg["196"] = ["conflict", "specified " + "balanced-sizes", "and " + "partition-level"]
    err_msg["198"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    
    err_msg["199"] = "There are samples whose cluster assignment is not uniquely determined."
    err_msg["200"] = "There are samples whose cluster assignment is not uniquely determined."
    
    err_msg["202"] = err_msg["184"]
    err_msg["204"] = err_msg["186"]
    err_msg["206"] = err_msg["190"]

    err_msg["207"] = ["does not satisfy", "must-link"]
    err_msg["208"] = ["does not satisfy", "cannot-link"]
    err_msg["209"] = ["does not satisfy", "partition-level"]
    err_msg["210"] = ["does not satisfy", "non-partition-level"]
    err_msg["211"] = ["does not satisfy", "limited-sizes"]
    err_msg["212"] = ["does not satisfy", "balanced-sizes"]
    err_msg["214"] = ["does not satisfy", "balanced-sizes"]
    err_msg["215"] = err_msg["207"]
    err_msg["216"] = err_msg["209"]
    err_msg["217"] = err_msg["210"]
    err_msg["219"] = err_msg["207"]
    err_msg["221"] = err_msg["207"]
    err_msg["223"] = err_msg["208"]
    err_msg["225"] = err_msg["209"]
    err_msg["227"] = err_msg["210"]
    err_msg["229"] = err_msg["211"]
    err_msg["231"] = err_msg["212"]
    err_msg["233"] = err_msg["214"]
    err_msg["235"] = err_msg["215"]
    err_msg["237"] = "Please set the instances in the order of partition_level -> non_partition_level -> must_link."
    err_msg["239"] = err_msg["225"]
    err_msg["241"] = err_msg["227"]

    err_msg["243"] = ["does not satisfy", "must-link"]
    err_msg["244"] = ["does not satisfy", "cannot-link"]
    err_msg["246"] = ["does not satisfy", "must-link"]
    err_msg["247"] = ["does not satisfy", "non-partition-level"]
    err_msg["249"] = ["does not satisfy", "must-link"]
    err_msg["250"] = ["does not satisfy", "limited-sizes"]
    err_msg["252"] = ["does not satisfy", "must-link"]
    err_msg["253"] = ["does not satisfy", "balanced-sizes"]
    err_msg["255"] = ["does not satisfy", "must-link"]
    err_msg["256"] = ["does not satisfy", "balanced-sizes"]
    err_msg["258"] = ["does not satisfy", "cannot-link"]
    err_msg["259"] = ["does not satisfy", "non-partition-level"]
    err_msg["261"] = ["does not satisfy", "cannot-link"]
    err_msg["262"] = ["does not satisfy", "limited-sizes"]
    err_msg["264"] = ["does not satisfy", "cannot-link"]
    err_msg["265"] = ["does not satisfy", "balanced-sizes"]
    err_msg["267"] = ["does not satisfy", "cannot-link"]
    err_msg["268"] = ["does not satisfy", "balanced-sizes"]
    err_msg["270"] = ["does not satisfy", "non-partition-level"]
    err_msg["271"] = ["does not satisfy", "limited-sizes"]
    err_msg["273"] = ["does not satisfy", "non-partition-level"]
    err_msg["274"] = ["does not satisfy", "balanced-sizes"]
    err_msg["276"] = ["does not satisfy", "non-partition-level"]
    err_msg["277"] = ["does not satisfy", "balanced-sizes"]
    err_msg["279"] = ["does not satisfy", "partition-level"]
    err_msg["280"] = ["does not satisfy", "must-link"]
    err_msg["282"] = ["does not satisfy", "partition-level"]
    err_msg["283"] = ["does not satisfy", "cannot-link"]
    err_msg["285"] = ["does not satisfy", "partition-level"]
    err_msg["286"] = ["does not satisfy", "non-partition-level"]
    err_msg["288"] = ["does not satisfy", "partition-level"]
    err_msg["289"] = ["does not satisfy", "limited-sizes"]
    err_msg["291"] = ["does not satisfy", "partition-level"]
    err_msg["292"] = ["does not satisfy", "balanced-sizes"]
    err_msg["294"] = ["does not satisfy", "partition-level"]
    err_msg["295"] = ["does not satisfy", "balanced-sizes"]
    err_msg["297"] = ["does not satisfy", "partition-level"]
    err_msg["298"] = ["does not satisfy", "non-partition-level"]
    err_msg["300"] = ["does not satisfy", "partition-level"]
    err_msg["301"] = "Please set the instances in the order of partition_level -> non_partition_level -> must_link."
    err_msg["303"] = "Please set the instances in the order of partition_level -> non_partition_level -> must_link."
    err_msg["304"] = "Please set the instances in the order of partition_level -> non_partition_level -> must_link."

    err_msg["308"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["310"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    
    err_msg["311"] = ["does not satisfy", "limited-sizes"]
    err_msg["312"] = ["does not satisfy", "limited-sizes"]
    err_msg["314"] = ["does not satisfy", "limited-sizes"]

    err_msg["316"] = ["does not satisfy", "must-link"]
    err_msg["317"] = ["does not satisfy", "limited-sizes"]
    err_msg["319"] = ["does not satisfy", "cannot-link"]
    err_msg["320"] = ["does not satisfy", "limited-sizes"]
    err_msg["322"] = ["does not satisfy", "partition-level"]
    err_msg["323"] = ["does not satisfy", "limited-sizes"]
    err_msg["325"] = ["does not satisfy", "non-partition-level"]
    err_msg["326"] = ["does not satisfy", "limited-sizes"]
    err_msg["328"] = ["does not satisfy", "must-link"]
    err_msg["329"] = ["does not satisfy", "limited-sizes"]
    err_msg["331"] = ["does not satisfy", "cannot-link"]
    err_msg["332"] = ["does not satisfy", "limited-sizes"]
    err_msg["334"] = ["does not satisfy", "partition-level"]
    err_msg["335"] = ["does not satisfy", "limited-sizes"]
    err_msg["337"] = ["does not satisfy", "non-partition-level"]
    err_msg["338"] = ["does not satisfy", "limited-sizes"]

    err_msg["345"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["346"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["347"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["348"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["349"] = ["does not satisfy", "partition-level"]
    err_msg["350"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["351"] = ["does not satisfy", "partition-level"]
    err_msg["352"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["353"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["354"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["355"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["356"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["357"] = ["does not satisfy", "non-partition-level"]
    err_msg["358"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["359"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."

    err_msg["360"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["362"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["363"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["364"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["365"] = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["366"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]

    return err_msg

@pytest.fixture(scope="module")
def create_constraint():
    test_constraint = {}

    test_constraint["108"] = {
        "must_link_instances": [(0, 1)],
    }
    test_constraint["109"] = {
        "cannot_link_instances": [(0, 1)],
    }
    test_constraint["110"] = {
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["111"] = {
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["112"] = {
        "limited_sizes": {0: 3, 1: 4, 2: 3},
    }
    test_constraint["113"] = {
        "balanced_sizes": (4, 3),
    }
    test_constraint["114"] = {
        "must_link_instances": [(0, 1)],
    }
    test_constraint["115"] = {
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["116"] = {
        "non_partition_level_instances": {0: (0,1)},
    }

    test_constraint["117"] = {
        "must_link_instances": [(0, 1)],
    }
    test_constraint["118"] = {
        "must_link_instances": [(0, 1)],
        "cannot_link_instances": [(0, 1)],
    }
    test_constraint["119"] = {
        "must_link_instances": [(0, 1)],
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["120"] = {
        "must_link_instances": [(0, 1)],
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["121"] = {
        "must_link_instances": [(0, 1)],
        "limited_sizes": {0: 3, 1: 4, 2: 3},
    }
    test_constraint["122"] = {
        "must_link_instances": [(0, 1)],
        "balanced_sizes": (4, 3),
    }

    test_constraint["123"] = {
        "must_link_instances": [(0, 1)],
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["124"] = {
        "cannot_link_instances": [(0, 1)],
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["125"] = {
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["126"] = {
        "partition_level_instances": {0: (0,1)},
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["127"] = {
        "partition_level_instances": {0: (0,1)},
        "limited_sizes": {0: 3, 1: 4, 2: 3},
    }
    test_constraint["128"] = {
        "partition_level_instances": {0: (0,1)},
        "balanced_sizes": (4, 3),
    }

    test_constraint["129"] = {
        "must_link_instances": [(0, 1)],
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["130"] = {
        "cannot_link_instances": [(0, 1)],
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["131"] = {
        "partition_level_instances": {0: (0,1)},
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["132"] = {
        "non_partition_level_instances": {0: (0,1)},
    }
    test_constraint["133"] = {
        "non_partition_level_instances": {0: (0,1)},
        "limited_sizes": {0: 3, 1: 4, 2: 3},
    }
    test_constraint["134"] = {
        "non_partition_level_instances": {0: (0,1)},
        "balanced_sizes": (4, 3),
    }

    test_constraint["135"] = {
        "must_link_instances": [(0, 1)],
    }
    test_constraint["136"] = {
        "partition_level_instances": {0: (0,1)},
    }
    test_constraint["137"] = {
        "non_partition_level_instances": {0: (0,1)},
    }

    test_constraint["151"] = {
        "must_link_instances" : [(0, 9), (1, 3), (4, 5)],
    }
    test_constraint["152"] = {
        "cannot_link_instances": [(0, 1), (2, 3), (4, 5)],
    }
    test_constraint["153"] = {
        "partition_level_instances": {0: (0,6,9), 1: (1, 3,  5, 7), 2: (2, 8)},
    }
    test_constraint["154"] = {
        "non_partition_level_instances": {0: (1, 2), 1: (4, 5), 2: (0, 3)},
    }
    test_constraint["155"] = {
        "limited_sizes": {0: 3, 1: 4, 2: 3},
    }
    test_constraint["156"] = {
        "balanced_sizes": None,
    }
    test_constraint["157"] = {
        "must_link_instances": [(0, 1), (2, 3), (4, 5)],
    }
    test_constraint["158"] = {
        "partition_level_instances": {0: (0,6,9), 1: (1, 3,  5, 7), 2: (2, 8)},
    }
    test_constraint["159"] = {
        "non_partition_level_instances": {0: (1, 2), 1: (4, 5), 2: (0, 3)},
    }
    test_constraint["160"] = test_constraint["151"]
    test_constraint["161"] = test_constraint["152"]
    test_constraint["162"] = test_constraint["153"]
    test_constraint["163"] = test_constraint["154"]
    test_constraint["164"] = test_constraint["155"]
    test_constraint["165"] = test_constraint["156"]
    test_constraint["166"] = test_constraint["157"]
    test_constraint["167"] = test_constraint["158"]
    test_constraint["168"] = test_constraint["159"]
    test_constraint["169"] = test_constraint["151"]
    test_constraint["170"] = test_constraint["152"]
    test_constraint["171"] = test_constraint["153"]
    test_constraint["172"] = test_constraint["154"]
    test_constraint["173"] = test_constraint["155"]
    test_constraint["174"] = test_constraint["156"]
    test_constraint["175"] = test_constraint["157"]
    test_constraint["176"] = test_constraint["158"]
    test_constraint["177"] = test_constraint["159"]

    test_constraint["179"] = {
        "must_link_instances" : [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances" : [(1, 4), (2, 3)]
    }
    test_constraint["180"] = {
        "must_link_instances" : [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances" : [(1, 4), (2, 1)]
    }
    test_constraint["181"] = {
        "must_link_instances" : [(0, 1), (2, 5), (3, 4)],
        "cannot_link_instances" : [(1, 4), (2, 3)]
    }
    test_constraint["182"] = {
        "must_link_instances" : [(0, 5), (1, 2), (3, 4)],
        "cannot_link_instances" : [(1, 4), (1, 2)]
    }
    test_constraint["183"] = {
        "must_link_instances" : [(0, 1), (0, 2), (3, 4)],
        "partition_level_instances" : {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["184"] = {
        "must_link_instances" : [(0, 1), (0, 2), (2, 3)],
        "partition_level_instances" : {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["185"] = {
        "must_link_instances" : [(0, 1), (2, 3), (4, 5)],
        "partition_level_instances" : {0: (0, 1), 1: (4, 5, 6), 2: (2, 3)}
    }
    test_constraint["186"] = {
        "must_link_instances" :  [(0, 1), (2, 3), (4, 5)],
        "partition_level_instances" : {0: (0, 1, 4), 1: (5, 6), 2: (2, 3)}
    }
    test_constraint["187"] = {
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["188"] = {
        "cannot_link_instances": [(0, 3), (4, 2), (1, 2)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["189"] = {
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_constraint["190"] = {
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (3, 4, 7)}
    }
    test_constraint["191"] = {
        "partition_level_instances": {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_constraint["192"] = {
        "partition_level_instances": {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8, 9)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_constraint["193"] = {
        "partition_level_instances": {0: (0, 1, 2, 3, 4), 1: (5, 6, 7, 8, 9)},
        "balanced_sizes": None,
    }
    test_constraint["194"] = {
        "partition_level_instances": {0: (0, 1, 2, 3), 1: (4, 5, 6, 7, 8, 9)},
        "balanced_sizes": None,
    }
    test_constraint["195"] = {
        "partition_level_instances": {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8, 9)},
        "balanced_sizes": None,
    }
    test_constraint["196"] = {
        "partition_level_instances": {0: (0, 1), 1: (2, 3, 4), 2: (5, 6, 7, 8, 9)},
        "balanced_sizes": None,
    }
    test_constraint["197"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["198"] = {
        "must_link_instances": [(0, 1), (8, 9), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (4, 7)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["201"] = test_constraint["183"]
    test_constraint["202"] = test_constraint["184"]
    test_constraint["203"] = {
        "must_link_instances": [(0, 1), (8, 9), (3, 4)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["204"] = {
        "must_link_instances": [(0, 1), (8, 9), (2, 3)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_constraint["205"] = test_constraint["189"]
    test_constraint["206"] = test_constraint["190"]

    test_constraint["307"] = {
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 5, 6, 7, 8)},
        "limited_sizes" :  {0: 4, 1: 0, 2: 6}
    }
    test_constraint["308"] = {
        "partition_level_instances": {0: (0, 1, 2), 1: (9, ), 2: (3, 4, 5, 6, 7, 8)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_constraint["309"] = {
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 5, 6, 7, 8)},
        "limited_sizes": {0: 4, 1: 0, 2: 6, 3: 0}
    }
    test_constraint["310"] = {
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 5, 6, 7, 8), 3: (9, )},
        "limited_sizes":  {0: 4, 1: 0, 2: 6, 3: 0}
    }

    test_constraint["359"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 10, 1:1, 2:0},
    }
    test_constraint["360"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 9, 1:1, 2:0},
    }
    test_constraint["361"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "limited_sizes": {0: 7, 1:3, 2:0},
    }
    test_constraint["362"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "limited_sizes": {0: 7, 1:2, 2:1},
    }
    test_constraint["363"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
    }
    test_constraint["364"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (0,1), 2: (0,1,2,3,4,5,6,7,8,9)},
    }
    test_constraint["365"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
    }
    test_constraint["366"] = {
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {1: (7,8), 2: (0,1,2,3,4,5,6,7,8,9)},
    }
    
    return test_constraint

@pytest.fixture(scope="module")
def create_test_data():
    test_data = {}
    labels_common = [0, 0, 0, 2, 2, 1, 0, 2, 1, 1]

    test_data["207"] = {
        "labels": labels_common,
        "const1": [(0, 1), (0, 2), (2, 6), (7, 8)]
    }
    test_data["208"] = {
        "labels": labels_common,
        "const1": [(0, 3), (4, 2), (1, 2)]
    }
    test_data["209"] = {
        "labels": labels_common,
        "const1": {0: (0, 1, 2), 2: (3, 4, 6)}
    }
    test_data["210"] = {
        "labels": labels_common,
        "const1": {0: (3, 4), 1: (6, 7), 2: (3, 4)}
    }
    test_data["211"] = {
        "labels": labels_common,
        "const1": {0: 3, 1: 3, 2: 4}
    }
    test_data["212"] = {
        "labels": [0] * 4 + [1] * 6,
        "const1": None
    }
    test_data["213"] = {
        "labels": [0] * 4 + [2] * 3 + [1] * 3,
        "const1": None
    }
    test_data["214"] = {
        "labels": [0] * 5 + [2] * 3 + [1] * 2,
        "const1": None
    }
    test_data["215"] = test_data["207"]
    test_data["216"] = test_data["209"]
    test_data["217"] = test_data["210"]

    test_data["218"] = {
        "labels": labels_common,
        "const1": [(0, 1), (2, 6)],
        "const2": [(3, 4), (8, 9)],
    }
    test_data["219"] = {
        "labels": labels_common,
        "const1": [(0, 1), (2, 6)],
        "const2": [(4, 5), (8, 9)]
    }
    test_data["220"] = {
        "labels": labels_common,
        "const1": [(0, 1), (0, 2)],
        "const2": [(3, 4), (8, 9)]
    }
    test_data["221"] = {
        "labels": labels_common,
        "const1": [(0, 1), (0, 2)],
        "const2": [(2, 4), (8, 9)]
    }
    test_data["222"] = {
        "labels": labels_common,
        "const1": [(0, 3), (4, 2)],
        "const2": [(7, 9)]
    }
    test_data["223"] = {
        "labels": labels_common,
        "const1": [(0, 3), (4, 2)],
        "const2": [(1, 2)]
    }
    test_data["224"] = {
        "labels": labels_common,
        "const1": {0: (0, 1, 2), 1: (8, 9)},
        "const2": {1: (5,), 2: (3, 4, 7)}
    }
    test_data["225"] = {
        "labels": labels_common,
        "const1": {0: (0, 1, 2), 1: (8, 9)},
        "const2": {1: (5,), 2: (3, 4, 6)}
    }
    test_data["226"] = {
        "labels": labels_common,
        "const1": {0: (3, 4), 1: (6, 7)},
        "const2": {2: (0, 1)}
    }
    test_data["227"] = {
        "labels": labels_common,
        "const1": {0: (3, 4), 1: (6, 7)},
        "const2": {2: (3, 4)}
    }
    test_data["228"] = {
        "labels": labels_common,
        "const1": {0: 4, 1: 4, 2: 3},
        "const2": {0: 5, 1: 3, 2: 3}
    }
    test_data["229"] = {
        "labels": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        "const1": {0: 4, 1: 5, 2: 2},
        "const2": {0: 5, 1: 4, 2: 2}
    }
    test_data["230"] = {
        "labels" : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "const1" : None,
        "const2" : None
    }
    test_data["231"] = {
        "labels" : [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        "const1" : None,
        "const2" : None
    }
    test_data["232"] = {
        "labels" : [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        "const1" : None,
        "const2" : None
    }
    test_data["233"] = {
        "labels" : [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
        "const1" : None,
        "const2" : None
    }
    test_data["234"] = test_data["218"]
    test_data["235"] = test_data["219"]
    test_data["236"] = test_data["220"]
    test_data["237"] = test_data["221"]
    test_data["238"] = test_data["224"]
    test_data["239"] = test_data["225"]
    test_data["240"] = test_data["226"]
    test_data["241"] = test_data["227"]

    test_data["311"] = {
        "labels": [0] * 4 + [1] + [2] * 5,
        "const1": {0: 5, 1: 0, 2: 5}
    }
    test_data["312"] = {
        "labels": [0] * 8 + [1, 2],
        "const1": {0: 9, 1: 1, 2: 0, 3:0}
    }
    test_data["313"] = {
        "labels" : [2] * 10,
        "const1" : {0 : 1, 1 : 0, 2 : 10},
        "const2" : {0 : 0, 1 : 1, 2 : 10}
    }
    test_data["314"] = {
        "labels" : [1] + [2] * 9,
        "const1" : {0 : 1, 1 : 0, 2 : 10},
        "const2" : {0 : 0, 1 : 1, 2 : 10}
    }

    test_data["345"] = {
        "labels": [0] * 10,
        "const1": {0: (0,1,2,3,4)},
        "const2": {0: (5,6,7,8,9)}
    }
    test_data["346"] = {
        "labels": [0] * 10,
        "const1": {0: (0,1,2,3,4)},
        "const2": {0: (5,6), 1:(7,8,9)}
    }
    test_data["347"] = {
        "labels": [0] * 10,
        "const1": {0: (0,1,2,3,4)},
        "const2": {0: (5,6,7,8,9)}
    }

    return test_data

@pytest.fixture(scope="module")
def create_test_multi_data():
    test_multi_data = {}
    labels_common = [0, 0, 0, 2, 2, 1, 0, 2, 1, 1]

    test_multi_data["242"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 3)]
    }
    test_multi_data["243"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 5)],
        "cannot_link_instances": [(1, 4), (2, 3)]
    }
    test_multi_data["244"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 6)]
    }
    test_multi_data["245"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["246"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 5)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["247"] ={
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 4)}
    }
    test_multi_data["248"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["249"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 5)],
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["250"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 3, 1: 4, 2: 3}
    }
    test_multi_data["251"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "balanced_sizes" : None
    }
    test_multi_data["252"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "must_link_instances": [(0, 1), (2, 6), (3, 5)],
        "balanced_sizes" : None
    }
    test_multi_data["253"] = {
        "labels": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "balanced_sizes" : None
    }
    test_multi_data["254"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "balanced_sizes" : None
    }
    test_multi_data["255"] ={
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 5)],
        "balanced_sizes" : None
    }
    test_multi_data["256"] = {
        "labels": [0, 0, 0, 2, 2, 1, 1, 1, 1, 1],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "balanced_sizes" : None
    }
    test_multi_data["257"] = {
        "labels": labels_common,
        "cannot_link_instances": [(1, 4), (2, 3)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["258"] = {
        "labels": labels_common,
        "cannot_link_instances": [(1, 4), (2, 6)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["259"] = {
        "labels": labels_common,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 7)}
    }
    test_multi_data["260"] = {
        "labels": labels_common,
        "cannot_link_instances": [(1, 4), (2, 3)],
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["261"] = {
        "labels": labels_common,
        "cannot_link_instances": [(1, 4), (2, 6)],
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["262"] = {
        "labels": labels_common,
        "cannot_link_instances": [(1, 4), (2, 3)],
        "limited_sizes": {0: 3, 1: 4, 2: 3}
    }
    test_multi_data["263"] = {
        "labels": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "balanced_sizes" : None
    }
    test_multi_data["264"] = {
        "labels": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        "cannot_link_instances": [(0, 3), (4, 2), (1, 9)],
        "balanced_sizes" : None
    }
    test_multi_data["265"] = {
        "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "cannot_link_instances": [(0, 4), (4, 2), (1, 5)],
        "balanced_sizes" : None
    }
    test_multi_data["266"] = {
        "labels": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        "cannot_link_instances": [(0, 4), (4, 7), (2, 9)],
        "balanced_sizes" : None
    }
    test_multi_data["267"] = {
        "labels": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        "cannot_link_instances": [(0, 5), (4, 7), (8, 9)],
        "balanced_sizes" : None
    }
    test_multi_data["268"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        "cannot_link_instances": [(0, 5), (4, 7), (2, 9)],
        "balanced_sizes" : None
    }
    test_multi_data["269"] = {
        "labels": labels_common,
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["270"] = {
        "labels": labels_common,
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 4)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["271"] = {
        "labels": labels_common,
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)},
        "limited_sizes": {0: 3, 1: 4, 2: 3}
    }
    test_multi_data["272"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "non_partition_level_instances": {0: (5, 6), 1: (3, 4)},
        "balanced_sizes" : None
    }
    test_multi_data["273"] = {
        "labels": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        "non_partition_level_instances": {0: (8, 9), 1: (3, 4)},
        "balanced_sizes" : None
    }
    test_multi_data["274"] = {
        "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "non_partition_level_instances": {0: (5, 6), 1: (2, 3)},
        "balanced_sizes" : None
    }
    test_multi_data["275"] = {
        "labels": labels_common,
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)},
        "balanced_sizes" : None
    }
    test_multi_data["276"] = {
        "labels": labels_common,
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 7)},
        "balanced_sizes" : None
    }
    test_multi_data["277"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 1, 1],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)},
        "balanced_sizes" : None
    }
    test_multi_data["278"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["279"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)}
    }
    test_multi_data["280"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (5, 6)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["281"] = {
        "labels": labels_common,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["282"] = {
        "labels": labels_common,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 6), 2: (3, 4, 7)}
    }
    test_multi_data["283"] = {
        "labels": labels_common,
        "cannot_link_instances": [(0, 3), (4, 2), (8, 9)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["284"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["285"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["286"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 7)}
    }
    test_multi_data["287"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["288"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (5, 4, 7)},
        "limited_sizes": {0: 4, 1: 3, 2: 3}
    }
    test_multi_data["289"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "limited_sizes": {0: 3, 1: 4, 2: 3}
    }
    test_multi_data["290"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 6, 7)},
        "balanced_sizes" : None
    }
    test_multi_data["291"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "partition_level_instances": {0: (0, 1, 2, 8), 1: (5, 6, 7)},
        "balanced_sizes" : None
    }
    test_multi_data["292"] = {
        "labels": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 6, 7)},
        "balanced_sizes" : None
    }
    test_multi_data["293"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4)},
        "balanced_sizes" : None
    }
    test_multi_data["294"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2, 8), 2: (3, 4)},
        "balanced_sizes" : None
    }
    test_multi_data["295"] = {
        "labels": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        "partition_level_instances": {0: (0, 1, 2), 2: (8, 9)},
        "balanced_sizes" : None
    }
    test_multi_data["296"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["297"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["298"] = {
        "labels": labels_common,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4)},
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 7)}
    }
    test_multi_data["299"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["300"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 4)],
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)}
    }
    test_multi_data["301"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 5), (3, 4)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)}
    }
    test_multi_data["302"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["303"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (2, 6), (3, 4)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2, 4)}
    }
    test_multi_data["304"] = {
        "labels": labels_common,
        "must_link_instances": [(0, 1), (0, 2), (3, 5)],
        "non_partition_level_instances": {0: (3, 4), 1: (6, 7), 2: (1, 2)}
    }
    test_multi_data["315"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["316"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "must_link_instances": [(0, 1), (0, 2), (8, 9)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["317"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 1],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["318"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "cannot_link_instances": [(1, 4), (2, 3)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["319"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "cannot_link_instances": [(1, 4), (2, 9)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["320"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 1],
        "cannot_link_instances": [(1, 4), (2, 3)],
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["321"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["322"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "partition_level_instances": {0: (0, 1, 2), 2: (4, 7, 9)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["323"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 1],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["324"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "non_partition_level_instances": {0: (3, 4), 2: (1, 2)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["325"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
        "non_partition_level_instances": {0: (0, 4), 2: (1, 2)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["326"] = {
        "labels": [0, 0, 0, 2, 2, 2, 2, 2, 2, 1],
        "non_partition_level_instances": {0: (3, 4), 2: (1, 2)},
        "limited_sizes": {0: 4, 1: 0, 2: 6}
    }
    test_multi_data["327"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["328"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "must_link_instances": [(0, 1), (0, 2), (8, 9)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["329"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["330"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "cannot_link_instances": [(1, 9), (2, 9)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["331"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "cannot_link_instances": [(1, 9), (2, 3)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["332"] = {
        "labels": [2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "cannot_link_instances": [(1, 9), (2, 9)],
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["333"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "partition_level_instances": {0: (0, 1, 2), 1: (9,)},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["334"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "partition_level_instances": {0: (0, 1, 2), 1: (8, )},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["335"] = {
        "labels": [0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        "partition_level_instances": {0: (0, 1, 2), 1: (9,)},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["336"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "non_partition_level_instances": {0: (9,), 1: (1, 2)},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["337"] = {
        "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "non_partition_level_instances": {0: (9, 8), 1: (1, 2)},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }
    test_multi_data["338"] = {
        "labels": [3, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "non_partition_level_instances": {0: (9,), 1: (1, 2)},
        "limited_sizes": {0: 9, 1: 1, 2: 0, 3: 0}
    }

    test_multi_data["348"] = {
        "labels": [0] * 10,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 10, 1:1, 2:0},
    }
    test_multi_data["349"] = {
        "labels": [0] * 9 + [1],
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 10, 1:1, 2:0},
    }
    
    test_multi_data["350"] = {
        "labels": [0] * 10,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "limited_sizes": {0: 8, 1:3, 2:0},
    }
    test_multi_data["351"] = {
        "labels": [0] * 7 + [1] * 2 + [0],
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "limited_sizes": {0: 8, 1:3, 2:0},
    }
    
    test_multi_data["352"] = {
        "labels": [0] * 10,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)}
    }
    test_multi_data["353"] = {
        "labels": [0] * 9 + [1],
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)}
    }
    test_multi_data["354"] = {
        "labels": [0] * 10,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)}
    }
    test_multi_data["355"] = {
        "labels": [0] * 7 + [1] * 2 + [0],
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)}
    }
    
    test_multi_data["356"] = {
        "labels": [0] * 10,
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 5, 1: 5, 2: 1}
    }
    test_multi_data["357"] = {
        "labels": [0] * 5 + [1] * 4 + [2],
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 5, 1: 5, 2: 1}
    }
    test_multi_data["358"] = {
        "labels": [0] * 5 + [1] * 4 + [0],
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": {0: 5, 1: 5, 2: 1}
    }

    return test_multi_data

@pytest.mark.parametrize(
    ["n_clusters", "lam", "e", "No"],
    [
        pytest.param(0,   None, "V", "1", id="No.1"),
        pytest.param(2.5, None, "T", "2", id="No.2"),
        pytest.param(3,   "12", "T", "3", id="No.3"),
    ]
)
def test_constructor_error(create_err_msg, n_clusters, lam, e, No):
    err_msg = create_err_msg.get(No)
    if (e == "V"):
        with pytest.raises(ValueError) as e:
            if (lam is None):
                ccl = ConstrainedClustering(n_clusters=n_clusters)
            else:
                ccl = ConstrainedClustering(n_clusters=n_clusters, lam=lam)
    elif (e == "T"):
        with pytest.raises(TypeError) as e:
            if (lam is None):
                ccl = ConstrainedClustering(n_clusters=n_clusters)
            else:
                ccl = ConstrainedClustering(n_clusters=n_clusters, lam=lam)

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["n_clusters", "lam", "weight_balanced_size", "dtype", "client"],
    [
        pytest.param(1, None, None, "N", "F", id="No.4"),
        pytest.param(2, None, None, "L", "N", id="No.5"),
        pytest.param(3, 12,   None, "N", "N", id="No.6"),
        pytest.param(3, 13.2, None, "L", "F", id="No.7"),
        pytest.param(3, None, 12,   "N", "F", id="No.8"),
        pytest.param(3, None, 13.2, "N", "F", id="No.9"),
    ]
)
def test_add_balanced_sizes_to_qubo_normal(n_clusters, lam,
                                           weight_balanced_size, dtype, client):
    ccl = ConstrainedClustering(n_clusters=n_clusters, lam=lam)
    data, _ = make_blobs(random_state=8, n_samples=12, n_features=2,
                         cluster_std=1.5, centers=3)
    if (dtype == "L"):
        data = data.tolist()
    if (client == "F"):
        client = setup_fixstars()
    elif (client == "N"):
        client = None
    else:
        raise Exception("Invalid Client")

    ccl.fit(data)

    if (weight_balanced_size is None):
        ccl.add_balanced_sizes_to_qubo()
    else:
        ccl.add_balanced_sizes_to_qubo(weight_balanced_size=weight_balanced_size)

    labels = ccl.predict(client)

    assert(isinstance(labels, list))
    assert(len(labels) == 12)
    assert(len(set(labels)) == n_clusters)
    for i in set(labels):
        c = 0
        for j in labels:
            if (i == j):
                c += 1
        assert(c == 12/n_clusters)

@pytest.mark.parametrize(
    ["weight_balanced_size", "e"],
    [
        pytest.param("11", "T", id="No.10"),
        pytest.param(12,   "A", id="No.11"),
    ]
)
def test_add_balanced_sizes_to_qubo_error(weight_balanced_size, e):
    data, _ = make_blobs(random_state=8, n_samples=12, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_balanced_sizes_to_qubo(weight_balanced_size=
                                           weight_balanced_size)
            assert("weight_balanced_size must be float" in str(e.value))
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_balanced_sizes_to_qubo(weight_balanced_size=
                                           weight_balanced_size)
            assert("This instance is not fitted yet" in str(e.value))
    else:
        raise Exception("Invalid Error Case")

@pytest.mark.parametrize(
    ["cannot_link_instances", "weight_cannot_link", "lam"],
    [
        pytest.param([(0, 1), (2, 3), (4, 5)], None, None, id="No.12"),
        pytest.param([(0, 1), (2, 3), (4, 5)], 10, None, id="No.13"),
        pytest.param([(0, 1), (2, 3), (4, 5)], None, 11, id="additional"),
        pytest.param([(0, 1), (2, 3), (4, 5)], 10, 11, id="additional"),
    ]
)
def test_add_cannot_link_to_qubo_normal(cannot_link_instances,
                                        weight_cannot_link, lam):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2, lam=lam)
    ccl.fit(data)
    client = setup_fixstars()
    if (weight_cannot_link is None):
        ccl.add_cannot_link_to_qubo(cannot_link_instances)
    else:
        ccl.add_cannot_link_to_qubo(cannot_link_instances,weight_cannot_link)

    labels = ccl.predict(client)

    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == 2)
    for i in cannot_link_instances:
        assert(labels[i[0]] != labels[i[1]])

@pytest.mark.parametrize(
    ["cannot_link_instances", "weight_cannot_link", "e", "err_msg"],
    [
        pytest.param([(0, 1, 2)],      None, "V", "must be two", id="No.14"),
        pytest.param([(0, -1)],        None, "V", "greater than or equal to 0", id="No.15"),
        pytest.param([(0, 10)],        None, "V", "less than number of samples", id="No.16"),
        pytest.param((0 ,1),           None, "T", "must be list", id="No.17"),
        pytest.param([[0, 1], [0, 2]], None, "T", "must contain tuples", id="No.18"),
        pytest.param([(0, 1.0)],       None, "T", "must be int", id="No.19"),
        pytest.param([(0, 1)],         "10", "T", "must be float", id="No.20"),
        pytest.param([(0, 1)],         None, "A", "This instance is not fitted yet", id="No.21"),
    ]
)
def test_add_cannot_link_to_qubo_error(cannot_link_instances,
                                       weight_cannot_link, e, err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.add_cannot_link_to_qubo(cannot_link_instances, weight_cannot_link)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_cannot_link_to_qubo(cannot_link_instances, weight_cannot_link)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_cannot_link_to_qubo(cannot_link_instances, weight_cannot_link)
    else:
        raise Exception("Invalid Case")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["limited_sizes", "weight_limited_sizes", "specified_cluster", "expected_cluster", "lam"],
    [
        pytest.param({0: 5, 1: 3, 2: 2}, None, 3, 3, None, id="No.22"),
        pytest.param({0: 9, 1: 1, 2: 0}, 10,   3, 2, None, id="No.23"),
        pytest.param({0: 5, 1: 3, 2: 2}, None, 3, 3, 11, id="additional"),
        pytest.param({0: 9, 1: 1, 2: 0}, 10,   3, 2, 11, id="additional"),
        pytest.param({0: 5, 1: 5, 2: 0, 3: 0}, None, 4, 2, None, id="No.305"),
    ]
)
def test_add_limited_sizes_to_qubo_normal(limited_sizes, weight_limited_sizes,
                                          specified_cluster, expected_cluster, lam):
    ccl = ConstrainedClustering(n_clusters=specified_cluster, lam=lam)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=specified_cluster)
    ccl.fit(data)
    if (weight_limited_sizes is None):
        ccl.add_limited_sizes_to_qubo(limited_sizes)
    else:
        ccl.add_limited_sizes_to_qubo(limited_sizes, weight_limited_sizes)
    client = setup_fixstars()
    labels = ccl.predict(client)

    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == expected_cluster)
    for i in set(labels):
        c = 0
        if (limited_sizes.get(i) is None):
            continue
        for j in labels:
            if (i == j):
                c += 1
        assert(limited_sizes[i] >= c)

@pytest.mark.parametrize(
    ["limited_sizes", "weight_limited_sizes", "e", "err_msg"],
    [
        pytest.param({0: -1, 1:5, 2:5}, None, "V", "greater than or equal to 0", id="No.24"),
        pytest.param({-1: 1, 0:0, 1:5, 2:5}, None, "V", "greater than or equal to 0", id="No.25"),
        pytest.param({3: 1, 0:0, 1:5, 2:5}, None, "V", "exceeds the number of clusters", id="No.26"),
        pytest.param({2:1}, None, "V", "Specify upper limit for all clusters", id="No.27"),
        pytest.param((2, 1), None, "T", "limited_sizes must be dict", id="No.28"),
        pytest.param({0:5, 1:4, 2: 1.0}, None, "T", "must be int", id="No.29"),
        pytest.param({0:5, 1:4, 2.0: 1}, None, "T", "must be int", id="No.30"),
        pytest.param({0:5, 1:3, 2: 2}, "10", "T", "must be float", id="No.31"),
        pytest.param({0:5, 1:3, 2: 2}, 10, "A", "not fitted yet", id="No.32"),
    ]
)
def test_add_limited_sizes_to_qubo_error(limited_sizes, weight_limited_sizes,
                                         e, err_msg):
    ccl = ConstrainedClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.add_limited_sizes_to_qubo(limited_sizes, weight_limited_sizes)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_limited_sizes_to_qubo(limited_sizes, weight_limited_sizes)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_limited_sizes_to_qubo(limited_sizes, weight_limited_sizes)
    else:
        raise Exception("Invalid Error Case")

    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["must_link_instances", "weight_must_link", "lam"],
    [
        pytest.param([(0, 1), (0, 9)], None, None, id="No.33"),
        pytest.param([(0, 1), (0, 9)], 10, None, id="No.34"),
        pytest.param([(0, 1), (0, 9)], None, 11, id="additional"),
        pytest.param([(0, 1), (0, 9)], 10, 11, id="additional"),
    ]
)
def test_add_must_link_to_qubo_normal(must_link_instances, weight_must_link, lam):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2, lam=lam)
    ccl.fit(data)
    client = setup_fixstars()
    if (weight_must_link is None):
        ccl.add_must_link_to_qubo(must_link_instances)
    else:
        ccl.add_must_link_to_qubo(must_link_instances, weight_must_link)

    labels = ccl.predict(client)

    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == 2)
    for i in must_link_instances:
        assert(labels[i[0]] == labels[i[1]])

@pytest.mark.parametrize(
    ["must_link_instances", "weight_must_link", "e", "err_msg"],
    [
        pytest.param([(0, 1, 2)],      None, "V", "must be two", id="No.35"),
        pytest.param([(0, -1)],        None, "V", "greater than or equal to 0", id="No.36"),
        pytest.param([(0, 10)],        None, "V", "less than number of samples", id="No.37"),
        pytest.param((0 ,1),           None, "T", "must be list", id="No.38"),
        pytest.param([[0, 1], [0, 2]], None, "T", "must contain tuples", id="No.39"),
        pytest.param([(0, 1.0)],       None, "T", "must be int", id="No.40"),
        pytest.param([(0, 1)],         "10", "T", "must be float", id="No.41"),
        pytest.param([(0, 1)],         None, "A", "This instance is not fitted yet", id="No.42"),
    ]
)
def test_add_must_link_to_qubo_error(must_link_instances,
                                       weight_must_link, e, err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.add_must_link_to_qubo(must_link_instances, weight_must_link)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_must_link_to_qubo(must_link_instances, weight_must_link)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_must_link_to_qubo(must_link_instances, weight_must_link)
    else:
        raise Exception("Invalid Case")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["non_partition_level_instances", "weight_non_partition_level", "lam", "n_clusters_apparent"],
    [
        pytest.param({0: (0, 9)}, None, None, 3, id="No.43"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 10, None, 3, id="No.44"),
        pytest.param({0: (0, 9)}, None, 11, 3, id="additional"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 10, 11, 3, id="additional"),
    ],
)
def test_add_non_partition_level_to_qubo_normal(non_partition_level_instances,
                                                weight_non_partition_level, lam, n_clusters_apparent):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=3, lam=lam)
    ccl.fit(data)
    if (weight_non_partition_level is None):
        ccl.add_non_partition_level_to_qubo(non_partition_level_instances)
    else:
        ccl.add_non_partition_level_to_qubo(non_partition_level_instances,
                                            weight_non_partition_level)
    client = setup_fixstars()
    labels = ccl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == n_clusters_apparent)
    for k in non_partition_level_instances.keys():
        for v in non_partition_level_instances[k]:
            assert(labels[v] != k)

@pytest.mark.parametrize(
    ["non_partition_level_instances", "weight_non_partition_level", "e", "err_msg"],
    [
        pytest.param({0: (-1, 9)},   None, "V", "greater than or equal to 0", id="No.45"),
        pytest.param({-1: (0, 9)},   None, "V", "greater than or equal to 0", id="No.46"),
        pytest.param({3: (0, 9)},    None, "V", "less than number of clusters", id="No.47"),
        pytest.param({0: (0, 10)},   None, "V", "less than number of samples", id="No.48"),
        pytest.param((0, 10),        None, "T", "must be dict", id="No.49"),
        pytest.param({1.0: (0, 9)},  None, "T", "must be int", id="No.50"),
        pytest.param({0: (0, 9.0)},  None, "T", "must be int", id="No.51"),
        pytest.param({0: 0},         None, "T", "must be tuples containing indices of samples", id="No.52"),
        pytest.param({0: (0, 9)},    "10", "T", "must be float", id="No.53"),
        pytest.param({0: (0, 9)},    None, "A", "This instance is not fitted yet", id="No.54"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}, None, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.343"),
    ]
)
def test_add_non_partition_level_to_qubo_error(non_partition_level_instances,
                                               weight_non_partition_level,
                                               e, err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.add_non_partition_level_to_qubo(non_partition_level_instances,
                                                weight_non_partition_level)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_non_partition_level_to_qubo(non_partition_level_instances,
                                                weight_non_partition_level)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_non_partition_level_to_qubo(non_partition_level_instances,
                                                weight_non_partition_level)
    elif (e == "R"):
        ccl.fit(data)
        ccl.add_non_partition_level_to_qubo(non_partition_level_instances,
                                            weight_non_partition_level)
        client = setup_fixstars()
        with pytest.raises(RuntimeError) as e:
            labels = ccl.predict(client)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["partition_level_instances", "weight_partition_level", "lam", "n_clusters_apparent"],
    [
        pytest.param({0: (0, 9)}, None, None, 3, id="No.55"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 10, None, 3, id="No.56"),
        pytest.param({0: (0, 9)}, None, 11, 3, id="additional"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 10, 11, 3, id="additional"),
    ],
)
def test_add_partition_level_to_qubo_normal(partition_level_instances,
                                            weight_partition_level, lam, n_clusters_apparent):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=3, lam=lam)
    ccl.fit(data)
    if (weight_partition_level is None):
        ccl.add_partition_level_to_qubo(partition_level_instances)
    else:
        ccl.add_partition_level_to_qubo(partition_level_instances,
                                            weight_partition_level)
    client = setup_fixstars()
    labels = ccl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == n_clusters_apparent)
    for k in partition_level_instances.keys():
        for v in partition_level_instances[k]:
            assert(labels[v] == k)

@pytest.mark.parametrize(
    ["partition_level_instances", "weight_partition_level", "e", "err_msg"],
    [
        pytest.param({0: (-1, 9)},   None, "V", "greater than or equal to 0", id="No.57"),
        pytest.param({-1: (0, 9)},   None, "V", "greater than or equal to 0", id="No.58"),
        pytest.param({3: (0, 9)},    None, "V", "less than number of clusters", id="No.59"),
        pytest.param({0: (0, 10)},   None, "V", "less than number of samples", id="No.60"),
        pytest.param((0, 10),        None, "T", "must be dict", id="No.61"),
        pytest.param({1.0: (0, 9)},  None, "T", "must be int", id="No.62"),
        pytest.param({0: (0, 9.0)},  None, "T", "must be int", id="No.63"),
        pytest.param({0: 0},         None, "T", "must be tuples containing indices of samples", id="No.64"),
        pytest.param({0: (0, 9)},    "10", "T", "must be float", id="No.65"),
        pytest.param({0: (0, 9)},    None, "A", "This instance is not fitted yet", id="No.66"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}, None, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.339"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6), 1: (7, 8, 9)}, None, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.340"),
    ]
)
def test_add_partition_level_to_qubo_error(partition_level_instances,
                                           weight_partition_level, e, err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.add_partition_level_to_qubo(partition_level_instances,
                                            weight_partition_level)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.add_partition_level_to_qubo(partition_level_instances,
                                            weight_partition_level)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.add_partition_level_to_qubo(partition_level_instances,
                                            weight_partition_level)
    elif (e == "R"):
        ccl.fit(data)
        ccl.add_partition_level_to_qubo(partition_level_instances,
                                            weight_partition_level)
        client = setup_fixstars()
        with pytest.raises(RuntimeError) as e:
            labels = ccl.predict(client)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

@pytest.fixture(scope="module")
def create_fit_data():
    test_data = {}

    invalid_listf2 = [[i ,i] for i in range(10)]
    invalid_listf2[9][1] = "h"
    invalid_listf3 = [[i ,i, i] for i in range(10)]
    invalid_listf3[9][2] = "h"

    test_data["67"] = {1: [1, 1]}
    test_data["68"] = np.array(invalid_listf2)
    test_data["69"] = invalid_listf3
    test_data["70"] = [[1,2,3], [1,2]]
    test_data["71"] = np.arange(8).reshape((2, 2, 2))
    test_data["72"] = [[[1,2], [1,2]], [[1,2], [1,2]]]
    test_data["73"] = np.arange(8).reshape((8,))
    test_data["74"] = [1, 2, 3, 4, 5, 6, 7, 8]

    return test_data


@pytest.mark.parametrize(
    ["No", "e", "err_msg"],
    [
        pytest.param("67", "T", "must be array-like object", id="No.67"),
        pytest.param("68", "T", "has invalid data type", id="No.68"),
        pytest.param("69", "T", "has invalid data type", id="No.69"),
        pytest.param("70", "V", "has invalid shape", id="No.70"),
        pytest.param("71", "V", "must be 2-d array", id="No.71"),
        pytest.param("72", "V", "must be 2-d array", id="No.72"),
        pytest.param("73", "V", "must be 2-d array", id="No.73"),
        pytest.param("74", "V", "must be 2-d array", id="No.74"),
    ]
)
def test_fit_error(create_fit_data, No, e, err_msg):
    indata = create_fit_data.get(No)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "T"):
        with pytest.raises(TypeError) as e:
            ccl.fit(indata)
    elif (e == "V"):
        with pytest.raises(ValueError) as e:
            ccl.fit(indata)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dwave-neal", id="No.75"),
    ],
)
@pytest.mark.neal
def test_predict_error_neal(lack_package):
    client = None
    ccl = ConstrainedClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = ccl.predict(client)

@pytest.mark.parametrize(
    ["lack_package"],
    [
        pytest.param("dimod", id="No.76"),
    ],
)
@pytest.mark.dimod
def test_predict_error_dimod(lack_package):
    client = None
    ccl = ConstrainedClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl.fit(data)
    print(f"A case when {lack_package} is not installed")
    with pytest.raises(RuntimeError) as e:
        label = ccl.predict(client)

@pytest.mark.parametrize(
    ["Case", "err_msg"],
    [
        pytest.param("InvalidToken", "exceptions were raised from solve() of the Fixstars Amplify SDK", id="No.77"),
        pytest.param("NotFitted", "This instance is not fitted yet", id="No.78"),
    ],
)
def test_predict_error(Case, err_msg):
    client = setup_fixstars()
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (Case == "InvalidToken"):
        ccl.fit(data)
        client.token = "hoge"
        with pytest.raises(RuntimeError) as e:
            labels = ccl.predict(client)
    elif (Case == "NotFitted"):
        with pytest.raises(AttributeError) as e:
            labels = ccl.predict(client)
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["must_link_instances", "lam"],
    [
        pytest.param([(0, 1), (0, 9)], None, id="No.79"),
        pytest.param([(0, 1), (0, 9)], 11, id="additional"),
    ]
)
def test_set_must_link_by_qbits_reduction_normal(must_link_instances, lam):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2, lam=lam)
    ccl.fit(data)
    client = setup_fixstars()
    ccl.set_must_link_by_qbits_reduction(must_link_instances)

    labels = ccl.predict(client)

    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == 2)
    for i in must_link_instances:
        assert(labels[i[0]] == labels[i[1]])

@pytest.mark.parametrize(
    ["must_link_instances", "e", "err_msg"],
    [
        pytest.param([(0, 1, 2)],      "V", "must be two", id="No.80"),
        pytest.param([(0, -1)],        "V", "greater than or equal to 0", id="No.81"),
        pytest.param([(0, 10)],        "V", "less than number of samples", id="No.82"),
        pytest.param((0 ,1),           "T", "must be list", id="No.83"),
        pytest.param([[0, 1], [0, 2]], "T", "must contain tuples", id="No.84"),
        pytest.param([(0, 1.0)],       "T", "must be int", id="No.85"),
        pytest.param([(0, 1)],         "A", "This instance is not fitted yet", id="No.86"),
    ]
)
def test_set_must_link_by_qbits_reduction_error(must_link_instances, e, err_msg):
    data, _ = make_blobs(random_state=8, n_samples=10,
                         n_features=2, cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=2)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.set_must_link_by_qbits_reduction(must_link_instances)
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.set_must_link_by_qbits_reduction(must_link_instances)
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.set_must_link_by_qbits_reduction(must_link_instances)
    else:
        raise Exception("Invalid Case")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["non_partition_level_instances", "lam", "n_clusters_apparent"],
    [
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, None, 3, id="No.87"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 11, 3, id="additional"),
    ],
)
def test_set_non_partition_level_by_qbits_reduction_normal(non_partition_level_instances, lam, n_clusters_apparent):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=3, lam=lam)
    ccl.fit(data)
    ccl.set_non_partition_level_by_qbits_reduction(non_partition_level_instances)
    client = setup_fixstars()
    labels = ccl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == n_clusters_apparent)
    for k in non_partition_level_instances.keys():
        for v in non_partition_level_instances[k]:
            assert(labels[v] != k)

@pytest.mark.parametrize(
    ["non_partition_level_instances", "e", "err_msg"],
    [
        pytest.param({0: (-1, 9)},  "V", "greater than or equal to 0", id="No.88"),
        pytest.param({-1: (0, 9)},  "V", "greater than or equal to 0", id="No.89"),
        pytest.param({3: (0, 9)},   "V", "less than number of clusters", id="No.90"),
        pytest.param({0: (0, 10)},  "V", "less than number of samples", id="No.91"),
        pytest.param((0, 10),       "T", "must be dict", id="No.92"),
        pytest.param({1.0: (0, 9)}, "T", "must be int", id="No.93"),
        pytest.param({0: (0, 9.0)}, "T", "must be int", id="No.94"),
        pytest.param({0: 0},        "T", "must be tuples containing indices of samples", id="No.95"),
        pytest.param({0: (0, 9)},   "A", "This instance is not fitted yet", id="No.96"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.344"),
    ]
)
def test_set_non_partition_level_by_qbits_reduction_error(
    non_partition_level_instances, e, err_msg
):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.set_non_partition_level_by_qbits_reduction(
                non_partition_level_instances
            )
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.set_non_partition_level_by_qbits_reduction(
                non_partition_level_instances
            )
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.set_non_partition_level_by_qbits_reduction(
                non_partition_level_instances
            )
    elif (e == "R"):
        ccl.fit(data)
        ccl.set_non_partition_level_by_qbits_reduction(
            non_partition_level_instances
        )
        client = setup_fixstars()
        with pytest.raises(RuntimeError) as e:
            labels = ccl.predict(client)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

@pytest.mark.parametrize(
    ["partition_level_instances", "lam", "n_clusters_apparent"],
    [
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, None, 3, id="No.97"),
        pytest.param({0: (0, 9), 1: (1, 8), 2:(2, 3, 6, 7)}, 11, 3, id="additional"),
    ],
)
def test_set_partition_level_by_qbits_reduction_normal(
        partition_level_instances, lam, n_clusters_apparent
):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=2)
    ccl = ConstrainedClustering(n_clusters=3, lam=lam)
    ccl.fit(data)
    ccl.set_partition_level_by_qbits_reduction(partition_level_instances)
    client = setup_fixstars()
    labels = ccl.predict(client)
    assert(isinstance(labels, list))
    assert(len(labels) == 10)
    assert(len(set(labels)) == n_clusters_apparent)
    for k in partition_level_instances.keys():
        for v in partition_level_instances[k]:
            assert(labels[v] == k)

@pytest.mark.parametrize(
    ["partition_level_instances", "e", "err_msg"],
    [
        pytest.param({0: (-1, 9)},  "V", "greater than or equal to 0", id="No.98"),
        pytest.param({-1: (0, 9)},  "V", "greater than or equal to 0", id="No.99"),
        pytest.param({3: (0, 9)},   "V", "less than number of clusters", id="No.100"),
        pytest.param({0: (0, 10)},  "V", "less than number of samples", id="No.101"),
        pytest.param((0, 10),       "T", "must be dict", id="No.102"),
        pytest.param({1.0: (0, 9)}, "T", "must be int", id="No.103"),
        pytest.param({0: (0, 9.0)}, "T", "must be int", id="No.104"),
        pytest.param({0: 0},        "T", "must be tuples containing indices of samples", id="No.105"),
        pytest.param({0: (0, 9)},   "A", "This instance is not fitted yet", id="No.106"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.341"),
        pytest.param({0: (0, 1, 2, 3, 4, 5, 6), 1: (7, 8, 9)}, "R", "The number of clusters returned by an optimization solver is less than the specified n_clusters.", id="No.342"),
    ]
)
def test_set_partition_level_by_qbits_reduction_error(
    partition_level_instances, e, err_msg
):
    data, _ = make_blobs(random_state=8, n_samples=10, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl = ConstrainedClustering(n_clusters=3)
    if (e == "V"):
        ccl.fit(data)
        with pytest.raises(ValueError) as e:
            ccl.set_partition_level_by_qbits_reduction(
                partition_level_instances
            )
    elif (e == "T"):
        ccl.fit(data)
        with pytest.raises(TypeError) as e:
            ccl.set_partition_level_by_qbits_reduction(
                partition_level_instances
            )
    elif (e == "A"):
        with pytest.raises(AttributeError) as e:
            ccl.set_partition_level_by_qbits_reduction(
                partition_level_instances
            )
    elif (e == "R"):
        ccl.fit(data)
        ccl.set_partition_level_by_qbits_reduction(partition_level_instances)
        client = setup_fixstars()
        with pytest.raises(RuntimeError) as e:
            labels = ccl.predict(client)
    else:
        raise Exception("Invalid Error")
    assert(err_msg in str(e.value))

def _create_name_method_map(ccl):
    # a mapping from name to method and constraint key
    temp = {
        "add_must_link":      (ccl.add_must_link_to_qubo, "must_link_instances"),
        "add_cannot_link":    (ccl.add_cannot_link_to_qubo, "cannot_link_instances"),
        "add_partition":      (ccl.add_partition_level_to_qubo, "partition_level_instances"),
        "add_non_partition":  (ccl.add_non_partition_level_to_qubo, "non_partition_level_instances"),
        "add_limited":        (ccl.add_limited_sizes_to_qubo, "limited_sizes"),
        "add_balanced":       (ccl.add_balanced_sizes_to_qubo, "balanced_sizes"),
        "set_must_link":      (ccl.set_must_link_by_qbits_reduction, "must_link_instances"),
        "set_partition":      (ccl.set_partition_level_by_qbits_reduction, "partition_level_instances"),
        "set_non_partition":  (ccl.set_non_partition_level_by_qbits_reduction, "non_partition_level_instances"),
    }
    return temp

@pytest.mark.parametrize(
    ["No", "Case", "add_method"],
    [
        pytest.param("107", "Init", None, id="No.107"),
        pytest.param("108", "Reset", True, id="No.108"),
        pytest.param("109", "Reset", True, id="No.109"),
        pytest.param("110", "Reset", True, id="No.110"),
        pytest.param("111", "Reset", True, id="No.111"),
        pytest.param("112", "Reset", True, id="No.112"),
        pytest.param("113", "Reset", True, id="No.113"),
        pytest.param("114", "Reset", False, id="No.114"),
        pytest.param("115", "Reset", False, id="No.115"),
        pytest.param("116", "Reset", False, id="No.116"),
    ]
)
def test_fit_attributes(No, Case, add_method, create_constraint):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)

    default_constraints = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }

    if Case == "Init":
        assert ccl.constraints == default_constraints
        assert ccl.add_constraint_flag == False
        assert ccl.set_constraint_flag == False
    elif Case == "Reset":
        given = create_constraint.get(No)
        for name, content in given.items():
            if name == "must_link_instances":
                if add_method:
                    ccl.add_must_link_to_qubo(content)
                else:
                    ccl.set_must_link_by_qbits_reduction(content)
            elif name == "cannot_link_instances":
                ccl.add_cannot_link_to_qubo(content)
            elif name == "partition_level_instances":
                if add_method:
                    ccl.add_partition_level_to_qubo(content)
                else:
                    ccl.set_partition_level_by_qbits_reduction(content)
            elif name == "non_partition_level_instances":
                if add_method:
                    ccl.add_non_partition_level_to_qubo(content)
                else:
                    ccl.set_non_partition_level_by_qbits_reduction(content)
            elif name == "limited_sizes":
                ccl.add_limited_sizes_to_qubo(content)
            elif name == "balanced_sizes":
                ccl.add_balanced_sizes_to_qubo()
            else:
                raise Exception("Invalid constraint name")
            
            expected_constraints = default_constraints.copy()
            expected_constraints[name] = content
        
        assert ccl.constraints == expected_constraints
        if add_method:
            assert ccl.add_constraint_flag == True
            assert ccl.set_constraint_flag == False
        else:
            assert ccl.add_constraint_flag == False
            assert ccl.set_constraint_flag == True

        data2, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                    cluster_std=1.5, centers=n_clusters)
        ccl.fit(data2)
        assert ccl.constraints == default_constraints
        assert ccl.add_constraint_flag == False
        assert ccl.set_constraint_flag == False
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["No", "Case"],
    [
        pytest.param("117", "set_must_link", id="No.117"),
        pytest.param("118", "set_must_link", id="No.118"),
        pytest.param("119", "set_must_link", id="No.119"),
        pytest.param("120", "set_must_link", id="No.120"),
        pytest.param("121", "set_must_link", id="No.121"),
        pytest.param("122", "set_must_link", id="No.122"),
        pytest.param("123", "set_partition", id="No.123"),
        pytest.param("124", "set_partition", id="No.124"),
        pytest.param("125", "set_partition", id="No.125"),
        pytest.param("126", "set_partition", id="No.126"),
        pytest.param("127", "set_partition", id="No.127"),
        pytest.param("128", "set_partition", id="No.128"),
        pytest.param("129", "set_non_partition", id="No.129"),
        pytest.param("130", "set_non_partition", id="No.130"),
        pytest.param("131", "set_non_partition", id="No.131"),
        pytest.param("132", "set_non_partition", id="No.132"),
        pytest.param("133", "set_non_partition", id="No.133"),
        pytest.param("134", "set_non_partition", id="No.134"),
        pytest.param("135", "add_must_link", id="No.135"),
        pytest.param("136", "add_partition", id="No.136"),
        pytest.param("137", "add_non_partition", id="No.137"),
    ]
)
def test_validate_add_set_exclusivity_error(No, Case, create_constraint):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)

    given = create_constraint.get(No)

    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)
    try:
        method, key = case_method_map[Case]
        if key in given:
            method(given[key])
        else:
            raise Exception(f"Constraint key '{key}' not found in given constraints.")
    except KeyError:
        raise Exception("Invalid Case")
    
    with pytest.raises(ValueError) as e:
        for name, content in given.items():
            if name == "must_link_instances":
                if "set_" in Case:
                    ccl.add_must_link_to_qubo(content)
                else:
                    ccl.set_must_link_by_qbits_reduction(content)
            elif name == "cannot_link_instances":
                ccl.add_cannot_link_to_qubo(content)
            elif name == "partition_level_instances":
                if "set_" in Case:
                    ccl.add_partition_level_to_qubo(content)
                else:
                    ccl.set_partition_level_by_qbits_reduction(content)
            elif name == "non_partition_level_instances":
                if "set_" in Case:
                    ccl.add_non_partition_level_to_qubo(content)
                else:
                    ccl.set_non_partition_level_by_qbits_reduction(content)
            elif name == "limited_sizes":
                ccl.add_limited_sizes_to_qubo(content)
            elif name == "balanced_sizes":
                ccl.add_balanced_sizes_to_qubo()
            else:
                raise Exception("Invalid constraint name")
    
        assert name in str(e) and "cannot be used at the same time." in str(e)

@pytest.mark.parametrize(
    ["No", "Case"],
    [
        pytest.param("138", "add", id="No.138"),
        pytest.param("139", "set", id="No.139"),
    ]
)
def test_validate_add_set_exclusivity_normal(No, Case):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)

    if Case == "add":
        ccl.add_cannot_link_to_qubo([(2, 3)])
        ccl.add_must_link_to_qubo([(0, 1)])
        expected = {
            "must_link_instances": [(0, 1)],
            "cannot_link_instances": [(2, 3)],
            "partition_level_instances": None,
            "non_partition_level_instances": None,
            "limited_sizes": None,
            "balanced_sizes": None,
        }

        assert ccl.constraints == expected
        assert ccl.add_constraint_flag == True
        assert ccl.set_constraint_flag == False
    elif Case == "set":
        ccl.set_partition_level_by_qbits_reduction({0: (0, 1)})
        ccl.set_non_partition_level_by_qbits_reduction({1: (2, 3)})
        expected = {
            "must_link_instances": None,
            "cannot_link_instances": None,
            "partition_level_instances": {0: (0, 1)},
            "non_partition_level_instances":  {1: (2, 3)},
            "limited_sizes": None,
            "balanced_sizes": None,
        }

        assert ccl.constraints == expected
        assert ccl.add_constraint_flag == False
        assert ccl.set_constraint_flag == True
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["No", "Case"],
    [
        pytest.param("140", "limited", id="No.140"),
        pytest.param("141", "balanced", id="No.141"),
    ]
)
def test_limited_balanced_error(No, Case):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)

    err_msg = "limited_sizes constraint and balanced_sizes constraint cannot be specified at the same time."
    if Case == "limited":
        ccl.add_limited_sizes_to_qubo({0: 3, 1: 4, 2: 3})
        with pytest.raises(ValueError) as e:
            ccl.add_balanced_sizes_to_qubo()
        assert err_msg in str(e.value)
    elif Case == "balanced":
        ccl.add_balanced_sizes_to_qubo()
        with pytest.raises(ValueError) as e:
            ccl.add_limited_sizes_to_qubo({0: 3, 1: 4, 2: 3})
        assert err_msg in str(e.value)
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["No", "Case", "constraint_1", "constraint_2"],
    [
        pytest.param("142", "add_must_link", [(0, 1), (2, 3)], [(4, 6), (8, 9)], id="No.142"),
        pytest.param("142", "add_must_link", [(0, 1), (2, 3)], [(2, 3), (4, 6), (8, 9)], id="additional142"),
        pytest.param("143", "add_cannot_link", [(1, 0), (3, 2)], [(6, 4), (9, 8)], id="No.143"),
        pytest.param("143", "add_cannot_link", [(1, 0), (3, 2)], [(6, 4), (9, 8), (1, 0)], id="additional143"),
        pytest.param("144", "add_partition", {0: (0,1)}, {1: (2,3), 2: (4, 5, 6)}, id="No.144"),
        pytest.param("145", "add_non_partition",  {0: (0,1), 1: (2, 3), 2: (4, 5, 6)}, {1: (2, 3, 7), 2: (8, )}, id="No.145"),
        pytest.param("146", "add_limited", {0: 5, 1: 3, 2: 3}, {0: 4, 1: 3, 2: 4}, id="No.146"),
        pytest.param("306", "add_limited", {0: 1, 1: 0, 2: 10}, {0: 0, 1: 1, 2: 10}, id="No.306"),
        pytest.param("147", "add_balanced", None, None, id="No.147"),
        pytest.param("148", "set_must_link", [(0, 1), (2, 3)], [(4, 6), (8, 9)], id="No.148"),
        pytest.param("149", "set_partition", {0: (0,1)}, {1: (2,3), 2: (4, 5, 6)}, id="No.149"),
        pytest.param("150", "set_non_partition",  {0: (0,1), 1: (2, 3), 2: (4, 5, 6)}, {1: (2, 3, 7), 2: (8, )}, id="No.150"),
    ]
)
def test_merge_constraints(No, Case, constraint_1, constraint_2):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)

    if Case == "add_must_link":
        ccl.add_must_link_to_qubo(constraint_1)
        ccl.add_must_link_to_qubo(constraint_2)
        expected = set(constraint_1 + constraint_2)
        assert set(ccl.constraints["must_link_instances"]) == expected
    elif Case == "add_cannot_link":
        ccl.add_cannot_link_to_qubo(constraint_1)
        ccl.add_cannot_link_to_qubo(constraint_2)
        expected = set([tuple(sorted(c)) for c in constraint_1] + [tuple(sorted(c)) for c in constraint_2])
        assert set(ccl.constraints["cannot_link_instances"]) == expected
    elif Case == "add_partition":
        ccl.add_partition_level_to_qubo(constraint_1)
        ccl.add_partition_level_to_qubo(constraint_2)
        assert ccl.constraints["partition_level_instances"] == {0: (0, 1), 1: (2, 3), 2: (4,5,6)}
    elif Case == "add_non_partition":
        ccl.add_non_partition_level_to_qubo(constraint_1)
        ccl.add_non_partition_level_to_qubo(constraint_2)
        assert ccl.constraints["non_partition_level_instances"] == {0: (0, 1), 1: (2, 3, 7), 2: (4,5,6,8)}
    elif Case == "add_limited":
        ccl.add_limited_sizes_to_qubo(constraint_1)
        ccl.add_limited_sizes_to_qubo(constraint_2)
        if No == "146":
            assert ccl.constraints["limited_sizes"] == {0: 4, 1: 3, 2: 3}
        else:
            assert ccl.constraints["limited_sizes"] == {0: 0, 1: 0, 2: 10}
    elif Case == "add_balanced":
        ccl.add_balanced_sizes_to_qubo()
        assert ccl.constraints["balanced_sizes"] == (4, 3)

        data2, _ = make_blobs(random_state=8, n_samples=15, n_features=2,
                            cluster_std=1.5, centers=n_clusters)
        ccl.fit(data2)
        ccl.add_balanced_sizes_to_qubo()
        assert ccl.constraints["balanced_sizes"] == (5, )

        ccl.add_balanced_sizes_to_qubo()
        assert ccl.constraints["balanced_sizes"] == (5, )
    elif Case == "set_must_link":
        ccl.set_must_link_by_qbits_reduction(constraint_1)
        ccl.set_must_link_by_qbits_reduction(constraint_2)
        expected = set(constraint_1 + constraint_2)
        assert set(ccl.constraints["must_link_instances"]) == expected
    elif Case == "set_partition":
        ccl.set_partition_level_by_qbits_reduction(constraint_1)
        ccl.set_partition_level_by_qbits_reduction(constraint_2)
        assert ccl.constraints["partition_level_instances"] == {0: (0, 1), 1: (2, 3), 2: (4,5,6)}
    elif Case == "set_non_partition":
        ccl.set_non_partition_level_by_qbits_reduction(constraint_1)
        ccl.set_non_partition_level_by_qbits_reduction(constraint_2)
        assert ccl.constraints["non_partition_level_instances"] == {0: (0, 1), 1: (2, 3, 7), 2: (4,5,6,8)}

@pytest.mark.parametrize(
    ["No", "Case", "name"],
    [
        pytest.param("151", "one_hot", "add_must_link",     id="No.151"),
        pytest.param("152", "one_hot", "add_cannot_link",   id="No.152"),
        pytest.param("153", "one_hot", "add_partition",     id="No.153"),
        pytest.param("154", "one_hot", "add_non_partition", id="No.154"),
        pytest.param("155", "one_hot", "add_limited",       id="No.155"),
        pytest.param("156", "one_hot", "add_balanced",      id="No.156"),
        pytest.param("157", "one_hot", "set_must_link",     id="No.157"),
        pytest.param("158", "one_hot", "set_partition",     id="No.158"),
        pytest.param("159", "one_hot", "set_non_partition", id="No.159"),
        pytest.param("160", "greater", "add_must_link",     id="No.160"),
        pytest.param("161", "greater", "add_cannot_link",   id="No.161"),
        pytest.param("162", "greater", "add_partition",     id="No.162"),
        pytest.param("163", "greater", "add_non_partition", id="No.163"),
        pytest.param("164", "greater", "add_limited",       id="No.164"),
        pytest.param("165", "greater", "add_balanced",      id="No.165"),
        pytest.param("166", "greater", "set_must_link",     id="No.166"),
        pytest.param("167", "greater", "set_partition",     id="No.167"),
        pytest.param("168", "greater", "set_non_partition", id="No.168"),
        pytest.param("169", "less",    "add_must_link",     id="No.169"),
        pytest.param("170", "less",    "add_cannot_link",   id="No.170"),
        pytest.param("171", "less",    "add_partition",     id="No.171"),
        pytest.param("172", "less",    "add_non_partition", id="No.172"),
        pytest.param("173", "less",    "add_limited",       id="No.173"),
        pytest.param("174", "less",    "add_balanced",      id="No.174"),
        pytest.param("175", "less",    "set_must_link",     id="No.175"),
        pytest.param("176", "less",    "set_partition",     id="No.176"),
        pytest.param("177", "less",    "set_non_partition", id="No.177"),
    ]
)
def test_predict_constraint_violation(mocker, No, Case, name, create_constraint):
    n_samples = 10
    n_clusters = 3
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    if Case == "one_hot":
        labels_dummy = [0, 0, 1, 1, 2, -1, 0, 1, 2, 0]
        err_msg = "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1."
    elif Case == "greater":
        labels_dummy = [0, 0, 1, 1, 2, 4, 0, 1, 2, 0]
        err_msg = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    elif Case == "less":
        labels_dummy = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        err_msg = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    else:
        raise Exception("Invalid Case")
    
    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)
    given = create_constraint.get(No)
    try:
        method, key = case_method_map[name]
        if key in given:
            if name == "add_balanced":
                method()
            else:
                method(given[key])
        else:
            raise Exception(f"Constraint key '{key}' not found in given constraints.")
    except KeyError:
        raise Exception("Invalid constraint name")

    with pytest.raises(RuntimeError) as e:
        mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
        ccl.predict(client)
    assert err_msg in str(e)

@pytest.mark.parametrize(
    ["No", "Case", "names", "option"],
    [
        pytest.param("178", "normal",   None, None, id="No.178"),
        pytest.param("179", "normal",   ["add_must_link", "add_cannot_link"], None, id="No.179"),
        pytest.param("180", "abnormal", ["add_must_link", "add_cannot_link"], None, id="No.180"),
        pytest.param("181", "normal",   ["add_must_link", "add_cannot_link"], None, id="No.181"),
        pytest.param("182", "abnormal", ["add_must_link", "add_cannot_link"], None, id="No.182"),
        pytest.param("183", "normal",   ["add_partition", "add_must_link"], None, id="No.183"),
        pytest.param("184", "abnormal", ["add_partition", "add_must_link"], None, id="No.184"),
        pytest.param("185", "normal",   ["add_partition", "add_must_link"], None, id="No.185"),
        pytest.param("186", "abnormal", ["add_partition", "add_must_link"], None, id="No.186"),
        pytest.param("187", "normal",   ["add_partition", "add_cannot_link"], None, id="No.187"),
        pytest.param("188", "abnormal", ["add_partition", "add_cannot_link"], None, id="No.188"),
        pytest.param("189", "normal",   ["add_partition", "add_non_partition"], None, id="No.189"),
        pytest.param("190", "abnormal", ["add_partition", "add_non_partition"], None, id="No.190"),
        pytest.param("191", "normal",   ["add_partition", "add_limited"], None, id="No.191"),
        pytest.param("192", "abnormal", ["add_partition", "add_limited"], None, id="No.192"),
        pytest.param("193", "normal",   ["add_partition", "add_balanced"], "divisible", id="No.193"),
        pytest.param("194", "abnormal", ["add_partition", "add_balanced"], "divisible", id="No.194"),
        pytest.param("195", "normal",   ["add_partition", "add_balanced"], None, id="No.195"),
        pytest.param("196", "abnormal", ["add_partition", "add_balanced"], None, id="No.196"),
        pytest.param("197", "normal",   ["add_partition", "add_must_link", "add_cannot_link"], None, id="No.197"),
        pytest.param("198", "abnormal", ["add_partition", "add_must_link", "add_cannot_link"], None, id="No.198"),
        pytest.param("201", "normal",   ["set_partition", "set_must_link"], None, id="No.201"),
        pytest.param("202", "abnormal", ["set_partition", "set_must_link"], None, id="No.202"),
        pytest.param("203", "normal",   ["set_partition", "set_must_link"], None, id="No.203"),
        pytest.param("204", "abnormal", ["set_partition", "set_must_link"], None, id="No.204"),
        pytest.param("205", "normal",   ["set_partition", "set_non_partition"], None, id="No.205"),
        pytest.param("206", "abnormal", ["set_partition", "set_non_partition"], None, id="No.206"),
        pytest.param("307", "normal",   ["add_partition", "add_limited"], None, id="No.307"),
        pytest.param("308", "abnormal", ["add_partition", "add_limited"], None, id="No.308"),
        pytest.param("309", "normal",   ["add_partition", "add_limited"], "four", id="No.309"),
        pytest.param("310", "abnormal", ["add_partition", "add_limited"], "four", id="No.310"),
        pytest.param("359", "abnormal",   ["add_partition", "add_limited"], None, id="No.359"),
        pytest.param("360", "abnormal", ["add_partition", "add_limited"], None, id="No.360"),
        pytest.param("361", "normal",   ["add_partition", "add_limited"], None, id="No.361"),
        pytest.param("362", "abnormal", ["add_partition", "add_limited"], None, id="No.362"),
        pytest.param("363", "abnormal",   ["add_partition", "add_non_partition"], None, id="No.363"),
        pytest.param("364", "abnormal", ["add_partition", "add_non_partition"], None, id="No.364"),
        pytest.param("365", "abnormal",   ["add_partition", "add_non_partition"], None, id="No.365"),
        pytest.param("366", "abnormal", ["add_partition", "add_non_partition"], None, id="No.366"),
    ]
)
def test_predict_conflict_constraints(No, Case, names, option, create_constraint, create_err_msg):
    n_samples = 10
    if option == "divisible":
        n_clusters = 2
    elif option == "four":
        n_clusters = 4
    else:
        n_clusters = 3
    
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    if names is None:
        ccl.predict(client)
        return

    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)
    given = create_constraint.get(No)
    for name in names:
        try:
            method, key = case_method_map[name]
            if key in given:
                if name == "add_balanced":
                    method()
                else:
                    method(given[key])
            else:
                raise Exception(f"Constraint key '{key}' not found in given constraints.")
        except KeyError:
            raise Exception("Invalid constraint name")

    if Case == "normal":
        try:
            ccl.predict(client)
        except RuntimeError as e:
            if "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1." in str(e):
                assert True
            else:
                assert False
        except Exception as e:
            raise e
    elif Case == "abnormal":
        with pytest.raises(RuntimeError) as e:
            ccl.predict(client)
        keywords = create_err_msg.get(No)
        assert all(k in str(e) for k in keywords)
    else:
        raise Exception("Invalid Case")

@pytest.mark.parametrize(
    ["No", "Case"],
    [
        pytest.param("199", "abnormal1", id="No.199"),
        pytest.param("200", "abnormal2", id="No.200"),
    ]
)
def test_predict_abnormal_partition_level(No, Case, create_err_msg):
    n_samples = 10
    n_clusters = 3
    
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    if Case == "abnormal1":
        ccl.add_partition_level_to_qubo({0: (0, 1, 2), 2:(0, 3, 4)})
    elif Case == "abnormal2":
        ccl.add_partition_level_to_qubo({0: (0, 1)})
        ccl.add_partition_level_to_qubo({1:(1, 2)})
    
    with pytest.raises(ValueError) as e:
        labels = ccl.predict(client)
    err_msg = create_err_msg.get(No)
    assert err_msg in str(e)

@pytest.mark.parametrize(
    ["No", "Case", "name", "option"],
    [
        pytest.param("207", "abnormal", "add_must_link",     None, id="No.207"),
        pytest.param("208", "abnormal", "add_cannot_link",   None, id="No.208"),
        pytest.param("209", "abnormal", "add_partition",     None, id="No.209"),
        pytest.param("210", "abnormal", "add_non_partition", None, id="No.210"),
        pytest.param("211", "abnormal", "add_limited",       None, id="No.211"),
        pytest.param("212", "abnormal", "add_balanced",      "divisible", id="No.212"),
        pytest.param("213", "normal",   "add_balanced",      "indivisible", id="No.213"),
        pytest.param("214", "abnormal", "add_balanced",      "indivisible", id="No.214"),
        pytest.param("215", "abnormal", "set_must_link",     None, id="No.215"),
        pytest.param("216", "abnormal", "set_partition",     None, id="No.216"),
        pytest.param("217", "abnormal", "set_non_partition", None, id="No.217"),
        pytest.param("311", "abnormal", "add_limited",       None, id="No.311"),
        pytest.param("312", "abnormal", "add_limited",       "four", id="No.312"),
    ]
)
def test_predict_conflict_labels_constraints(mocker, No, Case, name, option, create_test_data, create_err_msg):
    n_samples = 10
    if option == "divisible":
        n_clusters = 2
    elif option == "four":
        n_clusters = 4
    else:
        n_clusters = 3
    
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    test_data = create_test_data.get(No)
    labels_dummy = test_data["labels"]
    constraint = test_data["const1"]

    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)

    try:
        method, _ = case_method_map[name]
        if name == "add_balanced":
            method()
        else:
            method(constraint)
    except KeyError:
        raise Exception("Invalid constraint name")
    
    if Case == "abnormal":
        with pytest.raises(RuntimeError) as e:
            mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
            ccl.predict(client)
        keywords = create_err_msg.get(No)
        assert all(k in str(e) for k in keywords)
    elif Case == "normal":
        try:
            ccl.predict(client)
        except RuntimeError as e:
            if "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1." in str(e):
                assert True
            else:
                assert False
        except Exception as e:
            raise e

@pytest.mark.parametrize(
    ["No", "Case", "name", "option"],
    [
        pytest.param("218", "normal",   "add_must_link", None, id="No.218"),
        pytest.param("219", "abnormal", "add_must_link", None, id="No.219"),
        pytest.param("220", "normal",   "add_must_link", None, id="No.220"),
        pytest.param("221", "abnormal", "add_must_link", None, id="No.221"),
        pytest.param("222", "normal",   "add_cannot_link", None, id="No.222"),
        pytest.param("223", "abnormal", "add_cannot_link", None, id="No.223"),
        pytest.param("224", "normal",   "add_partition", None, id="No.224"),
        pytest.param("225", "abnormal", "add_partition", None, id="No.225"),
        pytest.param("226", "normal",   "add_non_partition", None, id="No.226"),
        pytest.param("227", "abnormal", "add_non_partition", None, id="No.227"),
        pytest.param("228", "normal",   "add_limited", None, id="No.228"),
        pytest.param("229", "abnormal", "add_limited", None, id="No.229"),
        pytest.param("230", "normal",   "add_balanced", "divisible", id="No.230"),
        pytest.param("231", "abnormal", "add_balanced", "divisible", id="No.231"),
        pytest.param("232", "normal",   "add_balanced", "indivisible", id="No.232"),
        pytest.param("233", "abnormal", "add_balanced", "indivisible", id="No.233"),
        pytest.param("234", "normal",   "set_must_link", None, id="No.234"),
        pytest.param("235", "abnormal", "set_must_link", None, id="No.235"),
        pytest.param("236", "normal",   "set_must_link", None, id="No.236"),
        pytest.param("237", "abnormal", "set_must_link", None, id="No.237"),
        pytest.param("238", "normal",   "set_partition", None, id="No.238"),
        pytest.param("239", "abnormal", "set_partition", None, id="No.239"),
        pytest.param("240", "normal",   "set_non_partition", None, id="No.240"),
        pytest.param("241", "abnormal", "set_non_partition", None, id="No.241"),
        pytest.param("313", "normal",   "add_limited", None, id="No.313"),
        pytest.param("314", "abnormal", "add_limited", None, id="No.314"),
        pytest.param("345", "abnormal",   "add_partition", None, id="No.345"),
        pytest.param("346", "abnormal",   "add_partition", None, id="No.346"),
        pytest.param("347", "abnormal",   "add_non_partition", None, id="No.347"),
    ]
)
def test_predict_conflict_labels_merged_constraints(mocker, No, Case, name, option, create_test_data, create_err_msg):
    n_samples = 10
    if option == "divisible":
        n_clusters = 2
    else:
        n_clusters = 3
    
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    test_data = create_test_data.get(No)
    labels_dummy = test_data["labels"]
    constraint_1 = test_data["const1"]
    constraint_2 = test_data["const2"]

    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)
    try:
        method, _ = case_method_map[name]
    except KeyError:
        raise Exception("Invalid constraint name")

    if option is None:
        method(constraint_1)
        method(constraint_2)
    else:
        method()
        method()

    if Case == "abnormal":
        with pytest.raises(RuntimeError) as e:
            mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
            ccl.predict(client)
        keywords = create_err_msg.get(No)
        assert all(k in str(e) for k in keywords)
    elif Case == "normal":
        try:
            if labels_dummy is not None:
                mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
            ccl.predict(client)
        except RuntimeError as e:
            if "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1." in str(e):
                assert True
            else:
                assert False
        except Exception as e:
            raise e

@pytest.mark.parametrize(
    ["No", "Case", "names", "option"],
    [
        pytest.param("242", "normal",   ["add_must_link", "add_cannot_link"],   None, id="No.242"),
        pytest.param("243", "abnormal", ["add_must_link", "add_cannot_link"],   None, id="No.243"),
        pytest.param("244", "abnormal", ["add_must_link", "add_cannot_link"],   None, id="No.244"),
        pytest.param("245", "normal",   ["add_must_link", "add_non_partition"], None, id="No.245"),
        pytest.param("246", "abnormal", ["add_must_link", "add_non_partition"], None, id="No.246"),
        pytest.param("247", "abnormal", ["add_must_link", "add_non_partition"], None, id="No.247"),
        pytest.param("248", "normal",   ["add_must_link", "add_limited"],       None, id="No.248"),
        pytest.param("249", "abnormal", ["add_must_link", "add_limited"],       None, id="No.249"),
        pytest.param("250", "abnormal", ["add_must_link", "add_limited"],       None, id="No.250"),
        pytest.param("251", "normal",   ["add_must_link", "add_balanced"],      "divisible",   id="No.251"),
        pytest.param("252", "abnormal", ["add_must_link", "add_balanced"],      "divisible",   id="No.252"),
        pytest.param("253", "abnormal", ["add_must_link", "add_balanced"],      "divisible",   id="No.253"),
        pytest.param("254", "normal",   ["add_must_link", "add_balanced"],      "indivisible", id="No.254"),
        pytest.param("255", "abnormal", ["add_must_link", "add_balanced"],      "indivisible", id="No.255"),
        pytest.param("256", "abnormal", ["add_must_link", "add_balanced"],      "indivisible", id="No.256"),
        pytest.param("257", "normal",   ["add_cannot_link", "add_non_partition"], None,          id="No.257"),
        pytest.param("258", "abnormal", ["add_cannot_link", "add_non_partition"], None,          id="No.258"),
        pytest.param("259", "abnormal", ["add_cannot_link", "add_non_partition"], None,          id="No.259"),
        pytest.param("260", "normal",   ["add_cannot_link", "add_limited"],       None,          id="No.260"),
        pytest.param("261", "abnormal", ["add_cannot_link", "add_limited"],       None,          id="No.261"),
        pytest.param("262", "abnormal", ["add_cannot_link", "add_limited"],       None,          id="No.262"),
        pytest.param("263", "normal",   ["add_cannot_link", "add_balanced"],      "divisible",   id="No.263"),
        pytest.param("264", "abnormal", ["add_cannot_link", "add_balanced"],      "divisible",   id="No.264"),
        pytest.param("265", "abnormal", ["add_cannot_link", "add_balanced"],      "divisible",   id="No.265"),
        pytest.param("266", "normal",   ["add_cannot_link", "add_balanced"],      "indivisible", id="No.266"),
        pytest.param("267", "abnormal", ["add_cannot_link", "add_balanced"],      "indivisible", id="No.267"),
        pytest.param("268", "abnormal", ["add_cannot_link", "add_balanced"],      "indivisible", id="No.268"),
        pytest.param("269", "normal",   ["add_non_partition", "add_limited"],  None,          id="No.269"),
        pytest.param("270", "abnormal", ["add_non_partition", "add_limited"],  None,          id="No.270"),
        pytest.param("271", "abnormal", ["add_non_partition", "add_limited"],  None,          id="No.271"),
        pytest.param("272", "normal",   ["add_non_partition", "add_balanced"], "divisible",   id="No.272"),
        pytest.param("273", "abnormal", ["add_non_partition", "add_balanced"], "divisible",   id="No.273"),
        pytest.param("274", "abnormal", ["add_non_partition", "add_balanced"], "divisible",   id="No.274"),
        pytest.param("275", "normal",   ["add_non_partition", "add_balanced"], "indivisible", id="No.275"),
        pytest.param("276", "abnormal", ["add_non_partition", "add_balanced"], "indivisible", id="No.276"),
        pytest.param("277", "abnormal", ["add_non_partition", "add_balanced"], "indivisible", id="No.277"),
        pytest.param("278", "normal",   ["add_partition", "add_must_link"],     None,          id="No.278"),
        pytest.param("279", "abnormal", ["add_partition", "add_must_link"],     None,          id="No.279"),
        pytest.param("280", "abnormal", ["add_partition", "add_must_link"],     None,          id="No.280"),
        pytest.param("281", "normal",   ["add_partition", "add_cannot_link"],   None,          id="No.281"),
        pytest.param("282", "abnormal", ["add_partition", "add_cannot_link"],   None,          id="No.282"),
        pytest.param("283", "abnormal", ["add_partition", "add_cannot_link"],   None,          id="No.283"),
        pytest.param("284", "normal",   ["add_partition", "add_non_partition"], None,          id="No.284"),
        pytest.param("285", "abnormal", ["add_partition", "add_non_partition"], None,          id="No.285"),
        pytest.param("286", "abnormal", ["add_partition", "add_non_partition"], None,          id="No.286"),
        pytest.param("287", "normal",   ["add_partition", "add_limited"],       None,          id="No.287"),
        pytest.param("288", "abnormal", ["add_partition", "add_limited"],       None,          id="No.288"),
        pytest.param("289", "abnormal", ["add_partition", "add_limited"],       None,          id="No.289"),
        pytest.param("290", "normal",   ["add_partition", "add_balanced"],      "divisible",   id="No.290"),
        pytest.param("291", "abnormal", ["add_partition", "add_balanced"],      "divisible",   id="No.291"),
        pytest.param("292", "abnormal", ["add_partition", "add_balanced"],      "divisible",   id="No.292"),
        pytest.param("293", "normal",   ["add_partition", "add_balanced"],      "indivisible", id="No.293"),
        pytest.param("294", "abnormal", ["add_partition", "add_balanced"],      "indivisible", id="No.294"),
        pytest.param("295", "abnormal", ["add_partition", "add_balanced"],      "indivisible", id="No.295"),
        pytest.param("296", "normal",   ["set_partition", "set_non_partition"], None,          id="No.296"),
        pytest.param("297", "abnormal", ["set_partition", "set_non_partition"], None,          id="No.297"),
        pytest.param("298", "abnormal", ["set_partition", "set_non_partition"], None,          id="No.298"),
        pytest.param("299", "normal",   ["set_partition", "set_must_link"],     None,          id="No.299"),
        pytest.param("300", "abnormal", ["set_partition", "set_must_link"],     None,          id="No.300"),
        pytest.param("301", "abnormal", ["set_partition", "set_must_link"],     None,          id="No.301"),
        pytest.param("302", "normal",   ["set_non_partition", "set_must_link"], None,          id="No.302"),
        pytest.param("303", "abnormal", ["set_non_partition", "set_must_link"], None,          id="No.303"),
        pytest.param("304", "abnormal", ["set_non_partition", "set_must_link"], None,          id="No.304"),
        pytest.param("315", "normal",   ["add_must_link", "add_limited"],     None, id="No.315"),
        pytest.param("316", "abnormal", ["add_must_link", "add_limited"],     None, id="No.316"),
        pytest.param("317", "abnormal", ["add_must_link", "add_limited"],     None, id="No.317"),
        pytest.param("318", "normal",   ["add_cannot_link", "add_limited"],   None, id="No.318"),
        pytest.param("319", "abnormal", ["add_cannot_link", "add_limited"],   None, id="No.319"),
        pytest.param("320", "abnormal", ["add_cannot_link", "add_limited"],   None, id="No.320"),
        pytest.param("321", "normal",   ["add_partition", "add_limited"],     None, id="No.321"),
        pytest.param("322", "abnormal", ["add_partition", "add_limited"],     None, id="No.322"),
        pytest.param("323", "abnormal", ["add_partition", "add_limited"],     None, id="No.323"),
        pytest.param("324", "normal",   ["add_non_partition", "add_limited"], None, id="No.324"),
        pytest.param("325", "abnormal", ["add_non_partition", "add_limited"], None, id="No.325"),
        pytest.param("326", "abnormal", ["add_non_partition", "add_limited"], None, id="No.326"),
        pytest.param("327", "normal",   ["add_must_link", "add_limited"],     "four", id="No.327"),
        pytest.param("328", "abnormal", ["add_must_link", "add_limited"],     "four", id="No.328"),
        pytest.param("329", "abnormal", ["add_must_link", "add_limited"],     "four", id="No.329"),
        pytest.param("330", "normal",   ["add_cannot_link", "add_limited"],   "four", id="No.330"),
        pytest.param("331", "abnormal", ["add_cannot_link", "add_limited"],   "four", id="No.331"),
        pytest.param("332", "abnormal", ["add_cannot_link", "add_limited"],   "four", id="No.332"),
        pytest.param("333", "normal",   ["add_partition", "add_limited"],     "four", id="No.333"),
        pytest.param("334", "abnormal", ["add_partition", "add_limited"],     "four", id="No.334"),
        pytest.param("335", "abnormal", ["add_partition", "add_limited"],     "four", id="No.335"),
        pytest.param("336", "normal",   ["add_non_partition", "add_limited"], "four", id="No.336"),
        pytest.param("337", "abnormal", ["add_non_partition", "add_limited"], "four", id="No.337"),
        pytest.param("338", "abnormal", ["add_non_partition", "add_limited"], "four", id="No.338"),
        pytest.param("348", "abnormal",   ["add_partition", "add_limited"], None, id="No.348"),
        pytest.param("349", "abnormal", ["add_partition", "add_limited"], None, id="No.349"),
        pytest.param("350", "abnormal",   ["add_partition", "add_limited"], None, id="No.350"),
        pytest.param("351", "abnormal", ["add_partition", "add_limited"], None, id="No.351"),
        pytest.param("352", "abnormal",   ["add_partition", "add_non_partition"], None, id="No.352"),
        pytest.param("353", "abnormal", ["add_partition", "add_non_partition"], None, id="No.353"),
        pytest.param("354", "abnormal",   ["add_partition", "add_non_partition"], None, id="No.354"),
        pytest.param("355", "abnormal", ["add_partition", "add_non_partition"], None, id="No.355"),
        pytest.param("356", "abnormal",   ["add_non_partition", "add_limited"], None, id="No.356"),
        pytest.param("357", "abnormal", ["add_non_partition", "add_limited"], None, id="No.357"),
        pytest.param("358", "abnormal", ["add_non_partition", "add_limited"], None, id="No.358"),
    ]
)
def test_predict_conflict_labels_multi_constraints(mocker, No, Case, names, option, create_test_multi_data, create_err_msg):
    n_samples = 10
    if option == "divisible":
        n_clusters = 2
    elif option == "four":
        n_clusters = 4
    else:
        n_clusters = 3
    
    ccl = ConstrainedClustering(n_clusters=n_clusters)
    data, _ = make_blobs(random_state=8, n_samples=n_samples, n_features=2,
                         cluster_std=1.5, centers=n_clusters)
    ccl.fit(data)
    client = setup_fixstars()

    test_multi_data = create_test_multi_data.get(No)
    labels_dummy = test_multi_data["labels"]

    # Use a mapping from name to method and constraint key
    case_method_map = _create_name_method_map(ccl)

    for name in names:
        try:
            method, key = case_method_map[name]
            if key in test_multi_data:
                if name == "add_balanced":
                    method()
                else:
                    method(test_multi_data[key])
            else:
                raise Exception(f"Constraint key '{key}' not found in given constraints.")
        except KeyError:
            raise Exception("Invalid constraint name")
    
    if Case == "abnormal":
        with pytest.raises(RuntimeError) as e:
            if labels_dummy is not None:
                mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
            ccl.predict(client)
        keywords = create_err_msg.get(No)

        print("\n=== Exception details ===")
        traceback.print_exception(type(e.value), e.value, e.value.__traceback__)
        
        assert all(k in str(e) for k in keywords)
    elif Case == "normal":
        try:
            if labels_dummy is not None:
                mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
            ccl.predict(client)
        except RuntimeError as e:
            if "The cluster labels of the samples that do not satisfy the one-hot constraint are set to -1." in str(e):
                assert True
            else:
                assert False
        except Exception as e:
            raise e
@pytest.mark.parametrize(
    ["method_name", "labels_dummy", "const1", "const2"],
    [
        pytest.param("add", [1] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], None, id="No.367"),
        pytest.param("add", [1] * 6, [(0, 1), (1, 2), (2, 3)], [(3, 4), (4, 5)], id="No.368"),
        pytest.param("add", [1] * 4 + [2] * 2, [(0, 1), (1, 2), (2, 3), (4, 5)], None, id="No.369"),
        pytest.param("add", [1] * 4 + [2] * 2, [(0, 1), (1, 2), (2, 3)], [(4, 5)], id="No.370"),
        pytest.param("set", [1] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], None, id="No.371"),
        pytest.param("set", [1] * 4 + [2] * 2, [(0, 1), (1, 2), (2, 3), (4, 5)], None, id="No.372"),
    ]
)
def test_add_must_link_to_qubo_error_additional(mocker, method_name, labels_dummy, const1, const2):
    ccl = ConstrainedClustering(n_clusters=3)
    data, _ = make_blobs(random_state=8, n_samples=6, n_features=2,
                         cluster_std=1.5, centers=3)
    ccl.fit(data)
    client = setup_fixstars()

    if method_name == "add":
        ccl.add_must_link_to_qubo(const1)
        if const2 is not None:
            ccl.add_must_link_to_qubo(const2)
    elif method_name == "set":
        ccl.set_must_link_by_qbits_reduction(const1)
    else:
        raise Exception("Invalid method name")
    
    mocker.patch("qklearn.cluster.solution2labels", return_value=labels_dummy)
    with pytest.raises(RuntimeError) as e:
        ccl.predict(client)
    assert "The number of clusters returned by an optimization solver is less than the specified n_clusters." in str(e)