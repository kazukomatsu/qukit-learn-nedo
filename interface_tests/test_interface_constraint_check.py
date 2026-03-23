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
from qklearn.utils import constraint_check as cc

@pytest.fixture(scope="module")
def create_err_msg():
    err_msg = {}

    err_msg["2"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["3"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["5"]  = "The number of clusters returned by an optimization solver is less than the specified n_clusters."
    err_msg["6"]  = "The number of clusters returned by an optimization solver is greater than the specified n_clusters."
    err_msg["15"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["17"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["19"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["21"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["23"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["25"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["27"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["29"] = ["conflict", "specified " + "balanced-sizes", "and " + "partition-level"]
    err_msg["31"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["33"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["34"] = "There are samples whose cluster assignment is not uniquely determined."
    err_msg["36"] = ["does not satisfy", "must-link"]
    err_msg["38"] = ["does not satisfy", "cannot-link"]
    err_msg["40"] = ["does not satisfy", "partition-level"]
    err_msg["42"] = ["does not satisfy", "non-partition-level"]
    err_msg["44"] = ["does not satisfy", "limited-sizes"]
    err_msg["46"] = ["does not satisfy", "balanced-sizes"]
    err_msg["48"] = ["does not satisfy", "must-link"]
    err_msg["49"] = ["does not satisfy", "cannot-link"]
    err_msg["51"] = ["does not satisfy", "partition-level"]
    err_msg["52"] = ["does not satisfy", "must-link"]
    err_msg["54"] = ["does not satisfy", "partition-level"]
    err_msg["55"] = ["does not satisfy", "cannot-link"]
    err_msg["57"] = ["does not satisfy", "partition-level"]
    err_msg["58"] = ["does not satisfy", "non-partition-level"]
    err_msg["60"] = ["does not satisfy", "partition-level"]
    err_msg["61"] = ["does not satisfy", "limited-sizes"]
    err_msg["63"] = ["does not satisfy", "partition-level"]
    err_msg["64"] = ["does not satisfy", "balanced-sizes"]
    err_msg["68"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["70"] = ["does not satisfy", "limited-sizes"]

    err_msg["83"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["84"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["85"] = ["conflict", "specified " + "must-link", "and " + "cannot-link"]
    err_msg["additional86"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["additional87"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["88"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["89"] = ["conflict", "specified " + "must-link", "and " + "partition-level"]
    err_msg["91"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["93"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["95"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["97"] = ["conflict", "specified " + "cannot-link", "and " + "partition-level"]
    err_msg["99"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["101"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["103"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["105"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]

    err_msg["107"] = ["does not satisfy", "must-link"]
    err_msg["109"] = ["does not satisfy", "must-link"]
    err_msg["111"] = ["does not satisfy", "cannot-link"]
    err_msg["113"] = ["does not satisfy", "cannot-link"]
    err_msg["115"] = ["does not satisfy", "partition-level"]
    err_msg["117"] = ["does not satisfy", "partition-level"]
    err_msg["119"] = ["does not satisfy", "non-partition-level"]
    err_msg["121"] = ["does not satisfy", "non-partition-level"]

    err_msg["123"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["125"] = ["conflict", "specified " + "limited-sizes", "and " + "partition-level"]
    err_msg["127"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    err_msg["129"] = ["conflict", "specified " + "non-partition-level", "and " + "partition-level"]
    return err_msg


@pytest.fixture(scope="module")
def create_constraint():
    test_constraint = {}

    test_constraint["14"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 3)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["15"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 1)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["16"] = {
        "must_link_instances": [(0, 1), (2, 5), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 3)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["17"] = {
        "must_link_instances": [(0, 1), (2, 5), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 5)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["18"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["19"] =  {
        "must_link_instances": [(0, 1), (0, 2), (2, 3)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["20"] = {
        "must_link_instances": [(0, 1), (5, 8), (3, 4)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 8), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["21"] =  {
        "must_link_instances": [(0, 1), (5, 8), (2, 3)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 8), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["22"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["23"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 2)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["24"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (1, 2)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["25"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (3, 4, 7)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["26"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 3, 2: 3},
        "balanced_sizes": None,
    }
    test_constraint["27"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 3, 2: 2},
        "balanced_sizes": None,
    }
    test_constraint["28"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (4, 3),
    }
    test_constraint["29"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2, 6), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (3, 2),
    }
    test_constraint["30"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["31"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (4, 7)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["32"] = {
        "must_link_instances": [(0, 1), (5, 8), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 8), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["33"] = {
        "must_link_instances": [(0, 1), (5, 8), (3, 4)],
        "cannot_link_instances": [(0, 3), (4, 2), (4, 7)],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 8), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["34"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2:(0, 3, 4)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["35"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4), (8,9)],
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["36"] = {
        "must_link_instances": [(0, 1), (0, 2), (2, 4), (8,9)],
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["37"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["38"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 2)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["39"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["40"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 6)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["41"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (1, 2)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["42"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (3, 4)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["43"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 3, 2: 3},
        "balanced_sizes": None,
    }
    test_constraint["44"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": {0: 3, 1: 3, 2: 4},
        "balanced_sizes": None,
    }
    test_constraint["45"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (4, 3),
    }
    test_constraint["46"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (3, ),
    }
    test_constraint["47"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 3)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["48"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 5)],
        "cannot_link_instances": [(1, 4), (2, 1)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["49"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": [(1, 4), (2, 6)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["50"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["51"] = {
        "must_link_instances": [(0, 1), (0, 2), (3, 4)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["52"] = {
        "must_link_instances": [(0, 1), (0, 5), (3, 4)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["53"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["54"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 5)],
        "partition_level_instances": {0: (0, 1, 2), 1: (5, 6), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["55"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 3), (4, 2), (1, 2)],
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["56"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (1, 2)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["57"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 5), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (3, 4, 7)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["58"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": {0: (3, 4), 1:(6, 7), 2: (1, 2, 4)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["59"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 3, 2: 3},
        "balanced_sizes": None,
    }
    test_constraint["60"] =  {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (5, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 3, 2: 2},
        "balanced_sizes": None,
    }
    test_constraint["61"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 3, 1: 4, 2: 3},
        "balanced_sizes": None,
    }
    test_constraint["62"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (4, 3),
    }
    test_constraint["63"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2, 8), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (4, 3),
    }
    test_constraint["64"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 7)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": (3, ),
    }
    test_constraint["67"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 2: (3, 4, 5, 6, 7, 8)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 0, 2: 6},
        "balanced_sizes": None,
    }
    test_constraint["68"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, 1, 2), 1: (9, ), 2: (3, 4, 5, 6, 7, 8)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 0, 2: 6},
        "balanced_sizes": None,
    }
    test_constraint["69"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 0, 2: 6},
        "balanced_sizes": None,
    }
    test_constraint["70"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": {0: 4, 1: 0, 2: 6},
        "balanced_sizes": None,
    }

    test_constraint["82"] = {
        "must_link_instances": [(0, 1)],
        "cannot_link_instances": [(0, 2)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["83"] = {
        "must_link_instances": [(0, 1)],
        "cannot_link_instances": [(0, 1)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["84"] = {
        "must_link_instances": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
        "cannot_link_instances": [(8, 9)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["additional85"] = {
        "must_link_instances": [(8, 9)],
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["85"] = {
        "must_link_instances": [(0, 9)],
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["86"] = {
        "must_link_instances": [(8, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {2: (8, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["additional86"] = {
        "must_link_instances": [(0, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {1: (0, ), 2: (8, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["87"] = {
        "must_link_instances": [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {2: (8, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["additional87"] = {
        "must_link_instances": [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {1: (0, ), 2: (8, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["additional88"] = {
        "must_link_instances": [(0, 1)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,), 2:(9, )},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["88"] = {
        "must_link_instances": [(0, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,), 2:(9, )},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["additional89"] = {
        "must_link_instances": [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,), 2:(9, )},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["89"] = {
        "must_link_instances": [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,), 2:(9, )},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["90"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1)],
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["91"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 9)],
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["92"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": {2: (8, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["93"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["94"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 9)],
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["95"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(8, 9)],
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["96"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["97"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (8, 9)],
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["98"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": {2: (1, 2)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["99"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": {2: (0, 1)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["100"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["101"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9), 2:(0, 9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["102"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (1, 2)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["103"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (0, 1)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["104"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (1,2,3,4,5,6,7,8,9), 2:(0, )},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["105"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (1,2,3,4,5,6,7,8,9), 2:(1, )},
        "limited_sizes": None,
        "balanced_sizes": None,
    }

    test_constraint["106"] = {
        "must_link_instances": [(0, 1)],
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["107"] = test_constraint["106"]
    test_constraint["108"] = {
        "must_link_instances": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["109"] = test_constraint["108"]
    test_constraint["110"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["111"] = test_constraint["110"]
    test_constraint["112"] = {
        "must_link_instances": None,
        "cannot_link_instances": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["113"] = test_constraint["112"]
    test_constraint["114"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {2: (0, 9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["115"] = test_constraint["114"]
    test_constraint["116"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0, ), 2:(1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["117"] = test_constraint["116"]
    test_constraint["118"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": {2: (0, 9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["119"] = test_constraint["118"]
    test_constraint["120"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["121"] = test_constraint["120"]

    test_constraint["122"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 10, 1:1, 2:0},
        "balanced_sizes": None,
    }
    test_constraint["123"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 9, 1:1, 2:0},
        "balanced_sizes": None,
    }
    test_constraint["124"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 7, 1:3, 2:0},
        "balanced_sizes": None,
    }
    test_constraint["125"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": None,
        "limited_sizes": {0: 7, 1:2, 2:1},
        "balanced_sizes": None,
    }
    test_constraint["126"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["127"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6,7,8,9)},
        "non_partition_level_instances": {0: (0,1), 2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["128"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }
    test_constraint["129"] = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": {0: (0,1,2,3,4,5,6), 1:(7,8,9)},
        "non_partition_level_instances": {1: (7,8), 2: (0,1,2,3,4,5,6,7,8,9)},
        "limited_sizes": None,
        "balanced_sizes": None,
    }

    return test_constraint



@pytest.mark.parametrize(
    ["No","e", "labels", "n_clusters"],
    [
        pytest.param("1", None, [0, 0, 1, 1, 2, 2, 3, 3], 4, id="No.1"),
        pytest.param("2", "R", [0, 0, 1, 1, 2, 2, 3, 4], 4, id="No.2"),
        pytest.param("3", "R", [0, 0, 1, 1, 2, 2, 2, 2], 4, id="No.3"),
        pytest.param("4", None, [0, 0, 0, 2, 2, 1, 0, 2, 1, 1], 3, id="No.4"),
        pytest.param("5", "R", [0, 0, 0, 2, 2, 1, 0, 2, 1, 1], 4, id="No.5"),
        pytest.param("6", "R", [0, 0, 0, 2, 2, 1, 0, 2, 1, 1], 2, id="No.6"),
        pytest.param("71", None, [0 for _ in range(10)], 1, id="No.71"),
        pytest.param("72", None, [0,1,2,3,4], 5, id="No.72"),
    ]
)
def test_cluster_number_constraint_check(No, e, labels, n_clusters, create_err_msg):
    if e is None:
        cc.cluster_number_constraint_check(labels, n_clusters)
    elif e == "R":
        with pytest.raises(RuntimeError) as e:
            cc.cluster_number_constraint_check(labels, n_clusters)
        err_msg = create_err_msg.get(No)
        assert err_msg in str(e)
    else:
        raise Exception("Invalid Exception")


@pytest.mark.parametrize(
    ["constraint_name", "content"],
    [
        pytest.param(None, None, id="No.7"),
        pytest.param("must_link_instances", [(0, 1), (0, 2)], id="No.8"),
        pytest.param("cannot_link_instances", [(0, 1), (0, 2)], id="No.9"),
        pytest.param("partition_level_instances", {0: (0, 1, 2)}, id="No.10"),
        pytest.param("non_partition_level_instances", {0: (0, 1, 2)}, id="No.11"),
        pytest.param("limited_sizes", {0: 4, 1: 3, 2: 3}, id="No.12"),
        pytest.param("balanced_sizes", (4, 3), id="No.13"),
        pytest.param("must_link_instances", [(0, 1)], id="No.73"),
        pytest.param("must_link_instances", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)], id="No.74"),
        pytest.param("cannot_link_instances", [(0, 1)], id="No.75"),
        pytest.param("cannot_link_instances", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)], id="No.76"),
        pytest.param("partition_level_instances", {2: (9, )}, id="No.77"),
        pytest.param("partition_level_instances", {0: (0,1,2,3,4,5,6,7,8,9)}, id="No.78"),
        pytest.param("non_partition_level_instances", {2: (9, )}, id="No.79"),
        pytest.param("non_partition_level_instances", {0: (0,1,2,3,4,5,6,7,8,9)}, id="No.80"),
        pytest.param("limited_sizes", {0: 10, 1: 10, 2: 10}, id="No.81"),
    ]
)
def test_contradiction_of_constraints_check_single_settings(constraint_name, content):
    n_samples = 10
    constraints = {
        "must_link_instances": None,
        "cannot_link_instances": None,
        "partition_level_instances": None,
        "non_partition_level_instances": None,
        "limited_sizes": None,
        "balanced_sizes": None,
    }

    if constraint_name is not None:
        constraints[constraint_name] = content
    cc.contradiction_of_constraints_check(n_samples, constraints)


@pytest.mark.parametrize(
    ["No", "normal"],
    [
        pytest.param("14", True, id="No.14"),
        pytest.param("15", False, id="No.15"),
        pytest.param("16", True, id="No.16"),
        pytest.param("17", False, id="No.17"),
        pytest.param("18", True, id="No.18"),
        pytest.param("19", False, id="No.19"),
        pytest.param("20", True, id="No.20"),
        pytest.param("21", False, id="No.21"),
        pytest.param("22", True, id="No.22"),
        pytest.param("23", False, id="No.23"),
        pytest.param("24", True, id="No.24"),
        pytest.param("25", False, id="No.25"),
        pytest.param("26", True, id="No.26"),
        pytest.param("27", False, id="No.27"),
        pytest.param("28", True, id="No.28"),
        pytest.param("29", False, id="No.29"),
        pytest.param("30", True, id="No.30"),
        pytest.param("31", False, id="No.31"),
        pytest.param("32", True, id="No.32"),
        pytest.param("33", False, id="No.33"),
        pytest.param("34", False, id="No.34"),
        pytest.param("67", True, id="No.67"),
        pytest.param("68", False, id="No.68"),
        pytest.param("82", True, id="No.82"),
        pytest.param("83", False, id="No.83"),
        pytest.param("84", False, id="No.84"),
        pytest.param("additional85", True, id="additional85"),
        pytest.param("85", False, id="No.85"),
        pytest.param("86", True, id="No.86"),
        pytest.param("additional86", False, id="additional86"),
        pytest.param("87", True, id="No.87"),
        pytest.param("additional87", False, id="additional87"),
        pytest.param("additional88", True, id="additional88"),
        pytest.param("88", False, id="No.88"),
        pytest.param("additional89", True, id="additional89"),
        pytest.param("89", False, id="No.89"),
        pytest.param("90", True, id="No.90"),
        pytest.param("91", False, id="No.91"),
        pytest.param("92", True, id="No.92"),
        pytest.param("93", False, id="No.93"),
        pytest.param("94", True, id="No.94"),
        pytest.param("95", False, id="No.95"),
        pytest.param("96", True, id="No.96"),
        pytest.param("97", False, id="No.97"),
        pytest.param("98", True, id="No.98"),
        pytest.param("99", False, id="No.99"),
        pytest.param("100", True, id="No.100"),
        pytest.param("101", False, id="No.101"),
        pytest.param("102", True, id="No.102"),
        pytest.param("103", False, id="No.103"),
        pytest.param("104", True, id="No.104"),
        pytest.param("105", False, id="No.105"),
        pytest.param("122", True, id="No.122"),
        pytest.param("123", False, id="No.123"),
        pytest.param("124", True, id="No.124"),
        pytest.param("125", False, id="No.125"),
        pytest.param("126", True, id="No.126"),
        pytest.param("127", False, id="No.127"),
        pytest.param("128", True, id="No.128"),
        pytest.param("129", False, id="No.129"),
    ]
)
def test_contradiction_of_constraints_check_comb_settings(No, normal, create_constraint, create_err_msg):
    n_samples = 10
    constraints = create_constraint.get(No)
    
    if normal:
        cc.contradiction_of_constraints_check(n_samples, constraints)
    else:
        if No == "34":
            with pytest.raises(ValueError) as e:
                cc.contradiction_of_constraints_check(n_samples, constraints)
            err_msg = create_err_msg.get(No)
            assert err_msg in str(e)
        else:
            with pytest.raises(RuntimeError) as e:
                cc.contradiction_of_constraints_check(n_samples, constraints)
            keywords = create_err_msg.get(No)
            assert all(k in str(e) for k in keywords)

@pytest.mark.parametrize(
    ["No", "normal", "labels_dummy"],
    [
        pytest.param("35", True, None, id="No.35"),
        pytest.param("36", False, None, id="No.36"),
        pytest.param("37", True, None, id="No.37"),
        pytest.param("38", False, None, id="No.38"),
        pytest.param("39", True, None, id="No.39"),
        pytest.param("40", False, None, id="No.40"),
        pytest.param("41", True, None, id="No.41"),
        pytest.param("42", False, None, id="No.42"),
        pytest.param("43", True, None, id="No.43"),
        pytest.param("44", False, None, id="No.44"),
        pytest.param("45", True, None, id="No.45"),
        pytest.param("46", False, None, id="No.46"),
        pytest.param("47", True, None, id="No.47"),
        pytest.param("48", False, None, id="No.48"),
        pytest.param("49", False, None, id="No.49"),
        pytest.param("50", True, None, id="No.50"),
        pytest.param("51", False, None, id="No.51"),
        pytest.param("52", False, None, id="No.52"),
        pytest.param("53", True, None, id="No.53"),
        pytest.param("54", False, None, id="No.54"),
        pytest.param("55", False, None, id="No.55"),
        pytest.param("56", True, None, id="No.56"),
        pytest.param("57", False, None, id="No.57"),
        pytest.param("58", False, None, id="No.58"),
        pytest.param("59", True, None, id="No.59"),
        pytest.param("60", False, None, id="No.60"),
        pytest.param("61", False, None, id="No.61"),
        pytest.param("62", True, None, id="No.62"),
        pytest.param("63", False, None, id="No.63"),
        pytest.param("64", False, None, id="No.64"),
        pytest.param("69", True, [0] * 4 + [2] * 6, id="No.69"),
        pytest.param("70", False, [0] * 4 + [1] + [2] * 5, id="No.70"),
        pytest.param("106", True, [0] * 2 + [1] * 4 + [2] * 4, id="No.106"),
        pytest.param("107", False, [0, 1] + [1] * 4 + [2] * 4, id="No.107"),
        pytest.param("108", True, [0] * 10, id="No.108"),
        pytest.param("109", False, [0] * 9 + [1], id="No.109"),
        pytest.param("110", True, [0, 2] + [1, 2] * 4, id="No.110"),
        pytest.param("111", False, [0, 0] + [1, 2] * 4, id="No.111"),
        pytest.param("112", True, [0] + [1] * 4 + [2] * 5, id="No.112"),
        pytest.param("113", False, [0] *2 + [1] * 3 + [2] * 5, id="No.113"),
        pytest.param("114", True, [2, 0, 0, 1, 1, 1, 2, 2, 2, 2], id="No.114"),
        pytest.param("115", False, [2, 0, 0, 1, 1, 1, 2, 2, 2, 0], id="No.115"),
        pytest.param("116", True, [0] + [2] * 9, id="No.116"),
        pytest.param("117", False, [0, 1] + [2] * 8, id="No.117"),
        pytest.param("118", True, [0] + [1] * 4 + [2] * 4 + [0], id="No.118"),
        pytest.param("119", False, [0] + [1] * 4 + [2] * 4 + [2], id="No.119"),
        pytest.param("120", True, [1] * 4 + [2] * 6, id="No.120"),
        pytest.param("121", False, [0] + [1] * 3 + [2] * 6 + [2], id="No.121"),
    ]
)
def test_instances_level_constraints_check(No, normal, labels_dummy, create_constraint, create_err_msg):
    if labels_dummy is None:
        labels = [0, 0, 0, 2, 2, 1, 0, 2, 1, 1]
    else:
        labels = labels_dummy
    constraints = create_constraint.get(No)

    if normal:
        cc.instances_level_constraints_check(labels, constraints)
    else:
        with pytest.raises(RuntimeError) as e:
            cc.instances_level_constraints_check(labels, constraints)
        keywords = create_err_msg.get(No)
        print(str(e))
        assert all(k in str(e) for k in keywords)

@pytest.mark.parametrize(
    ["must_link", "expected"],
    [
        pytest.param([(0, 1), (2, 3), (4, 5)], [(0, 1), (2, 3), (4, 5)], id="No.65"),
        pytest.param([(0, 1), (1, 2), (3, 4)], [(0, 1, 2), (3, 4)], id="No.66"),
    ]
)
def test_organize_must_link_instances(must_link, expected):
    observed = cc.organize_must_link_instances(must_link)
    assert all(set(exp) == set(obs) for exp, obs in zip(expected, observed))

