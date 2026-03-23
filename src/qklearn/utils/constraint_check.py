"""
MIT License

Copyright © 2023-2025 Tohoku University

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
import collections

def cluster_number_constraint_check(labels, n_clusters):
    # A method to check if the cluster number constraint is satisfied.
    if len(set(labels)) > n_clusters:
        raise RuntimeError("The number of clusters returned by an optimization solver is greater than the specified n_clusters.")
    elif len(set(labels)) < n_clusters:
        raise RuntimeError("The number of clusters returned by an optimization solver is less than the specified n_clusters.")

def contradiction_of_constraints_check(n_samples, constraints):
    # A method to check for contradictions between specified constraints.
    mlink_instances = constraints.get("must_link_instances")
    clink_instances = constraints.get("cannot_link_instances")
    plevel_instances = constraints.get("partition_level_instances")
    nplevel_instances = constraints.get("non_partition_level_instances")
    l_sizes = constraints.get("limited_sizes")
    b_sizes = constraints.get("balanced_sizes")

    # check for contradiction between must-link constraint and cannot-link constraint
    if (mlink_instances is not None) and (clink_instances is not None):
        organized_mlink_instances = organize_must_link_instances(mlink_instances)
        for mlink in organized_mlink_instances:
            for clink in clink_instances:
                if len(set(mlink) & set(clink)) > 1:
                    _raise_constraint_error("must-link constraint", "cannot-link constraint")
    
    # check for contradiction between partition-level constraint and other constraints.
    if plevel_instances is not None:
        # check if partition_level_instances is set correctly.
        pli_values = list(plevel_instances.values())
        for i, a in enumerate(pli_values):
            for b in pli_values[(i+1):]:
                if set(a) & set(b):
                    raise ValueError("There are samples whose cluster assignment is not uniquely determined.")
        
        compared_constraint = "partition-level constraint"
        affiliation = _create_affiliation_cluster_array(n_samples, plevel_instances)    

        if mlink_instances is not None:
            organized_mlink_instances = organize_must_link_instances(mlink_instances)
            _check_must_link_constraint(affiliation, organized_mlink_instances, compared_constraint)
        
        if clink_instances is not None:
            _check_cannot_link_constraint(affiliation, clink_instances, compared_constraint)
                
        if nplevel_instances is not None:
            _check_non_partition_level_constraint(affiliation, nplevel_instances, compared_constraint)
        
        if l_sizes is not None:
            _check_limited_sizes_constraint(affiliation, l_sizes, compared_constraint)

        if b_sizes is not None:
            _check_balanced_sizes_constraint(affiliation, b_sizes, compared_constraint)


def _raise_constraint_error(conflicting_constraint, reference_constraint=None):
    if reference_constraint is not None:
        raise RuntimeError(f"There is a conflict between the specified {conflicting_constraint} and {reference_constraint}. Please review the content of the constraints.")
    else:
        raise RuntimeError(f"The solution returned by an optimization solver does not satisfy the {conflicting_constraint}.")

def organize_must_link_instances(must_link):
    organized_must_link_instances = tuple()
    ml = must_link.copy()
    while ml:
        temp = ml.pop(0)
        merged = True
        while merged:
            merged = False
            index_temp = tuple()
            for i, pair in enumerate(ml):
                if set(temp) & set(pair):
                    temp += pair
                    index_temp += (i,)
                    merged = True
            ml = [v for i, v in enumerate(ml) if i not in index_temp]
        organized_must_link_instances += (tuple(sorted(set(temp), reverse=True)),)
    return list(organized_must_link_instances)

def _create_affiliation_cluster_array(n_samples, constraint_instances):
    # arr[i] = j means that sample i belongs to or does not belong to cluster j.
    # arr[i] = None means that it is not specified which cluster sample i belongs to or does not belong to.
    arr = [None for _ in range(n_samples)]
    for k, v in constraint_instances.items():
        for vv in v:
            arr[vv] = k
    return arr

def _check_links_consistency(affiliation, links, expected=True):
    for link in links:
        if expected:  # in the case of must-link
            pairs = [(link[0], sample_j) for sample_j in link[1:]]
        else:  # in the case of cannot-link
            pairs = [
                (sample_i, sample_j)
                for i, sample_i in enumerate(link)
                for sample_j in link[(i+1):]
            ]
        for sample_i, sample_j in pairs:
            if (affiliation[sample_i] is not None and affiliation[sample_j] is not None):
                observed = (affiliation[sample_i] == affiliation[sample_j])
                if expected != observed:
                    return True
    return False

def instances_level_constraints_check(labels, constraints):
    # A function to check whether the solution returned by an optimization solver satisfies the specified constraints.
    constraint_checks = [
        (constraints.get("must_link_instances"), _check_must_link_constraint),
        (constraints.get("cannot_link_instances"), _check_cannot_link_constraint),
        (constraints.get("partition_level_instances"), _check_partition_level_constraint),
        (constraints.get("non_partition_level_instances"), _check_non_partition_level_constraint), 
        (constraints.get("limited_sizes"), _check_limited_sizes_constraint),
        (constraints.get("balanced_sizes"), _check_balanced_sizes_constraint),
    ]

    for constraint_instances, constraint_func in constraint_checks:
        if constraint_instances is not None:
            constraint_func(labels, constraint_instances)


# must-link
def _check_must_link_constraint(target, mlink_instances, compared_constraint=None):
    if _check_links_consistency(target, mlink_instances, expected=True):
        if compared_constraint is None:
            _raise_constraint_error("must-link constraint")
        else:
            _raise_constraint_error("must-link constraint", compared_constraint)

# cannot-link
def _check_cannot_link_constraint(target, clink_instances, compared_constraint=None):
    if _check_links_consistency(target, clink_instances, expected=False):
        if compared_constraint is None:
            _raise_constraint_error("cannot-link constraint")
        else:
            _raise_constraint_error("cannot-link constraint", compared_constraint)

# partition-level
def _check_partition_level_constraint(target, plevel_instances):
    affiliation = _create_affiliation_cluster_array(len(target), plevel_instances)
    for n in range(len(target)):
        if affiliation[n] is not None and affiliation[n] != target[n]:
            _raise_constraint_error("partition-level constraint")

# non-partition-level
def _check_non_partition_level_constraint(target, nplevel_instances, compared_constraint=None):
    for k, v in nplevel_instances.items():
        for vv in v:
            if target[vv] is not None and target[vv] == k:
                if compared_constraint is None:
                    _raise_constraint_error("non-partition-level constraint")
                else:
                    _raise_constraint_error("non-partition-level constraint", compared_constraint)

# limited_sizes
def _check_limited_sizes_constraint(target, l_sizes, compared_constraint=None):
    if any(target.count(k) > v for k, v in l_sizes.items()):
        if compared_constraint is None:
            _raise_constraint_error("limited-sizes constraint")
        else:
            _raise_constraint_error("limited-sizes constraint", compared_constraint)

# balanced_sizes
def _check_balanced_sizes_constraint(target, b_sizes, compared_constraint=None):
    if None in target:
        target_filtered = [t for t in target if t is not None]
        cluster_counts = collections.Counter(target_filtered)
        check_list = [cc > b_sizes[0] for cc in cluster_counts.values()]
    else:
        cluster_counts = collections.Counter(target)
        check_list = [cc not in b_sizes for cc in cluster_counts.values()]
    
    if any(check_list):
        if compared_constraint is None:
            _raise_constraint_error("balanced-sizes constraint")
        else:
            _raise_constraint_error("balanced-sizes constraint", compared_constraint)