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

results = []
results_BiDViT = []
results_Constained = []

def format_float(val, width, fmt_str):
    return f"{val:{fmt_str}}" if val is not None else f"{'N/A':<{width}}"

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call":
        no = getattr(item, "no", None)
        name = getattr(item, "name", None)

        qk_ari = getattr(item, "qklearn_ANI", None)
        qk_nmi = getattr(item, "qklearn_NMI", None)
        qk_cost = getattr(item, "qklearn_cost", None)
        qk_lam_default = getattr(item, "qklearn_lam_default", None)
        qk_lam_success = getattr(item, "qklearn_lam_success", None)
        results.append((no, name, "qklearn", qk_ari, qk_nmi, qk_cost, qk_lam_default, qk_lam_success))

        sk_ari = getattr(item, "sklearn_ANI", None)
        sk_nmi = getattr(item, "sklearn_NMI", None)
        sk_cost = getattr(item, "sklearn_cost", None)
        sk_lam_default = getattr(item, "sklearn_lam_default", None)
        sk_lam_success = getattr(item, "sklearn_lam_success", None)
        results.append((no, name, "sklearn", sk_ari, sk_nmi, sk_cost, sk_lam_default, sk_lam_success))

        constraint = getattr(item, "constraint", None)
        if constraint is not None:
            const_ari = getattr(item, "const_ANI", None)
            const_nmi = getattr(item, "const_NMI", None)
            const_cost = getattr(item, "const_cost", None)
            results_Constained.append((no, name, "ConstarintedClstering", constraint, const_ari, const_nmi, const_cost))

            sk_ari = getattr(item, "sklearn_ANI", None)
            sk_nmi = getattr(item, "sklearn_NMI", None)
            sk_cost = getattr(item, "sklearn_cost", None)
            results_Constained.append((no, name, "KMeans", "N/A", sk_ari, sk_nmi, sk_cost))

        qk_results = getattr(item, "qklearn_results", None)
        sk_results = getattr(item, "sklearn_results", None)
        if qk_results is not None and sk_results is not None:
            for qk_res, sk_res in zip(qk_results, sk_results):
                results_BiDViT.append((no, name, "qklearn", qk_res[0], qk_res[1], qk_res[2], qk_res[3]))
                results_BiDViT.append((no, name, "sklearn", sk_res[0], sk_res[1], sk_res[2], sk_res[3]))


def pytest_sessionfinish(session, exitstatus):
    print("\n\n---- Test Summary ----")
    
    if len(results_BiDViT) > 0:
        print(f"{'No':<2} | {'Benchmark Name':<20} | {'Library':<8} | {'ARI':<8} | {'NMI':<8} | {'cost':<8} | {'n_clusters':<2}")
        print("-" * 82)
        for no, name, lib, ari, nmi, cost, n_clusters in results_BiDViT:
            print(f"{no:<2} | {name:<20} | {lib:<8} | {format_float(ari, 8, '<8.4f')} | {format_float(nmi, 8, '<8.4f')} | {format_float(cost, 8, '<8.2f')} | {n_clusters:<2}")
        print("-" * 82)
    elif len(results_Constained) > 0:
        print(f"{'No':<2} | {'Benchmark Name':<20} | {'Algorithm':<23} | {'Constraint':<23} | {'ARI':<8} | {'NMI':<8} | {'cost':<8}")
        print("-" * 111)
        for no, name, alg, const, ari, nmi, cost in results_Constained:
            print(f"{no:<2} | {name:<20} | {alg:<23} | {const:<23} | {format_float(ari, 8, '<8.4f')} | {format_float(nmi, 8, '<8.4f')} | {format_float(cost, 8, '<8.2f')}")
        print("-" * 111)
    else:
        print(f"{'No':<2} | {'Benchmark Name':<22} | {'Library':<8} | {'ARI':<8} | {'NMI':<8} | {'cost':<10} | {'lam_default':<12} | {'lam_success':<12}")
        print("-" * 111)
        for no, name, lib, ari, nmi, cost, lam_default, lam_success in results:
            print(f"{no:<2} | {name:<22} | {lib:<8} | {format_float(ari, 8, '<8.4f')} | {format_float(nmi, 8, '<8.4f')} | {format_float(cost, 10, '<10.2f')} | {format_float(lam_default, 12, '<12.4f')} | {format_float(lam_success, 12, '<12.4f')}")
        print("-" * 111)
