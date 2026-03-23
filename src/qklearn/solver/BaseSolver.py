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
from amplify import VariableGenerator
from amplify import solve as amplify_solve
from ..utils.measure import measure
import numpy as np
try:
    from dimod import BinaryQuadraticModel as BQM
    from neal import SimulatedAnnealingSampler as SAS
except ImportError as e:
    pass

class BaseSolver:
    @measure
    def set_BinaryMatrix(self):
        if not hasattr(self, "index2label"):
            gen = VariableGenerator()
            self.model = gen.matrix("Binary", self.n_points)
            self.model.quadratic = self.indexed_qubo
            # self.q = gen.array("Binary", self.n_points)
        else:
            gen = VariableGenerator()
            self.model = gen.matrix("Binary", self.n_clusters*self.n_points)
            self.model.quadratic = self.indexed_qubo
            # self.q = gen.array("Binary", self.n_clusters*self.n_points)
    
    def solve(self, client):
        self.set_BinaryMatrix()
        if client is None:
            quadratic = {(i, j): val
                         for i, row in enumerate(self.model.quadratic)
                         for j, val in enumerate(row)
                         if i != j}
            linear = {i : val
                      for i, row in enumerate(self.model.quadratic)
                      for j, val in enumerate(row)
                      if i == j}
            try:
                bqm = BQM(linear, quadratic, "BINARY")
                sa_sampler = SAS()
                self.result = sa_sampler.sample(bqm, num_reads=100)
            except NameError as e:
                raise RuntimeError("Install dimod and dwave-neal") from e
            except Exception as e:
                raise RuntimeError("exceptions were raised from solve() of the dimod or dwave-neal") from e
            self.timing['run_annealings'] = self.result.info["timing"]["sampling_ns"] * 10**-9
            self.solution = [int(v) for v in self.result.first.sample.values()]
        else:
            try:
                self.result = amplify_solve(self.model, client)
            except Exception as e:
                raise RuntimeError("exceptions were raised from solve() of the Fixstars Amplify SDK") from e
            self.__set_timing('run_annealings', self.result.execution_time.total_seconds())
            self.solution = [v for v in self.result.best.values.values()]

    def qubo_to_indexed_qubo(self, given_qubo=None):
        if given_qubo is None:
            given_qubo = self.qubo
        indexed_qubo = np.array(
            [
                [
                    0 if j < i
                    else 0 if (self.index2label[i], self.index2label[j]) not in given_qubo
                    else given_qubo[(self.index2label[i], self.index2label[j])]
                    for j in range(self.n_clusters * self.n_points)
                ]
                for i in range(self.n_clusters * self.n_points)
            ]
        )
        return indexed_qubo

    def _build_label_index_maps(self):
        self.label2index = {
            (i, a): i * self.n_clusters + a
            for i in range(self.n_points)
            for a in range(self.n_clusters)
        }
        self.index2label = {
            i * self.n_clusters + a: (i, a)
            for i in range(self.n_points)
            for a in range(self.n_clusters)
        }

    def _omit_zero_coefficients(self, qb):
        return {k: v for k, v in qb.items() if v != 0}

    def _merge_dicts(self, a, b, func=lambda x, y: x+y):
        d1 = a.copy()
        d2 = b.copy()
        d1 = {
            k: func(d2[k], v) if k in d2 else v 
            for k, v in d1.items()
        }
        d2.update(d1)
        return d2

    def __set_timing(self, fn, time):
        self.timing[fn] = time if fn not in self.timing else self.timing[fn]+time
