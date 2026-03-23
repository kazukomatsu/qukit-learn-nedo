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
import numpy as np
from .solver import BaseSolver
from .utils import measure, array_check

class LinearRegression(BaseSolver):
    """Perform linear regression analysis.
    """
    def __init__(self):
        self.timing = {}

    @measure
    def set_qubo(self, num_elements, basis, exponent_offset):
        self.num_elements = num_elements
        self.basis = basis
        self.exponent_offset = exponent_offset

        self.precision_vector = [-self.basis**(k+self.exponent_offset) for k in reversed(range(self.num_elements))] + [self.basis**(k+self.exponent_offset) for k in range(self.num_elements)]
        self.precision_matrix = np.kron(np.eye(self.num_dimensions+1), self.precision_vector)

        self.A = self.precision_matrix.T @ self.training_dataset.T @ self.training_dataset @ self.precision_matrix
        self.b =  - 2 * self.precision_matrix.T @ self.training_dataset.T @ self.regression_label

        self.indexed_qubo = 2*np.triu(self.A, k=1) + np.diag(np.diag(self.A)) + np.diag(self.b)

    def get_qubo(self):
        return self.indexed_qubo
    
    @measure
    def decode_solution(self):
        self.weights = self.precision_matrix @ np.array(list(self.solution))

    def fit(self, X, y, client, num_elements=3, basis=2, exponent_offset=0):
        """Fit linear model.

        Parameters
        ----------
        X : Array like of shape (n_sample, n_feature)
            Training data.
        y : Array like of shape (n_sample, )
            Target values.
        client : AmplifyClient
            A client object of Fixstars Amplify SDK.
        num_elements : int, default=3
            Upper limit of exponent of basis.
            The regression coefficient :math:`w_i` is approximated by the sum of the powers of the basis :math:`b` as follows:

            :math:`w_i = \Sigma_{n=0}^{num\_elements - 1}(b^{(n + exponent\_offset)}\hat{w_{in}}) - \Sigma_{n=0}^{num\_elements - 1}(b^{(num\_elements - 1 - n + exponent\_offset)}\hat{w_{in}^{\prime}})`

            where :math:`\hat{w_{in}}` and :math:`\hat{w_{in}^{\prime}}` are binary variables.
        basis : int, default=2
            A basis for approximating the regression coefficients with binary variables.
        exponent_offset : int, default=0
            Parameters that adjusts exponent of basis.

        Raises
        ------
        ValueError
            If variable 'X' or 'y' had Invalid shape.
        ValueError
            If variable 'num_elements' was less than 1.
        TypeError
            If an invalid data type was specified for an argument.
        IndexError
            If the lengths of 'X' and 'y' do not match.
        RuntimeError
            If exceptions were raised from the Fixstars Amplify SDK.
        """
        array_check(X, "X", 2)
        array_check(y, "y", 1)
        if (len(X) != len(y)):
            raise IndexError("The lengths of 'X' and 'y' do not match.")
        if isinstance(num_elements, bool) or not isinstance(num_elements, (int, np.integer)):
            raise TypeError("num_elements must be int")
        if isinstance(basis, bool) or not isinstance(basis, (int, np.integer)):
            raise TypeError("basis must be int")
        if isinstance(exponent_offset, bool) or not isinstance(exponent_offset, (int, np.integer)):
            raise TypeError("exponent_offset must be int")
        if (num_elements < 1):
            raise ValueError("num_elements must be greater than or equal to 1")

        self.num_points = len(X)
        self.num_dimensions = len(X[0])
        self.regression_label = y
        self.training_dataset = np.column_stack((X, np.ones(self.num_points)))

        self.set_qubo(num_elements, basis, exponent_offset)
        self.n_points = len(self.indexed_qubo)
        self.solve(client)
        self.decode_solution()

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : Array like of shape (n_sample, n_feature)
            Samples.

        Returns
        -------
        y : ndarray of shape (n_sample, )
            Predicted values.

        Raises
        ------
        ValueError
            If variable 'X' has invalid shape.
        TypeError
            If an invalid data type was specified for an argument.
        AttributeError
            If this instance is not fitted yet.
        """
        if (not hasattr(self, "indexed_qubo")):
            raise AttributeError("This instance is not fitted yet")
        array_check(X, "X", 2)

        X_tmp = np.column_stack((X, np.ones(len(X))))
        return X_tmp @ self.weights
