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

class SVC(BaseSolver):
    """Support Vector Classification."""

    def __init__(self):
        self.timing = {}

    @measure
    def set_qubo(self, num_elements, basis, exponent_offset, xi):
        self.num_elements = num_elements
        self.basis = basis
        self.exponent_offset = exponent_offset
        self.xi = xi

        qubo_tmp = np.zeros([self.num_elements*self.num_points, self.num_elements*self.num_points]) #QUBO
        self.indexed_qubo = np.zeros([self.num_elements*self.num_points, self.num_elements*self.num_points]) #QUBO、上三角行列

        for n in range(self.num_points):
            for m in range(self.num_points):
                for k in range(self.num_elements):
                    for j in range(self.num_elements):
                        #デルタの設定
                        delta_nm = 0
                        delta_kj = 0
                        if n==m:
                            delta_nm = 1
                        if k==j:
                            delta_kj = 1


                        #カーネル関数の計算
                        #k = np.exp(-gamma * np.linalg.norm(X[n]-X[m])**2)
                        kernel = np.dot(self.training_dataset[n],  self.training_dataset[m])

                        a = self.num_elements * n + k
                        b = self.num_elements * m + j
                        qubo_tmp[a, b] = 1/2 * self.basis**(k+j-2*self.exponent_offset) * self.training_label[n] * self.training_label[m] * (kernel + self.xi) - delta_nm * delta_kj * self.basis**(k-self.exponent_offset)

        for a in range(self.num_elements*self.num_points):
            for b in range(self.num_elements*self.num_points):
                if a < b:
                    self.indexed_qubo[a, b] = qubo_tmp[a, b] + qubo_tmp[b, a]
                if a == b:
                    self.indexed_qubo[a, b] = qubo_tmp[a, b]

    def get_qubo(self):
        return self.indexed_qubo
    
    @measure
    def decode_solution(self):
        #アニーリングにより得られた解をaに変換
        self.alpha = np.zeros(self.num_points)
        for n in range(self.num_points):
            for k in range(self.num_elements):
                self.alpha[n] += self.basis**(k-self.exponent_offset) * list(self.solution)[self.num_elements*n+k] #a[n]を計算

        #Cを計算しておく
        self.C = 0
        for k in range(self.num_elements):
            self.C += self.basis**(k+1-self.exponent_offset)

        #決定境界の傾きを計算
        b_1 = 0
        b_2 = 0
        for n in range(self.num_points):
            b_2 += self.alpha[n] * (self.C - self.alpha[n])

            b_3 = 0
            for m in range(self.num_points):
                kernel = np.dot(self.training_dataset[n],  self.training_dataset[m])
                b_3 += self.alpha[m] * self.training_label[m] * kernel

            b_1 += self.alpha[n] * (self.C - self.alpha[n]) * (self.training_label[n] - b_3)

        if b_2 != 0:
            self.b = b_1 / b_2 #傾きbを計算
        else:
            raise RuntimeError("The slope of the decision boundary obtained by the optimization solver is an invalid value. Please run the program multiple times or adjust the parameters.")

    def fit(self, X, t, client, num_elements=3, basis=2, exponent_offset=0, xi=1):
        """Fit the given training data to two classes using the quantum SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matrix containing the training data, where n_samples is
            the number of samples and n_features is the number of features.
        t : array-like of shape (n_samples,)
            Vector containing the label representing the class to which each sample belongs.
            This parameter represents the two classes by label 1 or -1.
        client : AmplifyClient object
            Solver client for Fixstars Amplify.
        num_elements : int, default=3
            The number of 'basis' used to approximate the resulting coefficients alpha of SVM training as binary variables,
            where alpha means the coefficients obtained by SVM training.
            This parameter affects the size of the QUBO matrix and the accuracy of the binary variable approximation.
        basis : int, default=2
            A basis for approximating the resulting coefficients alpha of SVM training as binary variables.
        exponent_offset : int, default=0
            Parameter to adjust the exponential portion of 'basis'. Increasing 'exponent_offset' results in a smaller C,
            the maximum value of alpha, whereas decreasing this value results in a larger C.
            The appropriate value of this parameter should be determined based on the selected 'basis', 'num_elements', and obtained alpha.
        xi : {int, float}, default=1
            Parameter corresponding to the constraint coefficients of the quadratic programming problem.
        
        Raises
        ------
        ValueError
            If the vector 't' contains a label other than 1 or -1, or if the vector 't' consists only of 1 or only of -1.
        ValueError
            If a parameter such as 'num_elements', 'basis' and 'xi' is out of the valid range.
            (e.g., 'num_elements' is less than 1, 'basis' is less than 2, and 'xi' is less than 0.)
        TypeError
            If an argument such as 'X', 'num_elements', 'basis', 'exponent_offset' and 'xi' has invalid data type.
        IndexError
            If the lengths of 'X' and 't' do not match.
        RuntimeError
            If Fixstars Amplify throw an exception.
        RuntimeError
            If the slope of the decision boundary obtained by the optimization solver is an invalid value.
        """

        # Array check
        array_check(X, "X", 2)
        array_check(t, "t", 1)
        
        # Parameter type check
        if not isinstance(num_elements, int):
            raise TypeError("'num_elements' has invalid data type.")
        if not isinstance(basis, int):
            raise TypeError("'basis' has invalid data type.")
        if not isinstance(exponent_offset, int):
            raise TypeError("'exponent_offset' has invalid data type.")
        if not isinstance(xi, (int, float)):
            raise TypeError("'xi' has invalid data type.")
        
        # Parameter value check
        if num_elements < 1:
            raise ValueError("'num_elements' must be greater than or equal to 1.")
        if basis < 2:
            raise ValueError("'basis' must be greater than or equal to 2.")
        if xi < 0:
            raise ValueError("'xi' must be greater than or equal to 0.")

        # Parameter length check
        if len(X) != len(t):
            raise IndexError("The lengths of 'X' and 't' do not match.")

        # Parameter validity check
        if set(t) != {1, -1}:
            raise ValueError("Label 1 or -1 must be specified in the vector 't', and both 1 and -1 must be specified at least once in 't'.")

        self.training_dataset = X
        self.num_points = len(X)
        self.num_dimensions = len(X[0])
        self.training_label = t

        self.set_qubo(num_elements, basis, exponent_offset, xi)
        self.n_points = len(self.indexed_qubo)
        self.solve(client)
        self.decode_solution()

    def predict(self, X):
        """Predict class labels for the given test data.

        Parameters
        ----------
        X : array like of shape (n_sample, n_features)
            Matrix containing the test data for which we want to predict class labels,
            where n_samples is the number of samples and n_features is the number of features.
        
        Returns
        -------
        test_label : ndarray of shape (n_sample,)
            Vector containing the index representing the class to which each sample belongs

        Raises
        ------
        ValueError
            If n_features of 'X' and that of the training data do not match.
        TypeError
            If 'X' has an invalid type.
        AttributeError
            If this instance is not fitted yet.
        """

        # Attribute check
        if not hasattr(self, "b"):
            raise AttributeError("This instance is not fitted yet")
        
        # Array check
        array_check(X, "X", 2)
        
        # Parameter value check
        if len(X[0]) != len(self.training_dataset[0]):
            raise ValueError("n_features of 'X' and that of the training data do not match.")

        test_label = np.zeros(len(X))
        for test in range(len(X)):
            #決定境界を計算
            f = self.b
            for n in range(self.num_points):
                kernel = np.dot(self.training_dataset[n],  X[test])
                f += self.alpha[n] * self.training_label[n] * kernel
            test_label[test] = np.sign(f) #クラスラベルを入れる
        return test_label