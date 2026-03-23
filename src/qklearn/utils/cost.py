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
from typing import Union

from scipy.spatial import distance  # type: ignore
import numpy as np
from numpy.typing import NDArray

from . import min_max

def cost(data: Union[list[list[Union[int, float]]], NDArray[Union[np.int64, np.float64]]],
		 label: Union[list[Union[int, float]], NDArray[Union[np.int64, np.float64]]],
		 scaling: str ='normal') -> float:
	"""Calculate the cost using the data and labels.

	Parameters
	----------
	data : array-like of shape (n_samples, n_features)
		An array of samples.
	label : array-like of shape (n_samples, )
		Indices of the cluster each sample belongs to.
	scaling : str, default="normal"
		A scaling method. The distance matrix is normalized using min-max
		scaling only when this parameter is set to "normal".

	Returns
	-------
	cost : float
		Sum of the distance between pairs of samples with the same label.
    """
	dist = distance.squareform(distance.pdist(data))
	if scaling == 'normal':
		dist = min_max(dist)

	leng = len(data)
	cost = sum(
		[
			dist[i,j]
			for i in range(0,leng)
			for j in range(i+1,leng)
			if label[i] == label[j]
		]
	)

	return cost
