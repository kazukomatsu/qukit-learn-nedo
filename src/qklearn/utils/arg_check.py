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
import numpy as np

def array_check(X, arg_name, dim, if_dist=False):
    if (not isinstance(X, (list, np.ndarray))):
        raise TypeError(f"{arg_name} must be array-like object.")
    # if X has invalid shape, numpy raises ValueError.
    try:
        tmp = np.array(X)
    except ValueError as e:
        raise ValueError(f"{arg_name} has invalid shape") from e
    if (len(tmp.shape) != dim):
        raise ValueError(f"{arg_name} must be {dim}-d array")
    if (not str(tmp.dtype).startswith(('int', 'uint', 'float'))):
        raise TypeError(f"{arg_name} has invalid data type")
    for i in tmp.shape:
        if(i == 0):
            raise ValueError(f"{arg_name} has an empty element")
    if if_dist:
        if(tmp.shape[0] != tmp.shape[1]):
            raise ValueError(f"{arg_name} must be square matrix")
