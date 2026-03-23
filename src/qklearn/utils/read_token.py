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
import yaml
import os

def read_token(service, path=None):
    """Read token from amplify-license.yaml

    Parameters
    ----------
    service : str
        A service name for an optimization solver to solve QUBO problems.
        A token is searched with this string as the key in amplify-license.yaml.
    path : str, default=None
        File path to amplify-license.yaml. If None, amplify-license.yaml in current directory is used.

    Raises
    ------
    ValueError
        If an amplify-license.yaml had invalid structure.
    ValueError
        If token of service was not found in amplify-license.yaml.
    FileNotFoundError
        If amplify-license.yaml was not found.
    TypeError
        If an invalid data type was specified for an argument.

    Returns
    -------
    token : str
        A token corresponding to "service".
    """
    if (not isinstance(service, str)):
        raise TypeError("\"service\" argument should be str.")
    if ((path is not None) and (not isinstance(path, str))):
        raise TypeError("\"path\" argument should be str.")

    if path is None:
        path = os.path.join(os.getcwd(), "amplify-license.yaml")

    with open(path) as f:
        tmp = yaml.safe_load(f)
    licenses = tmp.get("license")
    if (licenses is None):
        raise ValueError(f"{path} did not have \"license\" sequence")
    token = licenses.get(service)
    if (token is None):
        raise ValueError(f"{service} was not found in {path}.")

    return token
