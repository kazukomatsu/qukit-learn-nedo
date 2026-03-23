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

import pytest, os, shutil
from qklearn.utils import read_token

def backup(path=None):
    if (path is None):
        fpath = os.path.join(os.getcwd(), "amplify-license.yaml")
    else:
        fpath = path

    if (os.path.isfile(fpath)):
        shutil.move(fpath, f"{fpath}.backup")


def restore(path=None):
    if (path is None):
        fpath = os.path.join(os.getcwd(), "amplify-license.yaml")
    else:
        fpath = path

    if (os.path.isfile(fpath + ".backup")):
        shutil.move(f"{fpath}.backup", fpath)

def test_read_token_01():
    path = os.path.join(os.getcwd(), "amplify-license.yaml")
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")

    token = read_token("Fixstars")
    assert(token == "hoge")
    restore(path)

def test_read_token_02():
    path = os.path.join(os.getcwd(), "amplify-license.yaml")
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")
        f.write("        NEC: hogehoge\n")

    token = read_token("NEC")
    assert(token == "hogehoge")
    restore(path)

def test_read_token_03():
    path = os.path.join(os.getcwd(), "amplify-license.yaml")
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")
        f.write("        NEC: hogehoge\n")

    with pytest.raises(ValueError) as e:
        token = read_token("D-Wave")
    restore(path)

def test_read_token_04():
    path = os.path.join(os.getcwd(), "amplify-license.yaml")
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")
        f.write("        NEC: hogehoge\n")

    with pytest.raises(TypeError) as e:
        token = read_token(["NEC"])
    restore(path)

def test_read_token_05():
    path = "/tmp/amplify-license.yaml"
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")
        f.write("        NEC: hogehoge\n")

    token = read_token("NEC", path="/tmp/amplify-license.yaml")
    assert(token == "hogehoge")
    restore(path)

def test_read_token_06():
    path = "/tmp/amplify-license.yaml"
    backup(path)
    with pytest.raises(FileNotFoundError) as e:
        token = read_token("NEC", path="/tmp/amplify-license.yaml")
    restore(path)

def test_read_token_07():
    path = "/tmp/amplify-license.yaml"
    backup(path)
    with open(path, mode='w') as f:
        f.write("license:\n")
        f.write("        Fixstars: hoge\n")
        f.write("        NEC: hogehoge\n")
    with pytest.raises(TypeError) as e:
        token = read_token("NEC", path=["/tmp/amplify-license.yaml"])
    restore(path)

def test_read_token_08():
    path = "/tmp/amplify-license.yaml"
    backup(path)
    with open(path, mode='w') as f:
        f.write("Fixstars: hoge\n")
        f.write("NEC: hogehoge\n")
    with pytest.raises(ValueError) as e:
        token = read_token("NEC", path="/tmp/amplify-license.yaml")
    restore(path)
