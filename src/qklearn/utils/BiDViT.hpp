/*
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
*/
namespace std{
    template <>
    class hash<std::pair<int, int>>{
    public:
        size_t operator()(const std::pair<int, int>& x) const{
            return hash<int>()(x.first) ^ hash<int>()(x.second);
        }
    };
}

std::vector<std::pair<size_t, double>> get_max_variance(std::vector<std::vector<double>> data);

std::vector<int> get_binary_chunk(std::vector<std::pair<size_t, double>> data_col, int base);

std::vector<int> get_chunk(std::vector<std::vector<double>> data, int kappa);

double euclidean(std::vector<double> x, std::vector<double> y);

std::unordered_map<std::pair<int, int>, double> get_qubo(std::vector<std::vector<double>> data, std::vector<double> weight, double epsilon, double rev_rate);

std::vector<std::unordered_map<std::pair<int, int>, double>> get_qubo_and_S(std::vector<std::vector<double>> data, std::vector<double> weight, double epsilon, double rev_rate);

int get_rand(int min, int max);

std::vector<int> greedy(std::vector<std::vector<double>> data, std::vector<double> weight, double epsilon);

std::vector<int> get_cluster(std::vector<std::vector<double>> data, std::vector<int> solutions);