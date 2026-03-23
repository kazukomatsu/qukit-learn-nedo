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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <complex>
#include <unordered_map>
#include <cmath>
#include <random>
#include "BiDViT.hpp"
using namespace std;

vector<pair<size_t, double>> get_max_variance(vector<vector<double>> data){

    size_t size_row = data.size();
    size_t size_col = data[0].size();
    double mean, var;
    size_t index_max = 0;
    double flag = 0.0;

    for (size_t i = 0; i < size_col; i++){
        mean = 0.0; var=0.0;
        for (size_t j = 0; j < size_row; j++){
            mean += data[j][i];
        }
        mean /= size_row;
        for (size_t k = 0; k < size_row; k++){
            var += (data[k][i]-mean) * (data[k][i]-mean);
        }
        var /= size_row;
        if (var > flag){
            index_max = i; flag = var;
        }
    }

    vector<pair<size_t, double>> data_col(size_row);
    for (size_t i = 0; i < size_row; i++){
        data_col[i] = pair<size_t, double>(i, data[i][index_max]);
    }

    return data_col;
}

vector<int> get_binary_chunk(vector<pair<size_t, double>> data_col, int base){

    sort(
        data_col.begin(), data_col.end(),
        [](const pair<size_t, double>& x, const pair<size_t, double>& y){return x.second < y.second;}
    );
    size_t size = data_col.size();
    vector<int> binary_chunk(size, base);
    for (size_t i = size/2+1; i < size; i++) binary_chunk[data_col[i].first]++;

    return binary_chunk;

}

vector<int> get_chunk(vector<vector<double>> data, int kappa){
    
    size_t size = data.size();
    double x = double(size);
    int num_chunk = 1;
    vector<int> chunk(size, 0);
    while (x > kappa) {

        vector<vector<vector<double>>> data_per_chunk(num_chunk);
        for (auto&& vec : data_per_chunk) vec.reserve(size/num_chunk*2);
        for (size_t i = 0; i < size; i++) data_per_chunk[chunk[i]].push_back(data[i]);

        vector<vector<int>> muster_binary_chunk(num_chunk);
        for (int i = 0; i < num_chunk; i++){
            vector<pair<size_t, double>> data_col = get_max_variance(data_per_chunk[i]);
            vector<int> binary_chunk = get_binary_chunk(data_col, i*2);
            muster_binary_chunk[i] = binary_chunk;
        }

        vector<int> chunk_judge(size);
        vector<int> temp(num_chunk, 0);
        for (size_t i = 0; i < size; i++) chunk_judge[i] = muster_binary_chunk[chunk[i]][temp[chunk[i]]++];
        chunk = chunk_judge;

        x /= 2.0; num_chunk *= 2;
    }
    return chunk;

}

double euclidean(vector<double> x, vector<double> y){

    size_t size = x.size();
    double pow_norm;
    for (size_t i = 0; i < size; i++){
        pow_norm += (x[i]-y[i]) * (x[i]-y[i]);
    }
    return sqrt(pow_norm);

}

unordered_map<pair<int, int>, double> get_qubo(vector<vector<double>> data, vector<double> weight, double epsilon, double rev_rate){

    size_t size = data.size();
    double norm = 0;
    unordered_map<pair<int, int>, double> qubo;
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
            norm = euclidean(data[i], data[j]);
            if (i == j){
                qubo[make_pair(i, j)] = -1*weight[i];
            }else if (i < j && norm < epsilon){
                qubo[make_pair(i, j)] = (weight[i] < weight[j] ? weight[j] : weight[i]) * rev_rate;
            }
        }
    }
    return qubo;

}

vector<unordered_map<pair<int, int>, double>> get_qubo_and_S(vector<vector<double>> data, vector<double> weight, double epsilon, double rev_rate){

    size_t size = data.size();
    double norm = 0;
    unordered_map<pair<int, int>, double> qubo;
    unordered_map<pair<int, int>, double> S;
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
            norm = euclidean(data[i], data[j]);
            if (i == j){
                qubo[make_pair(i, j)] = -1*weight[i];
                S[make_pair(i, j)] = 1;
            }else if (i < j && norm < epsilon){
                qubo[make_pair(i, j)] = (weight[i] < weight[j] ? weight[j] : weight[i]) * rev_rate;
                S[make_pair(i, j)] = 1;
            }
        }
    }
    vector<unordered_map<pair<int, int>, double>> qubo_and_S(2);
    qubo_and_S[0] = qubo;
    qubo_and_S[1] = S;
    return qubo_and_S;

}

int get_rand(int min, int max){

    random_device rnd;
    mt19937 mt(rnd());
    uniform_int_distribution<> rand(min, max);
    return rand(mt);

}

vector<int> greedy(vector<vector<double>> data, vector<double> weight, double epsilon){

    size_t size_org = data.size();
    vector<int> solutions(size_org, 0);
    vector<int> index_list(size_org);
    for (size_t i = 0; i < size_org; i++) index_list[i] = i;

    vector<vector<int>> N(size_org, vector<int> (size_org, 0));
    for (size_t i = 0; i < size_org; i++){
        for (size_t j = 0; j < size_org; j++){
            if (euclidean(data[i], data[j]) < epsilon) N[i][j] = 1;
        }
    }

    for (size_t size = data.size(); size != 0;){
        vector<double> deg(size, 0);
        double temp = 0;

        for (size_t i = 0; i < size; i++){
            for (size_t j = 0; j < size; j++){
                if (i != j && N[i][j] == 1){
                    deg[i] += weight[j];
                }
            }
            deg[i] = deg[i] / weight[i];
            temp = (i == 0) ? deg[i] : (temp > deg[i] ? deg[i] : temp);
        }

        vector<int> degmin_list;
        degmin_list.reserve(size);
        for (size_t i = 0; i < size; i++){
            if (deg[i] == temp) degmin_list.push_back(i);
        }
        int degmin = degmin_list[get_rand(0, degmin_list.size()-1)];

        solutions[index_list[degmin]] = 1;

        vector<vector<double>> da;
        da.reserve(size);
        vector<double> we;
        we.reserve(size);
        vector<int> in;
        in.reserve(size);
        vector<vector<int>> Nd;
        Nd.reserve(size);
        for (size_t i = 0; i < size; i++){
            if (N[degmin][i] != 1){
                da.push_back(data[i]);
                we.push_back(weight[i]);
                in.push_back(index_list[i]);

                vector<int> Npre;
                Npre.reserve(size);
                for (size_t j = 0; j < size; j++){
                    if (N[degmin][j] != 1){
                        Npre.push_back(N[i][j]);
                    }
                }
                Nd.push_back(Npre);
            }
        }

        size = da.size();
        if (size == 1) break;

        data.resize(size);
        copy(da.begin(), da.end(), data.begin());
        weight.resize(size);
        copy(we.begin(), we.end(), weight.begin());
        index_list.resize(size);
        copy(in.begin(), in.end(), index_list.begin());
        N.resize(size);
        for (size_t i = 0; i < size; i++){
            N[i].resize(size);
            copy(Nd[i].begin(), Nd[i].end(), N[i].begin());
        }
    }
    return solutions;

}

vector<int> get_cluster(vector<vector<double>> data, vector<int> solutions){

    size_t size = data.size();
    vector<int> cluster(size);

    unordered_map<int, int> flag;
    unordered_map<int, int> card;
    int temp = 0;
    for (size_t i = 0; i < size; i++){
        if (solutions[i]){
            flag[i] = temp;
            card[temp] = i;
            temp++;
        }
    }

    size_t size_card = card.size();
    double norm, mirm;
    size_t clu = card[0];
    for (size_t i = 0; i < size; i++){
        if (solutions[i]) cluster[i] = flag[i];
        else{
            clu = card[0];
            mirm = euclidean(data[i], data[clu]);
            for (size_t j = 1; j < size_card; j++){
                norm = euclidean(data[i], data[card[j]]);
                if (mirm > norm){
                    clu = card[j];
                    mirm = norm;
                }
            }
            cluster[i] = flag[clu];
        }
    }
    return cluster;

}

/*pip install pybind11*/ 
namespace py = pybind11;
PYBIND11_PLUGIN(BiDViT){
    py::module m("BiDViT", "BiDViT");
    m.def("get_chunk", &get_chunk);
    m.def("get_qubo", &get_qubo);
    m.def("get_qubo_and_S", &get_qubo_and_S);
    m.def("greedy", &greedy);
    m.def("get_cluster", &get_cluster);
    return m.ptr();
}
