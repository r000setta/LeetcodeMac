#ifndef LEETCODEMAC_NUMMATRIX_H
#define LEETCODEMAC_NUMMATRIX_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class NumMatrix {
public:
    vector<vector<int>> pre;

    NumMatrix(vector<vector<int>> &matrix) {
        if (matrix.size() == 0) return;
        int m = matrix.size(), n = matrix[0].size();
        pre = vector<vector<int>>(m + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return pre[row2 + 1][col2 + 1] - pre[row2 + 1][col1] - pre[row1][col2 + 1] + pre[row1][col1];
    }

    vector<int> sortByBits(vector<int> &arr) {
        vector<int> bit(10001, 0);
        for (int i = 1; i <= 10000; ++i) {
            bit[i] = bit[i >> 1] + (i & 1);
        }
        sort(arr.begin(), arr.end(), [&](int x, int y) {
            return bit[x] == bit[y] ? x < y : bit[x] < bit[y];
        });
        return arr;
    }


};

#endif //LEETCODEMAC_NUMMATRIX_H
