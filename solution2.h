#ifndef LEETCODEMAC_SOLUTION2_H
#define LEETCODEMAC_SOLUTION2_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution2 {
public:
    bool findNumberIn2DArray(vector<vector<int>> &matrix, int target) {
        int x = matrix.size()-1, y = 0;
        while (x >= 0 && y < matrix[0].size()) {
            if (matrix[x][y] == target) {
                return true;
            } else if (target < matrix[x][y]) {
                x--;
            } else {
                y++;
            }
        }
        return false;
    }

    vector<int> findSquare(vector<vector<int>>& matrix) {

    }
};

#endif //LEETCODEMAC_SOLUTION2_H
