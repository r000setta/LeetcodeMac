#ifndef LEETCODEMAC_SOLUTION3_H
#define LEETCODEMAC_SOLUTION3_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution3 {
public:
    bool tictactoeCheck(unordered_set<int> &S, vector<vector<int>> &wins) {
        for (auto win:wins) {
            bool flag = true;
            for (auto pos:win) {
                if (!S.count(pos)) {
                    flag = false;
                    break;
                }
            }
            if (flag) return true;
        }
        return false;
    }

    string tictactoe(vector<vector<int>> &moves) {
        vector<vector<int>> wins = {
                {0, 1, 2},
                {3, 4, 5},
                {6, 7, 8},
                {0, 3, 6},
                {1, 4, 7},
                {2, 5, 8},
                {0, 4, 8},
                {2, 4, 6}
        };
        unordered_set<int> A, B;
        for (int i = 0; i < moves.size(); ++i) {
            int pos = moves[i][0] * 3 + moves[i][1];
            if ((i % 2) == 0) {
                A.insert(pos);
                if (tictactoeCheck(A, wins)) {
                    return "A";
                }
            } else {
                B.insert(pos);
                if (tictactoeCheck(B, wins)) {
                    return "B";
                }
            }
        }
        return moves.size() == 9 ? "Draw" : "Pending";
    }

    vector<int> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        int x = 4 * cheeseSlices - tomatoSlices, y = tomatoSlices - 2 * cheeseSlices;
        if (x <= 0 || y <= 0 || x % 2 || y % 2) return {};
        return {x / 2, y / 2};
    }

    int countSquares(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        int res = 0;
        for (int i = 0; i < m; ++i) {
            dp[i][0] = matrix[i][0];
            res += dp[i][0];
        }
        for (int i = 1; i < n; ++i) {
            dp[0][i] = matrix[0][i];
            res += dp[0][i];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = min(dp[i][j - 1], min(dp[i - 1][j], dp[i - 1][j - 1])) + 1;
                    res += dp[i][j];
                }
            }
        }
        return res;
    }

    int palindromePartition(string s, int k) {
        int n = s.size();
        vector<vector<int>> dp(n + 1, vector<int>(k + 1));
        dp[0][0] = 0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= min(k, i); ++j) {
                if (j == 1) dp[i][j] = palindromePartitionCost(s, 0, i - 1);
                else {
                    for (int i0 = j - 1; i0 < i; ++i0) {
                        dp[i][j] = min(dp[i][j], dp[i0][j - 1] + palindromePartitionCost(s, i0, i - 1));
                    }
                }
            }
        }
        return dp[n][k];
    }

    int palindromePartitionCost(string &s, int l, int r) {
        int ret = 0;
        for (int i = l, j = r; i < j; ++i, --j) {
            if (s[i] != s[j]) {
                ++ret;
            }
        }
        return ret;
    }
};


#endif //LEETCODEMAC_SOLUTION3_H
