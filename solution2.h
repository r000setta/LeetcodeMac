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
        int x = matrix.size() - 1, y = 0;
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

    vector<int> findSquare(vector<vector<int>> &matrix) {

    }

    bool validMountainArray(vector<int> &A) {
        int n = A.size(), i = 0;

        while (i + 1 < n && A[i] < A[i + 1]) {
            i++;
        }
        if (i == 0 || i == n - 1) return false;
        while (i + 1 < n && A[i] > A[i + 1]) {
            i++;
        }
        return i == n - 1;
    }

    int numSubmat(vector<vector<int>> &mat) {
        vector<vector<int>> dp(mat.size(), vector<int>(mat[0].size()));
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                if (j == 0) {
                    dp[i][j] = mat[i][j];
                } else if (mat[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i][j - 1] + 1;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                int val = dp[i][j];
                for (int k = i; k >= 0 && val; --k) {
                    val = min(val, dp[k][j]);
                    res += val;
                }
            }
        }
        return res;
    }

    int existDir[4][2] = {{1,  0},
                          {-1, 0},
                          {0,  1},
                          {0,  -1}};

    bool exist(vector<vector<char>> &board, string word) {
        bool res = false;
        vector<vector<bool>> vis(board.size(), vector<bool>(board[0].size()));
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); ++j) {
                if (board[i][j] == word[0]) {
                    vis.assign(board.size(), vector<bool>(board[0].size(), false));
                    res = res || existDFS(board, word, 0, i, j, vis);
                    if (res) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool existDFS(vector<vector<char>> &board, string &word, int cur,
                  int x, int y, vector<vector<bool>> &vis) {
        if (cur == word.size() - 1) return true;
        vis[x][y] = true;
        bool res = false;
        for (const auto &dir:existDir) {
            int tx = x + dir[0];
            int ty = y + dir[1];
            if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() || vis[tx][ty] == true ||
                word[cur + 1] != board[tx][ty]) {
                continue;
            } else {
                res = res || existDFS(board, word, cur + 1, tx, ty, vis);
                if (res) {
                    return true;
                }
            }
        }
        return false;
    }

    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int over = 0;
        if (!(E >= C || G <= A || F >= D || H <= B)) {
            int x1 = max(A, E);
            int y1 = max(B, F);
            int x2 = min(C, G);
            int y2 = min(D, H);
            over = (x2 - x1) * (y2 - y1);
        }

        return (C - A) * (D - B) + ((G - E) * (H - F) - over);
    }

    int maximalRectangle(vector<vector<char>> &matrix) {
        int res = 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == 0) {
                    dp[i][j] = matrix[i][j] - '0';
                } else if (matrix[i][j] == '1') {
                    dp[i][j] = dp[i][j - 1] + 1;
                } else {
                    dp[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int val = dp[i][j];
                for (int k = i; k >= 0 && val; --k) {
                    val = min(val, dp[k][j]);
                    res = max(res, val * (i - k + 1));
                }
            }
        }
        return res;
    }

    int maximalSquare(vector<vector<char>> &matrix) {
        if (matrix.size() == 0) return 0;
        int res = 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) dp[i][j] = matrix[i][j] - '0';
                else if (matrix[i][j] == '0') dp[i][j] = 0;
                else {
                    dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
                res = max(res, dp[i][j]);
            }
        }
        return res * res;
    }

    int countSquares(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) {
                    dp[i][j] = matrix[i][j];
                } else if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = min(min(dp[i][j - 1], dp[i - 1][j]),
                                   dp[i - 1][j - 1]) + 1;
                }
                res += dp[i][j];
            }
        }
        return res;
    }


    vector<int> spiralOrder(vector<vector<int>> &matrix) {
        if (matrix.size() == 0) return {};
        int l = 0, r = matrix[0].size() - 1, t = 0, b = matrix.size() - 1, x = 0;
        vector<int> res((r + 1) * (b + 1));
        while (true) {
            for (int i = l; i <= r; ++i) res[x++] = matrix[t][i];
            if (++t > b) break;
            for (int i = t; i <= b; i++) res[x++] = matrix[i][r];
            if (l > --r) break;
            for (int i = r; i >= l; i--) res[x++] = matrix[b][i];
            if (t > --b) break;
            for (int i = b; i >= t; i--) res[x++] = matrix[i][l];
            if (++l > r) break;
        }
        return res;
    }

    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n));
        int l = 0, t = 0, r = n - 1, b = n - 1, k = 1;
        while (true) {
            for (int i = l; i <= r; ++i) {
                res[t][i] = k;
                k++;
            }
            if (++t > b) break;
            for (int i = t; i <= b; i++) {
                res[i][r] = k;
                k++;
            }
            if (l > --r) break;
            for (int i = r; i >= l; --i) {
                res[b][i] = k;
                k++;
            }
            if (--b < t) break;
            for (int i = b; i >= t; --i) {
                res[i][l] = k;
                k++;
            }
            if (++l > r) break;
        }
        return res;
    }

    void setZeroes(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<bool> rows(m);
        vector<bool> cols(n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    if (!rows[i]) {
                        rows[i] = true;
                    }
                    if (!cols[j]) {
                        cols[j] = true;
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            if (rows[i]) {
                for (int j = 0; j < n; ++j) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (cols[i]) {
                for (int j = 0; j < m; ++j) {
                    matrix[j][i] = 0;
                }
            }
        }
    }

    vector<vector<int>> transpose(vector<vector<int>> &A) {
        int m = A.size(), n = A[0].size();
        vector<vector<int>> res(n, vector<int>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res[j][i] = A[i][j];
            }
        }
        return res;
    }

    vector<vector<int>> matrixBlockSum(vector<vector<int>> &mat, int K) {
        int m = mat.size(), n = mat[0].size();
        vector<vector<int>> res(m, vector<int>(n));
        vector<vector<int>> pre(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + mat[i - 1][j - 1];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res[i][j] = matrixBlockSumGet(pre, m, n, i + K + 1, j + K + 1)
                            - matrixBlockSumGet(pre, m, n, i + K + 1, j - K) -
                            matrixBlockSumGet(pre, m, n, i - K, j + K + 1)
                            + matrixBlockSumGet(pre, m, n, i - K, j - K);
            }
        }
        return res;
    }

    int matrixBlockSumGet(vector<vector<int>> &pre, int m, int n, int i, int j) {
        int x = max(min(i, m), 0);
        int y = max(min(j, n), 0);
        return pre[x][y];
    }

    int numMagicSquaresInside(vector<vector<int>> &grid) {
        if (grid.size() < 3 || grid[0].size() < 3) return 0;
        int res = 0;
        for (int i = 0; i < grid.size() - 2; ++i) {
            for (int j = 0; j < grid[0].size() - 2; ++j) {
                if (numMagicSquaresInsideCheck(grid, i, j)) {
                    res++;
                }
            }
        }
        return res;
    }

    bool numMagicSquaresInsideCheck(vector<vector<int>> &grid, int x, int y) {
        if (grid[x + 1][y + 1] != 5) return false;
        unordered_set<int> tmp;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int num = grid[x + i][y + j];
                if (num > 9 || num < 1) return false;
                tmp.insert(num);
            }
        }
        if (tmp.size() != 9) return false;
        int tar = grid[x][y] + grid[x + 1][y + 1] + grid[x + 2][y + 2];
        for (int i = 0; i < 3; ++i) {
            int res = 0;
            for (int j = 0; j < 3; ++j) {
                int cur = grid[x + i][y + j];
                res += cur;
            }
            if (res != tar) return false;
        }
        for (int i = 0; i < 3; ++i) {
            int res = 0;
            for (int j = 0; j < 3; ++j) {
                res += grid[x + j][y + i];
            }
            if (res != tar) return false;
        }

        return grid[x + 2][y] + grid[x + 1][y + 1] + grid[x][y + 2] == tar;
    }

    vector<int> luckyNumbers(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<int> rmin(m, INT_MAX);
        vector<int> cmax(n, 0);
        vector<int> ans;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                rmin[i] = min(rmin[i], matrix[i][j]);
                cmax[j] = max(cmax[j], matrix[i][j]);
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == rmin[i] && matrix[i][j] == cmax[j]) {
                    ans.push_back(matrix[i][j]);
                }
            }
        }
        return ans;
    }

    vector<vector<int>> diagonalSort(vector<vector<int>> &mat) {
        int n = mat.size(), m = mat[0].size();
        unordered_map<int, vector<int>> vs;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                vs[i - j].emplace_back(mat[i][j]);
            }
        }
        for (auto &v:vs) sort(v.second.begin(), v.second.end());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                mat[i][j] = vs[i - j].back();
                vs[i - j].pop_back();
            }
        }
        return mat;
    }
};

#endif //LEETCODEMAC_SOLUTION2_H
