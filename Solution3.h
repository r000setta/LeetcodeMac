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
#include <numeric>

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

    string rankTeams(vector<string> &votes) {
        int n = votes.size();
        unordered_map<char, vector<int>> ranking;
        for (char v:votes[0]) {
            ranking[v].resize(votes[0].size());
        }
        for (const auto &vote:votes) {
            for (int i = 0; i < vote.size(); ++i) {
                ++ranking[vote[i]][i];
            }
        }

        using pcv=pair<char, vector<int>>;
        vector<pcv> result(ranking.begin(), ranking.end());
        sort(result.begin(), result.end(), [](const pcv &l, const pcv &r) {
            return l.second > r.second || (l.second == r.second && l.first < r.first);
        });
        string ans;
        for (auto&[v, r]:result) {
            ans += v;
        }
        return ans;
    }

    int uniquePaths(int m, int n) {
        int N = n + m - 2;
        double res = 1;
        for (int i = 1; i < m; i++)
            res = res * (N - (m - 1) + i) / i;
        return (int) res;
    }

    bool isSubPath(ListNode *head, TreeNode *root) {
        if (root == nullptr) return false;
        return isSubPathHelp(head, root) || isSubPath(head, root->left) || isSubPath(head, root->right);
    }

    bool isSubPathHelp(ListNode *head, TreeNode *root) {
        if (head == nullptr) return true;
        if (root == nullptr) return false;
        if (head->val != root->val) return false;
        return isSubPathHelp(head->next, root->left) || isSubPathHelp(head->next, root->right);
    }

    vector<int> getNoZeroIntegers(int n) {
        for (int i = 1; i <= n / 2; ++i) {
            if (getNoZeroIntegersCheck(i) && getNoZeroIntegersCheck(n - i))
                return {i, n - i};
        }
        return {};
    }

    bool getNoZeroIntegersCheck(int n) {
        while (n) {
            if ((n % 10) == 0) return false;
            n /= 10;
        }
        return true;
    }

    int minFlips(int a, int b, int c) {
        int res = 0;
        while (c != 0) {
            int tar = c & 1;
            int x = a == 0 ? 0 : a & 1;
            int y = b == 0 ? 0 : b & 1;
            if (tar == 1) {
                if (!(x || y)) res++;
            } else {
                if (x) res++;
                if (y) res++;
            }
            c = c >> 1;
            a = a >> 1;
            b = b >> 1;
        }
        while (a != 0) {
            if (a & 1) res++;
            a >>= 1;
        }
        while (b != 0) {
            if (b & 1) res++;
            b >>= 1;
        }
        return res;
    }

    vector<int> makeConnectedFa;

    int makeConnectedFind(int x) {
        return x == makeConnectedFa[x] ? x : makeConnectedFa[x] = makeConnectedFind(makeConnectedFa[x]);
    }

    int makeConnected(int n, vector<vector<int>> &connections) {
        if (connections.size() < n - 1) {
            return -1;
        }
        makeConnectedFa.resize(n);
        iota(makeConnectedFa.begin(), makeConnectedFa.end(), 0);
        int part = n;
        for (auto &&c:connections) {
            int p = makeConnectedFind(c[0]), q = makeConnectedFind(c[1]);
            if (p != q) {
                --part;
                makeConnectedFa[p] = q;
            }
        }
        return part - 1;
    }
};

#endif //LEETCODEMAC_SOLUTION3_H
