#ifndef LEETCODEMAC_DPSOLUTION_H
#define LEETCODEMAC_DPSOLUTION_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <numeric>
#include <unordered_set>
#include "solution.h"

using namespace std;

class DPSolution {
public:
    string longestDiverseString(int a, int b, int c) {
        vector<vector<char>> v;
        v.push_back({(char) a, 'a'});
        v.push_back({(char) b, 'b'});
        v.push_back({(char) c, 'c'});
        string res;
        while (res.size() < a + b + c) {
            sort(v.rbegin(), v.rend());
            if ((res.size() > 0) && (res.back() == v[0][1])) {
                if (v[1][0]-- > 0) res.push_back(v[1][1]);
                else return res;
            } else {
                if (v[0][0]-- > 0) res.push_back(v[0][1]);
                if (v[0][0]-- > 0) res.push_back(v[0][1]);
            }
        }
        return res;
    }

    int maxSumAfterPartitioning(vector<int> &arr, int k) {
        int n = arr.size();
        vector<int> dp(n + 1);
        for (int i = 0; i <= n; ++i) {
            int curMax = 0;
            for (int j = i - 1; (i - j) < k && j >= 0; --j) {
                curMax = max(curMax, arr[j]);
                dp[i] = max(dp[i], dp[i] + (i - j) * curMax);
            }
        }
        return dp[n];
    }

    int splitArray(vector<int> &nums, int m) {
        int n = nums.size();
        vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, LLONG_MAX));
        vector<long long> sub(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            sub[i + 1] = sub[i] + nums[i];
        }
        dp[0][0] = 0;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= min(i, m); ++j) {
                for (int k = 0; k < i; ++k) {
                    dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sub[i] - sub[k]));
                }
            }
        }
        return (int) dp[n][m];
    }

    int rob(vector<int> &nums) {
        if (nums.size() == 0) return 0;
        if (nums.size() == 1) return nums[0];
        vector<int> dp(nums.size());
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < nums.size(); ++i) {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.size() - 1];
    }

    int rob2(vector<int> &nums) {
        int n = nums.size();
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        return max(rob2Help(nums, 0, n - 2), rob2Help(nums, 1, n - 1));
    }

    int rob2Help(vector<int> &nums, int l, int r) {
        int dpi = 0;
        int dpi_2 = 0;
        int dpi_1 = 0;
        for (int i = l; i <= r; ++i) {
            dpi = max(dpi_1, nums[i] + dpi_2);
            dpi_2 = dpi_1;
            dpi_1 = dpi;
        }
        return dpi;
    }

    int rob3(TreeNode *root) {
        unordered_map<TreeNode *, int> memo;
        return rob3Help(root, memo);
    }

    int rob3Help(TreeNode *root, unordered_map<TreeNode *, int> &memo) {
        if (root == nullptr) return 0;
        if (memo.count(root)) return memo[root];
        int mon = root->val;
        if (root->right != nullptr) mon += rob3Help(root->right->left, memo) + rob3Help(root->right->right, memo);
        if (root->left != nullptr) mon += rob3Help(root->left->left, memo) + rob3Help(root->left->right, memo);
        mon = max(mon, rob3Help(root->left, memo) + rob3Help(root->right, memo));
        memo[root] = mon;
        return mon;
    }

    vector<vector<int>> kClosest(vector<vector<int>> &points, int K) {
        priority_queue<pair<int, int>> q;
        for (int i = 0; i < K; ++i) {
            q.emplace(points[i][0] * points[i][0] + points[i][1] * points[i][1], i);
        }
        int n = points.size();
        for (int i = K; i < n; ++i) {
            int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, i);
            }
        }
        vector<vector<int>> ans;
        while (!q.empty()) {
            ans.push_back(points[q.top().second]);
            q.pop();
        }
        return ans;
    }

    string shortestPalindrome(string s) {
        int n = s.size();
        int base = 131, mod = 1e7;
        int left = 0, right = 0, mul = 1;
        int best = -1;
        for (int i = 0; i < n; ++i) {
            left = ((long long) left * base + s[i]) % mod;
            right = (right + (long long) mul * s[i]) % mod;
            if (left == right) best = i;
            mul = ((long long) mul * base) % mod;
        }
        string add = (best == n - 1 ? "" : s.substr(best + 1, n));
        reverse(add.begin(), add.end());
        return add + s;
    }

    int lcs(string s1, string s2) {
        int m1 = s1.size(), m2 = s2.size();
        vector<vector<int>> dp(m1 + 1, vector<int>(m2 + 1));
        for (int i = 1; i <= m1; ++i) {
            for (int j = 1; j <= m2; ++j) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m1][m2];
    }

    bool isMatch(string s, string p) {
        int m = p.size(), n = s.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= m; ++i) {
            if (p[i - 1] != '*') break;
            dp[i][0] = true;
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[i - 1] == s[j - 1] || p[i - 1] == '?') dp[i][j] = dp[i - 1][j - 1];
                else if (p[i - 1] == '*') dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            }
        }
        return dp[m][n];
    }

    bool isMatch2(string s, string p) {
        int m = p.size(), n = s.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
        dp[0][0] = true;
        for (int i = 1; i <= m; ++i) {
            if (p[i - 1] == '*' && dp[i - 2][0]) dp[i][0] = true;
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[i - 1] == s[j - 1] || p[i - 1] == '.')
                    dp[i][j] = dp[i - 1][j - 1];
                else if (p[i - 1] == '*') {
                    if (p[i - 2] == s[j - 1] || p[i - 2] == '.') {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1] || dp[i - 2][j];
                    } else if (p[i - 2] != s[j - 1]) {
                        dp[i][j] = dp[i - 2][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    int cuttingRope(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 1;
        if (n == 3) return 2;
        vector<unsigned long long> dp(n + 1);
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for (unsigned long long i = 4; i <= n; ++i) {
            for (int j = 1; j < i; ++j) {
                dp[i] = max(i, max(dp[i], j * dp[i - j]));
            }
        }
        return dp[n] % 1000000007 == 1000000008 ? -1 : dp[n] % 1000000007;
    }

    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for (int i = 0; i <= m; ++i) {
            dp[i][0] = i;
        }
        for (int i = 0; i < n; ++i) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {

            }
        }
    }

    int maxDotProduct(vector<int> &nums1, vector<int> &nums2) {
        int m = nums1.size(), n = nums2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MIN));
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                dp[i][j] = max(dp[i][j], nums1[i - 1] * nums2[j - 1]);
                dp[i][j] = max(dp[i][j], nums1[i - 1] * nums2[j - 1] + dp[i - 1][j - 1]);
                dp[i][j] = max(dp[i][j], dp[i][j - 1]);
                dp[i][j] = max(dp[i][j], dp[i - 1][j]);
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1]);
            }
        }
        return dp[m][n];
    }

    bool isContinue(char prev, char curr) {
        if (prev == 'z') return curr == 'a';
        return prev + 1 == curr;
    }

    int findSubstringInWraproundString(string p) {
        vector<int> dp(26, 0);
        int n = p.size();
        int k = 0;
        for (int i = 0; i < n; ++i) {
            if (i > 0 && isContinue(p[i - 1], p[i])) {
                ++k;
            } else {
                k = 1;
            }
            dp[p[i] - 'a'] = max(dp[p[i] - 'a'], k);
        }
        return accumulate(dp.begin(), dp.end(), 0);
    }

    int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 0;
        if (n == 1) return 9;
        vector<int> dp(n + 1);
        dp[0] = 1;
        dp[1] = 9;
        for (int i = 2; i <= n; ++i) {
            int k = 9, tar = 9;
            for (int j = i - 1; j > 0; --j) {
                tar *= k;
                --k;
            }
            dp[i] = tar;
        }
        return accumulate(dp.begin(), dp.end(), 0);
    }

    int findTargetSumWays(vector<int> &nums, int S) {
        vector<vector<int>> dp(nums.size(), vector<int>(2001));
        dp[0][nums[0] + 1000] = 1;
        dp[0][-nums[0] + 1000] = 1;
    }

    int lastStoneWeightII(vector<int> &stones) {

    }

    int lastStoneWeight(vector<int> &stones) {
        priority_queue<int> p(stones.begin(), stones.end());
        while (!p.empty() && p.size() >= 2) {
            int a = p.top();
            p.pop();
            int b = p.top();
            p.pop();
            if (a != b) {
                p.emplace(a - b);
            }
        }
        return p.empty() ? 0 : p.top();
    }

    int coinChange(vector<int> &coins, int amount) {
        if (amount == 0) return 0;
        if (coins.empty()) return -1;
        vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; ++i) {
            for (int coin:coins) {
                if (i >= coin) {
                    dp[i] = min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    bool canPartition(vector<int> &nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2) return false;
        sum /= 2;
        int m = nums.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(sum + 1));
        for (int i = 0; i <= m; ++i) {
            dp[i][0] = true;
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= sum; ++j) {
                if (j < nums[i - 1]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[m][sum];
    }

    int findMaxForm(vector<string> &strs, int m, int n) {
        int len = strs.size();
        vector<vector<vector<int>>> dp(len + 1, vector<vector<int>>(m + 1, vector<int>(n + 1)));
        for (int i = 1; i <= len; ++i) {
            string s = strs[i];
            int c0 = 0, c1 = 0;
            for (char c:s) {
                if (c == '0') c0++;
                else c1++;
            }
            for (int j = 0; j <= m; ++j) {
                for (int k = 0; k <= n; ++k) {
                    if (c0 > j || c1 > k) {
                        dp[i][j][k] = dp[i - 1][j][k];
                    } else {
                        dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - c0][k - c1] + 1);
                    }
                }
            }
        }
        return dp[len][m][n];
    }
};

#endif //LEETCODEMAC_DPSOLUTION_H
