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
#include "solution.h"

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

//    string rankTeams(vector<string> &votes) {
//        int n = votes.size();
//        unordered_map<char, vector<int>> ranking;
//        for (char v:votes[0]) {
//            ranking[v].resize(votes[0].size());
//        }
//        for (const auto &vote:votes) {
//            for (int i = 0; i < vote.size(); ++i) {
//                ++ranking[vote[i]][i];
//            }
//        }
//
//        using pcv = pair<char, vector<int>>;
//        vector<pcv> result(ranking.begin(), ranking.end());
//        sort(result.begin(), result.end(), [](const pcv &l, const pcv &r) {
//            return l.second > r.second || (l.second == r.second && l.first < r.first);
//        });
//        string ans;
//        for (auto&[v, r]:result) {
//            ans += v;
//        }
//        return ans;
//    }

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

    bool lemonadeChange(vector<int> &bills) {
        vector<int> vec(11);
        for (int i:bills) {
            if (i == 5) {
                vec[5]++;
            } else if (i == 10) {
                if (vec[5] > 0) {
                    vec[5]--;
                    vec[10]++;
                } else {
                    return false;
                }
            } else if (i == 20) {
                if (vec[10] > 0 && vec[5] > 0) {
                    vec[10]--;
                    vec[5]--;
                } else if (vec[5] >= 3) {
                    vec[5] -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    int wiggleMaxLength(vector<int> &nums) {
        int up = 1, down = 1;
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] > nums[i - 1]) {
                up = down + 1;
            } else if (nums[i] < nums[i - 1]) {
                down = up + 1;
            }
        }
        return nums.size() == 0 ? 0 : max(down, up);
    }

    int minTimeToVisitAllPoints(vector<vector<int>> &points) {
        int res = 0;
        for (int i = 1; i < points.size(); ++i) {
            int x = abs(points[i][0] - points[i - 1][0]);
            int y = abs(points[i][1] - points[i - 1][1]);
            res += min(x, y) + abs(x - y);
        }
        return res;
    }

    int countServers(vector<vector<int>> &grid) {
        int m = grid.size(), n = grid[0].size();
        vector<int> cntm(m), cntn(n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    ++cntm[i];
                    ++cntn[j];
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1 && (cntm[i] > 1 || cntn[j] > 1)) {
                    ++ans;
                }
            }
        }
        return ans;
    }

    struct Trie {
        unordered_map<char, Trie *> child;
        priority_queue<string> words;
    };

    void addWord(Trie *root, const string &word) {
        Trie *cur = root;
        for (const char &ch:word) {
            if (!cur->child.count(ch)) {
                cur->child[ch] = new Trie();
            }
            cur->words.push(word);
            if (cur->words.size() > 3) {
                cur->words.pop();
            }
        }
    }

    vector<vector<string>> suggestedProducts(vector<string> &products, string searchWord) {
        Trie *root = new Trie();
        for (const auto &word:products) {
            addWord(root, word);
        }
        vector<vector<string>> ans;
        Trie *cur = root;
    }

    vector<vector<int>> shiftGrid(vector<vector<int>> &grid, int k) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> res = grid;
        vector<int> arr;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                arr.push_back(grid[i][j]);
            }
        }
        vector<int> arr1 = arr;
        for (int i = 0; i < arr.size(); ++i) {
            arr1[(k + i) % (n * m)] = arr[i];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res[i][j] = arr1[i * m + j];
            }
        }
        return res;
    }

    int maxSumDivThree(vector<int> &nums) {
        int n = nums.size();
        vector<vector<int>> dp(n + 1, vector<int>(3, 0));
        dp[0][0] = 0, dp[0][1] = INT_MIN, dp[0][2] = INT_MIN;
        for (int i = 1; i <= n; ++i) {
            if (nums[i - 1] % 3 == 0) {
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][0] + nums[i - 1]);
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][1] + nums[i - 1]);
                dp[i][2] = max(dp[i - 1][2], dp[i - 1][2] + nums[i - 1]);
            } else if (nums[i - 1] % 3 == 1) {
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] + nums[i - 1]);
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + nums[i - 1]);
                dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + nums[i - 1]);
            } else if (nums[i - 1] % 3 == 2) {
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + nums[i - 1]);
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][2] + nums[i - 1]);
                dp[i][2] = max(dp[i - 1][2], dp[i - 1][0] + nums[i - 1]);
            }
        }
        return dp[n][0];
    }

    bool containsDuplicate(vector<int> &nums) {
        unordered_set<int> cnt;
        for (int n:nums) {
            if (cnt.count(n)) return true;
            cnt.insert(n);
        }
        return false;
    }

    int subtractProductAndSum(int n) {
        unsigned int sum = 0;
        unsigned int pro = 1;
        while (n != 0) {
            int t = n % 10;
            sum += t;
            pro *= t;
            n /= 10;
        }
        return pro - sum;
    }

//    vector<vector<int>> groupThePeople(vector<int> &groupSizes) {
//        unordered_map<int, vector<int>> groups;
//        for (int i = 0; i < groupSizes.size(); ++i) {
//            groups[groupSizes[i]].push_back(i);
//        }
//        vector<vector<int>> ans;
//        for (auto&[gsize, users]:groups) {
//            for (auto iter = users.begin(); iter != users.end(); iter = next(iter, gsize)) {
//                ans.emplace_back(iter, next(iter, gsize));
//            }
//        }
//        return ans;
//    }

    int numberOfMatches(int n) {
        int res = 0;
        while (n != 0) {
            if (n % 2) {
                res += (n - 1) / 2;
                n = (n - 1) / 2 + 1;
            } else {
                res += n / 2;
                n /= 2;
            }
        }
        return res;
    }

    int minPartitions(string n) {
        int m = -1;
        for (char c:n) {
            m = max(m, c - '0');
        }
        return m;
    }

    int stoneGameVII(vector<int> &stones) {
        int m = stones.size();
        vector<int> pre(m + 1);
        pre[0] = 0;
        for (int i = 1; i < pre.size(); ++i) {
            pre[i] = pre[i - 1] + stones[i - 1];
        }
        int a = 0, b = 0;
        bool flag = true;
        int l = 0, r = stones.size() - 1;
        while (l <= r) {
            if (flag) {
                if (stones[l] > stones[r]) {
                    r--;
                } else {
                    l++;
                }
                a += pre[r + 1] - pre[l];
                flag = !flag;
            } else {
                if (r - l <= 2) {
                    if (stones[l] > stones[r]) {
                        r--;
                    } else {
                        l++;
                    }
                    b += pre[r + 1] - pre[l];
                    flag = !flag;
                } else {
                    if (stones[l + 1] > stones[r - 1]) {
                        l++;
                    } else {
                        r--;
                    }
                    b += pre[r + 1] - pre[l];
                    flag = !flag;
                }
            }
        }
        return a - b;
    }

    bool stoneGame(vector<int> &piles) {
        int m = piles.size();
        vector<vector<int>> dp(m, vector<int>(m));
        for (int i = 0; i < m; ++i) {
            dp[i][i] = piles[i];
        }
        for (int i = m - 2; i >= 0; --i) {
            for (int j = i + 1; j < m; ++j) {
                dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
            }
        }
        return dp[0][m - 1] > 0;
    }

    int getDecimalValue(ListNode *head) {
        ListNode *cur = head;
        int ans = 0;
        while (cur != nullptr) {
            ans=ans*2+cur->val;
            cur=cur->next;
        }
        return ans;
    }

    vector<vector<string>> groupAnagrams(vector<string> &strs) {
        unordered_map<string, vector<string>> mp;
        for (string &str:strs) {
            string key = str;
            sort(key.begin(), key.end());
            mp[key].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};

#endif //LEETCODEMAC_SOLUTION3_H
