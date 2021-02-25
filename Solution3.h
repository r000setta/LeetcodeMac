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
        int n = stones.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (i == j) dp[i][j] = stones[i];
                else dp[i][j] = stones[j] + dp[i][j - 1];
            }
        }
        vector<vector<int>> res(n, vector<int>(n));
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                if (j - i == 1) res[i][j] = max(stones[i], stones[j]);
                else res[i][j] = max(dp[i + 1][j] - res[i + 1][j], dp[i][j - 1] - res[i][j - 1]);
            }
        }
        return res[0][n - 1];
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
            ans = ans * 2 + cur->val;
            cur = cur->next;
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

    int monotoneIncreasingDigits(int N) {
        string strN = to_string(N);
        int i = 1;
        while (i < strN.length() && strN[i - 1] <= strN[i]) {
            i += 1;
        }
        if (i < strN.length()) {
            while (i > 0 && strN[i - 1] > strN[i]) {
                strN[i - 1] -= 1;
                i -= 1;
            }
            for (i += 1; i < strN.length(); ++i) {
                strN[i] = '9';
            }
        }
        return stoi(strN);
    }

    bool wordPattern(string pattern, string str) {
        unordered_map<string, char> str2ch;
        unordered_map<char, string> ch2str;
        int m = str.size();
        int i = 0;
        for (auto ch:pattern) {
            if (i >= m) return false;

        }
    }

    int maxProfit(vector<int> &prices, int fee) {
        int m = prices.size();
        int dp[65535][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < m; ++i) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[m - 1][0];
    }

    int minCostClimbingStairs(vector<int> &cost) {

    }

    string reformatNumber(string number) {
        string tmp = "";
        for (char c:number) {
            if (isdigit(c)) {
                tmp += c;
            }
        }
        string res = "";
    }

    vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
        vector<vector<int>> res;
        int level = 0;
        if (!root) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            vector<int> tmp(size);
            for (int i = 0; i < size; ++i) {
                auto t = q.front();
                q.pop();
                if (level % 2) tmp[size - i - 1] = t->val;
                else tmp[i] = t->val;
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
            level++;
            res.emplace_back(tmp);
        }
        return res;
    }

    int candy(vector<int> &ratings) {
        int n = ratings.size();
        vector<int> left(n);
        for (int i = 0; i < n; ++i) {
            if (i > 0 && ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            } else {
                left[i] = 1;
            }
        }
        int right = 0, ret = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (i < n - 1 && ratings[i] > ratings[i + 1]) {
                right++;
            } else {
                right = 1;
            }
            ret += max(left[i], right);
        }
        return ret;
    }

    int countPairs(vector<int> &deliciousness) {
        unordered_map<int, int> mp;
        auto MOD = 1000000007;
        int ans = 0;
        int len = deliciousness.size();
        for (int num:deliciousness) {
            int p = 1;
            for (int i = 0; i <= 21; ++i) {
                if (p >= num && mp.count(p - num)) {
                    ans += mp[p - num];
                    ans %= MOD;
                }
                p *= 2;
            }
            mp[num]++;
        }
        return ans;
    }

    int waysToSplit(vector<int> &nums) {
        int m = nums.size();
        if (m < 3) return 0;
        vector<int> pre(m + 1);
        pre[0] = 0, pre[1] = nums[0];
        for (int i = 2; i < pre.size(); ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        int res = 0;
        auto MOD = 1000000007;
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                int left = pre[i + 1] - pre[0];
                int mid = pre[j + 1] - pre[i + 1];
                int right = pre[m] - pre[j + 1];
                if (left <= mid && mid <= right) {
                    res++;
                    res %= MOD;
                }
            }
        }
        return res;
    }

    bool halvesAreAlike(string s) {
        int r1 = 0, r2 = 0;
        for (int i = 0; i < s.size() / 2; ++i) {
            if (halvesAreAlikeCheck(s[i])) {
                r1++;
            }
        }
        for (int i = s.size() / 2; i < s.size(); ++i) {
            if (halvesAreAlikeCheck(s[i])) {
                r2++;
            }
        }
        return r1 == r2;
    }

    inline bool halvesAreAlikeCheck(char a) {
        a = tolower(a);
        return a == 'a' || a == 'o' || a == 'e' || a == 'i' || a == 'u';
    }

#define PII pair<int,int>

    int eatenApples(vector<int> &apples, vector<int> &days) {
        int res = 0;
        priority_queue<PII, vector<PII >, greater<>> save;
        for (int i = 0; i < apples.size() || !save.empty(); ++i) {
            while (!save.empty() && save.top().first == i) {
                save.pop();
            }
            if (i < apples.size() && apples[i] != 0) {
                save.push(make_pair(i + days[i], apples[i]));
            }
            if (!save.empty()) {
                PII tmp = save.top();
                save.pop();
                res++;
                tmp.second--;
                if (tmp.second > 0) {
                    save.push(tmp);
                }
            }
        }
        return res;
    }

    int firstUniqChar(string s) {
        vector<int> vec(26);
        for (char c:s) {
            vec[c - 'a']++;
        }
        for (int i = 0; i < s.size(); ++i) {
            if (vec[s[i] - 'a'] == 1) {
                return i;
            }
        }
        return -1;
    }

    int maximalRectangle(vector<vector<char>> &matrix) {
        if (matrix.size() == 0) return 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == 0) {
                    dp[i][j] = matrix[i][j] - '0';
                } else {
                    if (matrix[i][j] == '1') {
                        dp[i][j] = dp[i][j - 1] + 1;
                    }
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dp[i][j] != 0) {
                    int tmp = dp[i][j];
                    for (int k = i; k >= 0; --k) {
                        if (dp[k][j] == 0) {
                            break;
                        }
                        int height = i - k + 1;
                        tmp = min(tmp, dp[k][j]);
                        res = max(res, tmp * height);
                    }
                }
            }
        }
        return res;
    }

    bool isIsomorphic(string s, string t) {
        unordered_map<char, char> smap;
        unordered_map<char, char> tmap;
        if (s.size() != t.size()) return false;
        for (int i = 0; i < s.size(); ++i) {
            char x = s[i], y = t[i];
            if (smap.count(x) && smap[x] != y || tmap.count(y) && tmap[y] != x) return false;
            smap[x] = y;
            tmap[y] = x;
        }
        return true;
    }

    int pivotIndex(vector<int> &nums) {
        int m = nums.size();
        vector<int> pre(m + 1);
        pre[0] = 0;
        for (int i = 1; i <= m; ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        for (int i = 0; i < m; ++i) {
            int x, y;
            if (i == 0) x = 0; else x = pre[i];
            if (i == m - 1) y = 0; else y = pre[m] - pre[i + 1];
            if (x == y) return i;
        }
        return -1;
    }

    int maximumProduct(vector<int> &nums) {
        sort(nums.begin(), nums.end());
        int m = nums.size();
        return max(nums[0] * nums[1] * nums[m - 1], nums[m - 1] * nums[m - 2] * nums[m - 3]);
    }

    int smallFind(vector<int> &parent, int x) {
        if (parent[x] != x) {
            parent[x] = smallFind(parent, parent[x]);
        }
        return parent[x];
    }

    void smallUnion(vector<int> &parent, int x, int y) {
        int px = smallFind(parent, x);
        int py = smallFind(parent, y);
        if (px != py) {
            parent[px] = py;
        }
    }

//    string smallestStringWithSwaps(string s, vector<vector<int>> &pairs) {
//        int n = pairs.size();
//        vector<int> parent(n);
//        vector<char> res(n);
//        for (int i = 0; i < n; ++i) parent[i] = i;
//        for (const auto &p:pairs) {
//            smallUnion(parent, p[0], p[1]);
//        }
//        unordered_map<int, vector<int>> mp;
//        for (int i = 0; i < n; ++i) {
//            mp[smallFind(parent, i)].push_back(i);
//        }
//        for (auto&[k, v]:mp) {
//            vector<int> c = v;
//            sort(v.begin(), v.end(), [&](auto a, auto b) {
//                return s[a] < s[b];
//            });
//            for (int i = 0; i < c.size(); ++i) {
//                res[c[i]] = s[v[i]];
//            }
//        }
//        s = "";
//        for (const auto &e:res) s += e;
//        return s;
//    }

    int calculate(string s) {
        int x = 1, y = 0;
        for (const auto &c:s) {
            if (c == 'A') {
                x = x * 2 + y;
            } else {
                y = y * 2 + x;
            }
        }
        return x + y;
    }

    vector<int> fraction(vector<int> &cont) {
        int high = 1, low = *cont.rbegin();
        for (int i = cont.size() - 2; i >= 0; --i) {
            high += low * cont[i];
            swap(low, high);
        }
        return vector<int>{low, high};
    }

    int numWays(int n, vector<vector<int>> &relation, int k) {
        vector<vector<int>> dp(k + 1, vector<int>(n + 1));
        dp[0][0] = 1;
        for (int i = 0; i < k; ++i) {
            for (const auto &r:relation) {
                dp[i + 1][r[1]] += dp[i][r[0]];
            }
        }
        return dp[k][n - 1];
    }

//    int minimumEffortPath(vector<vector<int>> &heights) {
//        int dirs[4][2] = {{-1, 0},
//                          {1,  0},
//                          {0,  1},
//                          {0,  -1}};
//        int m = heights.size(), n = heights[0].size();
//        int left = 0, right = 99999, ans = 0;
//        while (left <= right) {
//            int mid = (left + right) / 2;
//            queue<pair<int, int>> q;
//            q.emplace(0, 0);
//            vector<bool> s(m * n);
//            s[0] = true;
//            while (!q.empty()) {
//                auto[x, y]=q.front();
//                q.pop();
//                for (auto &dir:dirs) {
//                    int nx = x + dir[0];
//                    int ny = y + dir[1];
//                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !s[nx * n + ny] &&
//                        abs(heights[x][y] - heights[nx][ny]) <= mid) {
//                        q.emplace(nx, ny);
//                        s[nx * n + ny] = true;
//                    }
//                }
//            }
//            if (s[m * n - 1]) {
//                ans = mid;
//                right = mid - 1;
//            } else {
//                left = mid + 1;
//            }
//        }
//        return ans;
//    }

    int paintingPlanHelp(int n, int a) {
        if (a == 0) return 1;
        int x = n, y = a, t = a;
        for (int i = 0; i < t - 1; ++i) {
            n--;
            a--;
            x *= n;
            y *= a;
        }
        return (int) x / y;
    }

    int paintingPlan(int n, int k) {
        if (k == n * n) return 1;
        if (k == 0) return 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (n * (i + j) - i * j == k) {
                    res += paintingPlanHelp(n, i) + paintingPlanHelp(n, j);
                }
            }
        }
        return res;
    }

    int breakfastNumber(vector<int> &staple, vector<int> &drinks, int x) {
        int res = 0;
        int m = staple.size();
        int mod = 1e9 + 7;
        sort(staple.begin(), staple.end());
        sort(drinks.begin(), drinks.end());
        for (int i = 0; i < m; ++i) {
            if (staple[i] >= x) break;
            for (int j = 0; j < m; ++j) {
                if (staple[i] + staple[j] <= x) {
                    res++;
                } else {
                    break;
                }
            }
        }
        return res % mod;
    }

    int coinChange(vector<int> &coins, int amount) {
        vector<int> dp(amount + 1);
        dp[0] = 0;
        for (int coin:coins) {
            for (int i = coin; i <= amount; ++i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
        if (dp[amount] == amount + 1) {
            dp[amount] = -1;
        }
        return dp[amount];
    }

    int numSimilarGroupsFind(vector<int> &parent, int x) {
        if (parent[x] != x) {
            parent[x] = numSimilarGroupsFind(parent, parent[x]);
        }
        return parent[x];
    }

    void numSimilarGroupsUnion(vector<int> &parents, int x, int y) {
        int px = numSimilarGroupsFind(parents, x);
        int py = numSimilarGroupsFind(parents, y);
        if (px != py) {
            parents[px] = py;
        }
    }

    int numSimilarGroups(vector<string> &strs) {
        int n = strs.size();
        vector<int> parent(n);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                string s1 = strs[i], s2 = strs[j];
                if (isSimilar(s1, s2)) {
                    numSimilarGroupsUnion(parent, i, j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (parent[i] == i) {
                res++;
            }
        }
        return res;
    }

    bool isSimilar(string s1, string s2) {
        if (s1.size() != s2.size()) return false;
        int cnt = 0;
        for (int i = 0; i < s1.size(); ++i) {
            if (s1[i] != s2[i]) {
                cnt++;
            }
        }
        if (cnt == 2 || cnt == 0) return true;
        return false;
    }

    int singleNumber(vector<int> &nums) {
        int res = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            res = res ^ nums[i];
        }
        return res;
    }

    vector<int> fairCandySwap(vector<int> &A, vector<int> &B) {
        int sa = accumulate(A.begin(), A.end(), 0);
        int sb = accumulate(B.begin(), B.end(), 0);
        int tmp = (sa - sb) / 2;
        vector<int> res(2);
        for (int i:A) {
            for (int j:B) {
                if (i - j == tmp) {
                    res[0] = i;
                    res[1] = j;
                    return res;
                }
            }
        }
        return res;
    }

    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
        int rv = root->val, pv = p->val, qv = q->val;
        if (rv > pv && rv > qv) {
            return lowestCommonAncestor(root->left, p, q);
        } else if (rv < pv && rv < qv) {
            return lowestCommonAncestor(root->right, p, q);
        } else {
            return root;
        }
    }

    int majorityElement(vector<int> &nums) {
        int count = 1, candidate = -1;
        for (int i:nums) {
            if (candidate == i) {
                count++;
            } else if (--count == 0) {
                candidate = i;
                count = 1;
            }
        }
        return candidate;
    }

    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        ListNode *res = new ListNode(0);
        ListNode *tmp = res;
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                tmp->next = l1;
                l1 = l1->next;
                tmp = tmp->next;
            } else {
                tmp->next = l2;
                l2 = l2->next;
                tmp = tmp->next;
            }
        }
        if (l1) {
            tmp->next = l1;
        }
        if (l2) {
            tmp->next = l2;
        }
        return res->next;
    }

    bool isPalindrome(int x) {
        string a = to_string(x);
        int l = 0, r = a.size() - 1;
        while (l < r) {
            if (a[l] != a[r]) return false;
            l++;
            r--;
        }
        return true;
    }

    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (!headA || !headB) return nullptr;
        ListNode *pa = headA, *pb = headB;
        while (pa != pb) {
            pa = pa == nullptr ? headB : pa->next;
            pb = pb == nullptr ? headA : pb->next;
        }
        return pa;
    }

    int maxSubArray(vector<int> &nums) {
        int res = INT_MIN;
        int m = nums.size();
        vector<int> dp(m);
        dp[0] = nums[0];
        for (int i = 1; i < m; ++i) {
            dp[i] = max(nums[i], dp[i - 1] + nums[i]);
            res = max(res, dp[i]);
        }
        return res;
    }

    vector<double> medianSlidingWindow(vector<int> &nums, int k) {
        vector<double> res;
        multiset<double> st;
        for (int i = 0; i < nums.size(); ++i) {
            if (st.size() >= k) st.erase(st.find(nums[i - k]));
            st.insert(nums[i]);
            if (i >= k - 1) {
                auto mid = st.begin();
                std::advance(mid, k / 2);
                res.push_back((*mid + *prev(mid, 1 - k % 2)) / 2);
            }
        }
        return res;
    }

    double findMaxAverage(vector<int> &nums, int k) {
        vector<int> pre(nums.size() + 1);
        pre[0] = 0;
        for (int i = 1; i <= nums.size(); ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        double res = -INFINITY;
        for (int i = 0; i <= nums.size() - k; ++i) {
            res = max((double) (pre[i + k] - pre[i]) / k, res);
        }
        return res;
    }

    TreeNode *mergeTrees(TreeNode *t1, TreeNode *t2) {
        if (t1 == nullptr) {
            return t2;
        }
        if (t2 == nullptr) {
            return t1;
        }
        TreeNode *res = new TreeNode(t1->val + t2->val);
        res->left = mergeTrees(t1->left, t2->left);
        res->right = mergeTrees(t1->right, t2->right);
        return res;
    }

    int hammingDistance(int x, int y) {
        if (x > y) swap(x, y);
        int res = 0;
        while (x != 0) {
            if ((x & 1) != (y & 1)) {
                res++;
            }
            x >>= 1;
            y >>= 1;
        }
        while (y != 0) {
            if ((y & 1) != 0) {
                res++;
            }
            y >>= 1;
        }
        return res;
    }

    TreeNode *invertTree(TreeNode *root) {
        if (root == nullptr) return nullptr;
        swap(root->right, root->left);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }

    int minPathSum(vector<vector<int>> &grid) {
        int dp[201][201]{0};
        dp[0][0] = grid[0][0];
        for (int i = 1; i < grid.size(); ++i) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < grid[0].size(); ++j) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < grid.size(); ++i) {
            for (int j = 1; j < grid[0].size(); ++j) {
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
            }
        }
        return dp[grid.size() - 1][grid[0].size() - 1];
    }

    string maximumTime(string time) {
        if (time[0] == '?') {
            if (time[1] != '?' && time[1] > '3') {
                time[0] = '1';
            } else {
                time[0] = '2';
            }
        }
        if (time[1] == '?') {
            if (time[0] == '2') {
                time[1] = '3';
            } else {
                time[1] = '9';
            }
        }
        if (time[2] == '?') {
            time[2] = '5';
        }
        if (time[3] == '?') {
            time[3] = '9';
        }
        return time;
    }

    int minCharacters(string a, string b) {
        int n = a.size(), m = b.size();
        vector<int> va(26, 0), vb(26, 0);
        for (char c:a) va[c - 'a']++;
        for (char c:b) vb[c - 'a']++;

        int ret = m + n;
        int case1 = ret;
        for (int i = 0; i < 25; ++i) {
            int cur = 0;
            for (int j = i + 1; j < 26; ++j) cur += va[j];
            for (int j = 0; j <= i; ++j) cur += vb[j];
            case1 = min(case1, cur);
        }
        int case2 = ret;
        for (int i = 0; i < 25; ++i) {
            int cur = 0;
            for (int j = i + 1; j < 26; ++j) cur += vb[j];
            for (int j = 0; j <= i; ++j) cur += va[j];
            case2 = min(case2, cur);
        }
        int case3 = ret;
        for (int i = 0; i < 26; ++i) {
            int cur = 0;
            for (int j = 0; j < 26; ++j) {
                if (j == i) continue;
                cur += va[j] + vb[j];
            }
            case3 = min(case3, cur);
        }
        ret = min(case1, min(case2, case3));
        return ret;
    }

    int kthLargestValue(vector<vector<int>> &matrix, int k) {
        vector<vector<int>> dp(matrix.size(), vector<int>(matrix[0].size(), 0));
        dp[0][0] = matrix[0][0];
        priority_queue<int, vector<int>, greater<>> pq;
        pq.push(matrix[0][0]);
        for (int i = 1; i < matrix.size(); ++i) {
            dp[i][0] = dp[i - 1][0] ^ matrix[i][0];
            pq.push(dp[i][0]);
            if (pq.size() > k) {
                pq.pop();
            }
        }
        for (int i = 1; i < matrix[0].size(); ++i) {
            dp[0][i] = dp[0][i - 1] ^ matrix[0][i];
            pq.push(dp[0][i]);
            if (pq.size() > k) {
                pq.pop();
            }
        }
        for (int i = 1; i < matrix.size(); ++i) {
            for (int j = 1; j < matrix[0].size(); ++j) {
                dp[i][j] = dp[i - 1][j] ^ dp[i][j - 1] ^ dp[i - 1][j - 1] ^ matrix[i][j];
                pq.push(matrix[i][j]);
                if (pq.size() > k) {
                    pq.pop();
                }
            }
        }
        return pq.top();
    }

    bool checkPossibility(vector<int> &nums) {
        int n = nums.size(), cnt = 0;
        for (int i = 0; i < n - 1; ++i) {
            int x = nums[i], y = nums[i + 1];
            if (x > y) {
                cnt++;
                if (cnt > 1) {
                    return false;
                }
                if (i > 0 && y < nums[i - 1]) {
                    nums[i + 1] = x;
                }
            }
        }
        return true;
    }

    bool check(vector<int> &nums) {
        if (is_sorted(nums.begin(), nums.end())) {
            return true;
        }
        int n = nums.size(), cnt = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] > nums[i + 1]) {
                cnt++;
                if (cnt > 1) return false;
            }
        }
        return nums[n - 1] <= nums[0];
    }

    int maximumScore(int a, int b, int c) {
        if (a + b <= c) return a + b;
        if (a + c <= b) return a + c;
        if (b + c <= a) return b + c;
        return (a + b + c) / 2;
    }

    ListNode *reverseKGroup(ListNode *head, int k) {
        ListNode *thead = head;
        ListNode *res = new ListNode(0), *tmp = res, *tmp2 = res;
        int cnt = 0;
        while (thead != nullptr) {
            cnt++;
            thead = thead->next;
        }
        int n = cnt / k;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                ListNode *t = head->next;
                head->next = tmp2->next;
                tmp2->next = head;
                head = t;
            }
            while (tmp->next != nullptr) {
                tmp = tmp->next;
            }
            tmp2 = tmp;
        }
        tmp2->next = head;
        return res->next;
    }

    int maxTurbulenceSize(vector<int> &A) {
        if (A.size() <= 1) return A.size();
        int res = 0;
        int l = 0, r = 1;
        while (r < A.size()) {
            if (r - l <= 1) {
                if (A[l] != A[r]) {
                    res = max(r - l + 1, res);
                    r++;
                    continue;
                } else {
                    l++;
                    r++;
                }
            } else {
                if ((A[r - 1] < A[r - 2] && A[r - 1] < A[r]) || (A[r - 1] > A[r - 2] && A[r - 1] > A[r])) {
                    res = max(r - l + 1, res);
                    r++;
                } else if (A[r] == A[r - 1]) {
                    l = r;
                    r++;
                } else {
                    l = r - 1;
                    r++;
                }
            }
        }
        res = max(r - l, res);
        return res;
    }

    int sumOfUnique(vector<int> &nums) {
        vector<int> v(101);
        for (int i:nums) v[i]++;
        int sum = 0;
        for (int i = 0; i < v.size(); ++i) {
            if (v[i] == 1) {
                sum += i;
            }
        }
        return sum;
    }

    int maxAbsoluteSum(vector<int> &nums) {
        vector<int> pre(nums.size() + 1, 0);
        for (int i = 1; i < pre.size(); ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        sort(pre.begin(), pre.end());
        return pre[nums.size()] - pre[0];
    }

    int minimumLength(string s) {
        int l = 0, r = s.size() - 1;
        while (s[l] == s[r] && l < r) {
            while (s[l] == s[l + 1] && l < r) l++;
            while (s[r] == s[r - 1] && l < r) r--;
            l++;
            r--;
            if (l >= r) break;
        }
        return max(r - l + 1, 0);
    }

    int largestAltitude(vector<int> &gain) {
        vector<int> v(gain.size() + 1, 0);
        int res = 0;
        for (int i = 1; i < v.size(); ++i) {
            v[i] = v[i - 1] + gain[i - 1];
            res = max(res, v[i]);
        }
        return res;
    }

    bool checkInclusion(string s1, string s2) {
        int n = s1.size(), m = s2.size();
        if (n > m) return false;
        vector<int> cnt1(26), cnt2(26);
        for (int i = 0; i < n; ++i) {
            ++cnt1[s1[i] - 'a'];
            ++cnt2[s2[i] - 'a'];
        }
        if (cnt1 == cnt2) return true;
        for (int i = n; i < m; ++i) {
            ++cnt2[s2[i] - 'a'];
            --cnt2[s2[i - n] - 'a'];
            if (cnt1 == cnt2) return true;
        }
        return false;
    }


    int GetMostDistinct(vector<int> &A, int K) {
        unordered_map<int, int> mp;
        int left = 0, right = 0, ret = 0;
        while (right < A.size()) {
            ++mp[A[right++]];
            while (mp.size() > K) {
                --mp[A[left]];
                if (mp[A[left]] == 0) mp.erase(A[left]);
                ++left;
            }
            // 如果这里改成 ret = max(ret, right - left)，那么此函数就是 LeetCode 904 题的解：求长度最大的子数组（此子数组中包含不同整数个数最多为K）
            ret += right - left;
        }
        return ret;
    }

    int subarraysWithKDistinct(vector<int> &A, int K) {
        return GetMostDistinct(A, K) - GetMostDistinct(A, K - 1);
    }

    int characterReplacement(string s, int k) {
        vector<int> v(26, 0);
        int tmp = 0;
        int l = 0, r = 0;
        while (r < s.size()) {
            v[s[r] - 'A']++;
            tmp = max(tmp, v[s[r] - 'A']);
            if (r - l + 1 > tmp + k) {
                v[s[l] - 'A']--;
                l++;
            }
        }
        return s.size() - l;
    }

    int numSubarrayProductLessThanK(vector<int> &nums, int k) {
        if (k <= 1) return 0;
        int p = 1, res = 0, l = 0;
        for (int r = 0; r < nums.size(); ++r) {
            p *= nums[r];
            while (p >= k) p /= nums[l++];
            res += r - l + 1;
        }
        return res;
    }

    ListNode *removeNthFromEnd(ListNode *head, int n) {
        ListNode *t = new ListNode(0);
        t->next = head;
        ListNode *f = t, *s = t;
        for (int i = 0; i < n; ++i) {
            f = f->next;
        }
        while (f->next != nullptr) {
            f = f->next;
            s = s->next;
        }
        ListNode *del = s->next;
        s->next = del->next;
        return t->next;
    }

    int totalFruit(vector<int> &tree) {
        int l = 0, r = 0;
        int res = 0;
        unordered_map<int, int> mp;
        while (r < tree.size()) {
            if (tree[r] == 0) {
                l = r + 1;
                r++;
                mp.clear();
                continue;
            }
            if (r - l + 1 <= 2) {
                mp[tree[r]]++;
                r++;
                continue;
            }
            mp[tree[r]]++;
            if (mp.size() > 2) {
                res = max(res, r - l);
                while (mp.size() > 2) {
                    mp[tree[l]]--;
                    if (mp[tree[l]] == 0) {
                        mp.erase(tree[l]);
                    }
                    l++;
                }
            }
            r++;
        }
        res = max(res, r - l);
        return res;
    }

    int numRescueBoats(vector<int> &people, int limit) {
        sort(people.begin(), people.end());
        int i = 0, j = people.size() - 1;
        int res = 0;
        while (i <= j) {
            res++;
            if (people[i] + people[j] <= limit) {
                i++;
            }
            j--;
        }
        return res;
    }

    int findPairs(vector<int> &nums, int k) {
        if (nums.size() < 2) return 0;
        sort(nums.begin(), nums.end());
        int count = 0;

    }

    int minSwapsCouplesFind(vector<int> &p, int x) {
        if (p[x] == x) return x;
        return p[x] = minSwapsCouplesFind(p, p[x]);
    }

    void minSwapsCouplesUnion(vector<int> &p, int x, int y) {
        int px = minSwapsCouplesFind(p, x);
        int py = minSwapsCouplesFind(p, y);
        p[px] = py;
    }

    int minSwapsCouples(vector<int> &row) {
        int n = row.size();
        vector<int> p(n);
        int c = 0;
        for (int i = 0; i < n; ++i) p[i] = i;
        for (int i = 0; i < n - 1; i += 2) {
            int n1 = row[i] / 2, n2 = row[i + 1] / 2;
            if (n1 != n2) {
                if (minSwapsCouplesFind(p, n1) == minSwapsCouplesFind(p, n2)) {
                    c++;
                } else {
                    minSwapsCouplesUnion(p, n1, n2);
                }
            } else {
                c++;
            }
        }
        return n / 2 - c;
    }

    bool canPlaceFlowers(vector<int> &flowerbed, int n) {
        if (n == 0) return true;
        for (int i = 0; i < flowerbed.size(); ++i) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) &&
                (i == flowerbed.size() - 1 || flowerbed[i + 1] == 0)) {
                n--;
                if (n <= 0) return true;
                flowerbed[i] = 1;
            }
        }
        return false;
    }

    int minCost(string s, vector<int> &cost) {
        int res = 0;
        int i = 0;
        while (i < s.size()) {
            char cur = s[i];
            int m = 0, sum = 0;
            while (i < s.size() && cur == s[i]) {
                m = max(m, cost[i]);
                sum += cost[i];
                ++i;
            }
            res += (sum - m);
        }
        return res;
    }

    string getSmallestString(int n, int k) {
        string res = "";
        for (int i = n; i >= 1; --i) {
            int t = k - 26 * (i - 1);
            if (t > 0) {
                res += char(t + 'a' - 1);
                k -= t;
            } else {
                res += 'a';
                k -= 1;
            }
        }
        return res;
    }

    int findMaxConsecutiveOnes(vector<int> &nums) {
        int res = 0, l = 0;
        while (l < nums.size()) {
            if (nums[l] == 1) {
                int cur = 0;
                while (l < nums.size() && nums[l] == 1) {
                    l++;
                    cur++;
                }
                res = max(res, cur);
            } else {
                l++;
            }
        }
        return res;
    }

    int longestOnes(vector<int> &A, int K) {
        int l = 0, r = 0, count = 0, res = 0;
        while (r < A.size()) {
            count += A[r] == 0;
            while (count > K) {
                count -= A[l] == 0;
                l++;
            }
            res = max(res, r - l + 1);
            r++;
        }
        return res;
    }

    TreeNode *first = nullptr;

    TreeNode *convertBiNode(TreeNode *root) {
        TreeNode *res = nullptr;
        convertBiNodeHelp(root, res);
        return first;
    }

    void convertBiNodeHelp(TreeNode *root, TreeNode *pre) {
        if (root == nullptr) return;
        convertBiNodeHelp(root->left, pre);
        if (pre == nullptr) {
            pre = root;
            first = root;
        } else {
            pre->right = root;
            root->left = nullptr;
            pre = root;
        }
        convertBiNodeHelp(root->right, pre);
    }

    TreeNode *balanceBST(TreeNode *root) {
        vector<int> vec;
        balanceBSTHelp(root, vec);
        return balanceBSTBuild(vec, 0, vec.size() - 1);
    }

    TreeNode *balanceBSTBuild(vector<int> &vec, int l, int r) {
        int mid = (l + r) >> 1;
        TreeNode *res = new TreeNode(vec[mid]);
        if (l <= mid - 1) res->left = balanceBSTBuild(vec, l, mid - 1);
        if (r >= mid + 1) res->right = balanceBSTBuild(vec, mid + 1, r);
        return res;
    }

    void balanceBSTHelp(TreeNode *root, vector<int> &vec) {
        if (root == nullptr) return;
        balanceBSTHelp(root->left, vec);
        vec.emplace_back(root->val);
        balanceBSTHelp(root->right, vec);
    }

    TreeNode *bstToGst(TreeNode *root) {
        int cur = 0;
        bstToGstHelp(root, cur);
        return root;
    }

    void bstToGstHelp(TreeNode *root, int &cur) {
        if (root == nullptr) return;
        bstToGstHelp(root->right, cur);
        cur += root->val;
        root->val = cur;
        bstToGstHelp(root->left, cur);
    }

    int arrayPairSum(vector<int> &nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < nums.size(); i += 2) {
            res += nums[i];
        };
        return res;
    }

    int removeCoveredIntervals(vector<vector<int>> &intervals) {
        sort(intervals.begin(), intervals.end(), [](const auto &a, const auto &b) {
            return a[0] == b[0] ? a[1] > b[1] : a[0] < b[0];
        });
        int r = 0;
        int res = 0;
        while (r < intervals.size() - 1) {
            int i = 1;
            while (r + i < intervals.size() && intervals[r][1] >= intervals[r + i][1]) {
                res++;
                i++;
            }
            r += i;
        }
        return intervals.size() - res;
    }

    vector<int> pancakeSort(vector<int> &arr) {
        int n = arr.size();
        vector<int> res;
        for (int i = n - 1; i > 0; --i) {
            int j = max_element(arr.begin(), arr.begin() + i + 1) - arr.begin();
            if (j > 0) {
                reverse(arr.begin(), arr.begin() + j + 1);
                res.push_back(j + 1);
            }
            reverse(arr.begin(), arr.begin() + i + 1);
            res.push_back(i + 1);
        }
        return res;
    }

    vector<vector<int>> levelOrder(TreeNode *root) {
        vector<vector<int>> res;
        if (root == nullptr) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int n = q.size();
            vector<int> tmp;
            for (int i = 0; i < n; ++i) {
                auto cur = q.front();
                q.pop();
                tmp.emplace_back(cur->val);
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res.emplace_back(tmp);
        }
        return res;
    }
};

class KthLargest {
public:
    priority_queue<int, vector<int>, greater<>> pq;
    int k;

    KthLargest(int k, vector<int> &nums) {
        pq = priority_queue<int, vector<int>, greater<>>();
        this->k = k;

        for (int n:nums) {
            pq.push(n);
            if (pq.size() > k) {
                pq.pop();
            }
        }
    }

    int add(int val) {
        pq.push(val);
        if (pq.size() > k) {
            pq.pop();
        }
        return pq.top();
    }
};

class NumArray {
public:
    vector<int> tree;
    int n;

    NumArray(vector<int> &nums) {
        if (nums.size() > 0) {
            n = nums.size();
            tree = vector<int>(n * 2);
            buildTree(nums);
        }
    }

    void buildTree(vector<int> &nums) {
        for (int i = n, j = 0; i < 2 * n; ++i, ++j) {
            tree[i] = nums[j];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }

    void update(int index, int val) {
        index += n;
        tree[index] = val;
        while (index > 0) {
            int left = index, right = index;
            if (index % 2 == 0) {
                right = index + 1;
            } else {
                left = index - 1;
            }
            tree[index / 2] = tree[left] + tree[right];
            index /= 2;
        }
    }

    int sumRange(int left, int right) {
        left += n;
        right += n;
        int sum = 0;
        while (left <= right) {
            if ((left % 2) == 1) {
                sum += tree[left];
                left++;
            }
            if ((right % 2) == 0) {
                sum += tree[right];
                right--;
            }
            left /= 2;
            right /= 2;
        }
        return sum;
    }
};

#endif //LEETCODEMAC_SOLUTION3_H
