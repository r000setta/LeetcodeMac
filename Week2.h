#ifndef LEETCODEMAC_WEEK2_H
#define LEETCODEMAC_WEEK2_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class WeekSolution2 {
public:
    int minimumEffortPathDir[4][2] = {{0,  1},
                                      {0,  -1},
                                      {-1, 0},
                                      {1,  0}};

    int minimumEffortPath(vector<vector<int>> &heights) {
        int l = 0, r = 1e6;
        int mid, ans;
        while (l <= r) {
            mid = l + ((r - 1) >> 1);
            vector<vector<bool>> vis(heights.size(), vector<bool>(heights[0].size()));
            if (minimumEffortPathHelp(heights, vis, 0, 0, mid)) {
                r = mid - 1;
                ans = mid;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    bool minimumEffortPathHelp(vector<vector<int>> &heights, vector<vector<bool>> &vis, int x, int y, int val) {
        if (x == heights.size() - 1 && y == heights[0].size() - 1) return true;
        vis[x][y] = true;
        for (const auto &dir:minimumEffortPathDir) {
            int tx = x + dir[0];
            int ty = y + dir[1];
            if (tx >= 0 && tx < heights.size() && ty >= 0 && ty < heights[0].size() && !vis[tx][ty] &&
                abs(heights[x][y] - heights[tx][ty]) <= val) {
                if (minimumEffortPathHelp(heights, vis, tx, ty, val)) {
                    return true;
                }
            }
        }
        return false;
    }

    string sortString(string s) {
        vector<int> cnt(26);
        for (char c:s) {
            cnt[c - 'a']++;
        }
        string res;
        while (res.size() < s.size()) {
            for (int i = 0; i < cnt.size(); ++i) {
                if (cnt[i] > 0) {
                    res += i + 'a';
                    cnt[i]--;
                }
            }
            for (int i = cnt.size() - 1; i >= 0; --i) {
                if (cnt[i] > 0) {
                    res += i + 'a';
                    cnt[i]--;
                }
            }
        }
        return res;
    }

    vector<int> mostVisited(int n, vector<int> &rounds) {
        vector<int> res;
        int start = rounds[0], end = rounds.back();
        if (start <= end) {
            for (int i = start; i <= end; ++i) res.push_back(i);
            return res;
        } else {
            for (int i = 1; i <= end; ++i) res.push_back(i);
            for (int i = start; i <= n; ++i) res.push_back(i);
            return res;
        }
    }

    int maxCoins(vector<int> &piles) {
        sort(piles.begin(), piles.end());
        int l = 0, r = piles.size() - 1;
        int res = 0;
        while (r - l >= 2) {
            res += piles[r - 1];
            l++;
            r -= 2;
        }
        return res;
    }

    bool containsPattern(vector<int> &arr, int m, int k) {
        int n = arr.size();
        for (int l = 0; l <= n - m * k; ++l) {
            int offset;
            for (offset = 0; offset < m * k; ++offset) {
                if (arr[l + offset] != arr[l + offset % m]) {
                    break;
                }
            }
            if (offset == m * k) return true;
        }
        return false;
    }

    int getMaxLen(vector<int> &nums) {
        vector<vector<int>> dp(nums.size() + 1, vector<int>(2));
        dp[0][0] = dp[0][1] = 0;
        int ans = 0;
        for (int i = 1; i <= nums.size(); ++i) {
            if (nums[i - 1] == 0) {
                dp[i][0] = dp[i][1] = 0;
            } else if (nums[i - 1] > 0) {
                dp[i][0] = dp[i - 1][0] + 1;
                dp[i][1] = dp[i - 1][1] ? (dp[i - 1][1] + 1) : 0;
            } else {
                dp[i][0] = dp[i - 1][1] ? (dp[i - 1][1] + 1) : 0;
                dp[i][1] = dp[i - 1][0] + 1;
            }
            ans = max(ans, dp[i][0]);
        }
        return ans;
    }

    int minDays(vector<vector<int>> &grid) {
        if (check(grid)) return 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] == 0) continue;
                grid[i][j] = 0;
                if (check(grid)) return 1;
                grid[i][j] = 1;
            }
        }
        return 2;
    }

    bool check(const vector<vector<int>> &grid) {
        int x = 0, y = 0;
        int cnt = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[i].size(); ++j) {
                if (grid[i][j] == 0) continue;
                cnt++;
                x = i;
                y = j;
            }
        }
    }

    int maximumGap(vector<int> &nums) {
        if (nums.size() < 2) return 0;
        sort(nums.begin(), nums.end());
        int res = -1;
        for (int i = 1; i < nums.size(); ++i) {
            res = max(nums[i] - nums[i - 1], res);
        }
        return res;
    }

    string thousandSeparator(int n) {
        string res;
        int cnt = 0;
        while (n) {
            if (!cnt % 3 && cnt != 0) res += '.';
            char t = n % 10 + '0';
            res += t;
            n /= 10;
            cnt++;
        }
        reverse(res.begin(), res.end());
        return res;
    }

    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>> &edges) {
        vector<int> cnt(n);
        vector<int> res;
        for (const auto &e:edges) {
            cnt[e[1]]++;
        }
        for (int i = 0; i < n; ++i) {
            if (!cnt[i]) res.push_back(i);
        }
        return res;
    }

    int minOperations(vector<int> &nums) {
        int res = 0;
        while (!minOperationsCheck(nums)) {
            for (auto &num:nums) {
                if (num % 2) {
                    res++;
                }
                num /= 2;
            }
            res++;
        }
        for (auto &n:nums) {
            if (n == 1) res++;
        }
        return res;
    }

    bool minOperationsCheck(vector<int> &nums) {
        for (auto &i:nums) {
            if (i != 0 && i != 1) return false;
        }
        return true;
    }

    int findKthPositive(vector<int> &arr, int k) {
        int cnt = 0;
        int i = 0, j = 0;
        while (cnt < k) {
            if (j == arr.size()) {
                cnt++;
                i++;
                continue;
            }
            if (i != arr[j]) {
                cnt++;
                i++;
            } else {
                i++;
                j++;
            }
        }
        return i - 1;
    }

    bool canConvertString(string s, string t, int k) {
        if (s.size() != t.size()) return false;
        vector<int> cnt(26);
        int n = s.size();
        for (int i = 0; i < n; ++i) {
            int diff = t[i] - s[i];
            if (diff < 0) diff += 26;
            cnt[diff]++;
        }
        for (int i = 0; i < 26; ++i) {
            int maxval = i + 26 * (cnt[i] - 1);
            if (maxval > k) {
                return false;
            }
        }
        return true;
    }

    int minInsertions(string s) {
        int i = 0, j = 0, res = 0;
        for (char c:s) {
            if (c == '(') {
                i += 2;
            } else {
                if (i != 0) {
                    i--;
                } else {
                    j++;
                }
            }
        }
        if (!(j % 2)) {
            res += j / 2;
        } else {
            res += j / 2;
            res += 2;
        }
        res += i;
        return res;
    }

    vector<int> searchRange(vector<int> &nums, int target) {
        vector<int> res{-1, -1};
        if (nums.size() == 1) {
            if (nums[0] == target) {
                res[0] = res[1] = 0;
            }
            return res;
        }
        int l = searchRangeLeft(nums, target);
        int r = searchRangeRight(nums, target);
        return {l, r};
    }

    int searchRangeLeft(vector<int> &nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] == target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left >= nums.size() || nums[left] != target) return -1;
        return left;
    }

    int searchRangeRight(vector<int> &nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] == target) {
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0 || nums[right] != target) return -1;
        return right;
    }

    int maximumWealth(vector<vector<int>> &accounts) {
        int res = -1;
        for (int i = 0; i < accounts.size(); ++i) {
            int sum = 0;
            for (int j = 0; j < accounts[0].size(); ++j) {
                sum += accounts[i][j];
            }
            res = max(res, sum);
        }
        return res;
    }

    vector<int> mostCompetitive(vector<int> &nums, int k) {
        stack<int> stk;
        stk.push(-1);
        int m = nums.size();
        for (int i = 0; i < m; ++i) {
            while (nums[i] < stk.top() && k - stk.size() + 1 < m - i) {
                stk.pop();
            }
            if (stk.size() < k + 1) {
                stk.push(nums[i]);
            }
        }
        vector<int> res(k);
        for (int i = k - 1; i >= 0; --i) {
            res[i] = stk.top();
            stk.pop();
        }
        return res;
    }

    void mostCompetitiveHelp(vector<int> &nums, vector<int> &res, int k, int l) {
        if (k == 0) return;
        int m = nums.size();
        for (int i = l; i < m - 1; ++i) {
            if (m - i + 1 == k) {
                for (int j = i; j < m - 1; j++) {
                    res.push_back(nums[j]);
                    mostCompetitiveHelp(nums, res, 0, 0);
                }
            }
            if (nums[i] < nums[i + 1]) {
                res.push_back(nums[i]);
                mostCompetitiveHelp(nums, res, k--, i + 1);
            }
        }
    }

    int numIdenticalPairs(vector<int> &nums) {
        vector<int> vec(101);
        for (int i:nums) {
            vec[i]++;
        }
        int res = 0;
        for (int i:vec) {
            if (i >= 2) {
                int t = (i * (i - 1)) / 2;
                res += t;
            }
        }
        return res;
    }

    int numSub(string s) {
        long long res = 0, t = 0;
        int mod = 1e9 + 7;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '1') {
                t++;
            } else {
                if (t != 0) {
                    unsigned long long sum = ((t + 1) * t) >> 1;
                    t = 0;
                    res += sum % mod;
                }
            }
        }
        if (t != 0) {
            unsigned long long sum = ((t + 1) * t) >> 1;
            t = 0;
            res += sum % mod;
        }
        return res % mod;
    }

//    double maxProbability(int n, vector<vector<int>> &edges, vector<double> &succProb, int start, int end) {
//        vector<vector<pair<double, int>>> graph;
//        for (int i = 0; i < edges.size(); ++i) {
//            auto &e = edges[i];
//            graph[e[0]].emplace_back(succProb[i], e[1]);
//            graph[e[1]].emplace_back(succProb[i], e[0]);
//        }
//
//        priority_queue<pair<double, int>> que;
//        vector<double> prob(n, 0);
//        que.emplace(1, start);
//        prob[start] = 1;
//        while (!que.empty()) {
//            auto[pr, node]=que.top();
//            que.pop();
//            if (pr < prob[node]) {
//                continue;
//            }
//            for (auto&[prNext, nodeNext]:graph[node]) {
//
//            }
//        }
//    }

    string reformatDate(string date) {
        unordered_map<string, string> mp = {
                {"Jan", "01"},
                {"Feb", "02"},
                {"Mar", "03"},
                {"Apr", "04"},
                {"May", "05"},
                {"Jun", "06"},
                {"Jul", "07"},
                {"Aug", "08"},
                {"Sep", "09"},
                {"Oct", "10"},
                {"Nov", "11"},
                {"Dec", "12"}
        };
        stringstream ss(date);
        string year, month, day;
        ss >> day >> month >> year;
        month = mp[month];
        day.pop_back();
        day.pop_back();
        if (day.size() == 1) day = "0" + day;
        return year + "-" + month + "-" + day;
    }

    int rangeSum(vector<int> &nums, int n, int left, int right) {
        int len = nums.size();
        vector<int> pre(len + 1);
        pre[1] = nums[0];
        for (int i = 2; i < pre.size(); ++i) {
            pre[i] = pre[i - 1] + nums[i - 1];
        }
        unsigned int len2 = ((1 + len) * len) >> 1;
        vector<unsigned int> vec(len2);
        for (int i = 0, x = 0; x < len; ++x) {
            for (int y = x; y < len; ++y, ++i) {
                vec[i] = pre[y + 1] - pre[x];
            }
        }
        sort(vec.begin(), vec.end());
        unsigned long res = 0;
        unsigned long mod = 1e9 + 7;
        for (int i = left - 1; i <= right - 1; ++i) {
            res = (res + vec[i]) % mod;
        }
        return res;
    }

    int minDifference(vector<int> &nums) {
        if (nums.size() <= 4) return 0;
        sort(nums.begin(), nums.end());
        int res = INT_MAX;
        for (int i = 0; i < 4; ++i) {
            res = min(res, nums[nums.size() - 4 + i] - nums[i]);
        }
        return res;
    }

    bool canMakeArithmeticProgression(vector<int> &arr) {
        if (arr.size() <= 2) return true;
        sort(arr.begin(), arr.end());
        int tar = arr[1] - arr[0];
        for (int i = 1; i < arr.size() - 1; ++i) {
            if (arr[i + 1] - arr[i] != tar) {
                return false;
            }
        }
        return true;
    }

    int numSubmat(vector<vector<int>> &mat) {
        int m = mat.size(), n = mat[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == 0) dp[i][j] = mat[i][j];
                else
                    dp[i][j] = mat[i][j] == 1 ?
                               dp[i][j - 1] + 1 : 0;
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int t = dp[i][j];
                for (int k = i; k >= 0 && t; k--) {
                    t = min(t, dp[k][j]);
                    res += t;
                }
            }
        }
        return res;
    }

    bool canBeEqual(vector<int> &target, vector<int> &arr) {
        if (target.size() != arr.size()) return false;
        sort(target.begin(), target.end());
        sort(arr.begin(), arr.end());
        for (int i = 0; i < arr.size(); ++i) {
            if (target[i] != arr[i]) return false;
        }
        return true;
    }

    bool hasAllCodes(string s, int k) {
        if (s.size() < k) return false;
        unordered_set<string> st;
        for (int i = 0; i <= s.size() - k; ++i) {
            st.insert(s.substr(i, k));
        }
        return st.size() == (1 << k);
    }

    vector<bool> checkIfPrerequisite(int n, vector<vector<int>> &prerequisites, vector<vector<int>> &queries) {
        vector<vector<bool>> d(n, vector<bool>(n));
        for (auto &r:prerequisites) d[r[0]][r[1]] = true;
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    d[i][j] = d[i][j] || (d[i][k] && d[k][j]);
                }
            }
        }
        vector<bool> res;
        for (auto &q:queries) {
            res.emplace_back(d[q[0]][q[1]]);
        }
        return res;
    }

    bool isPossible(vector<int> &nums) {
        unordered_map<int, int> nc, tail;
        for (auto num:nums) nc[num]++;
        for (auto num:nums) {
            if (nc[num] == 0) continue;
            else if (nc[num] > 0 && tail[num - 1] > 0) {
                nc[num]--;
                tail[num - 1]--;
                tail[num]++;
            } else if (nc[num] > 0 && nc[num + 1] > 0 && nc[num + 2] > 0) {
                nc[num]--;
                nc[num + 1]--;
                nc[num + 2]--;
            } else {
                return false;
            }
        }
        return true;
    }

    vector<string> stringMatching(vector<string> &words) {
        sort(words.begin(), words.end(), [](const auto &w1, const auto &w2) {
            return w1.size() == w2.size() ? w1 < w2 : w1.size() < w2.size();
        });
        vector<string> res;
        for (int i = 0; i < words.size(); ++i) {
            for (int j = i; j < words.size(); ++j) {
                if (i != j && words[j].find(words[i]) != words[j].npos) {
                    res.emplace_back(words[i]);
                    break;
                }
            }
        }
        return res;
    }

    vector<int> processQueries(vector<int> &queries, int m) {
        vector<int> p(m);
        iota(p.begin(), p.end(), 1);
        vector<int> ans(queries.size());
        for (int i = 0; i < queries.size(); ++i) {
            for (int j = 0; j < m; ++j) {
                if (p[j] == queries[i]) {
                    ans[i] = j;
                    p.erase(p.begin() + j);
                    p.insert(p.begin(), queries[i]);
                    break;
                }
            }
        }
        return ans;
    }

    string entityParser(string text) {
        map<string, string> pool = {
                {"&quot;",  "\""},
                {"&apos;",  "'"},
                {"&amp;",   "&"},
                {"&gt;",    ">"},
                {"&lt;",    "<"},
                {"&frasl;", "/"}
        };
        string key;
        string res;
        for (auto c:text) {
            if (c == '&') {
                if (!key.empty()) {
                    res += key;
                    key.erase();
                }
                key.push_back(c);
            } else if (c != ';') {
                key.push_back(c);
            } else {
                key.push_back(c);
                if (pool.find(key) != pool.end()) {
                    res += pool[key];
                    key.erase();
                } else {
                    res += key;
                    key.erase();
                }
            }
        }
        if (!key.empty()) {
            res += key;
        }
        return res;
    }

    vector<int> minSubsequence(vector<int> &nums) {
        sort(nums.begin(), nums.end(), greater<int>());
        int sum = 0;
        for (int n:nums) sum += n;
        int tar = 0;
        vector<int> res;
        for (int i:nums) {
            res.emplace_back(i);
            tar += i;
            if (tar > sum / 2) {
                break;
            }
        }
        return res;
    }

    string interpret(string command) {
        string res;
        string key;
        for (auto c:command) {
            if (c == 'G') {
                res += 'G';
            } else {
                key += c;
                if (key == "()") {
                    key = "";
                    res += 'o';
                } else if (key == "(al)") {
                    key = "";
                    res += "al";
                }
            }
        }
        if (key != "") {
            res += key;
        }
        return res;
    }

    int maxOperations(vector<int> &nums, int k) {
        sort(nums.begin(), nums.end());
        int res = 0;
        int l = 0, r = nums.size() - 1;
        while (l <= r) {
            if (nums[l] + nums[r] == k) {
                l++;
                r--;
                res++;
            } else if (nums[l] + nums[r] > k) {
                r--;
            } else {
                l++;
            }
        }
        return res;
    }

    int concatenatedBinary(int n) {
        unsigned long long res = 0;
        unsigned long long mod = 1e9 + 7;
        vector<string> v(n);
        if (n == 1) return 1;
        v[0] = "1";
        for (int i = 1; i < n; ++i) {
            if ((i & 1) == 0) {
                v[i] = v[i / 2 - 1] + '1';
            } else {
                v[i] = v[i / 2] + '0';
            }
        }
        int k = 0, f = 0;
        for (int i = v.size() - 1; i >= 0; --i) {
            for (int j = v[i].size() - 1; j >= 0; --j) {
                unsigned long long tmp;
                if (f == 0) {
                    tmp = v[i][j] - '0';
                    f++;
                } else {
                    unsigned long long t = (2 << (k++)) % mod;
                    tmp = (v[i][j] - '0') * t;
                }
                res += tmp;
            }
        }
        return res;
    }

    vector<int> smallerNumbersThanCurrent(vector<int> &nums) {
        vector<int> cnt(101), sum(101);
        for (int n:nums) {
            cnt[n]++;
        }
        sum[0] = cnt[0];
        for (int i = 1; i < sum.size(); ++i) {
            sum[i] = sum[i - 1] + cnt[i];
        }
        vector<int> res(nums.size());
        for (int i = 0; i < res.size(); ++i) {
            res[i] = sum[nums[i]] - cnt[nums[i]];
        }
        return res;
    }
};

#endif //LEETCODEMAC_WEEK2_H
