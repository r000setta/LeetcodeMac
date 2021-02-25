#ifndef LEETCODEMAC_SOLUTION5_H
#define LEETCODEMAC_SOLUTION5_H

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

class Solution5 {
public:
    int strStr(string haystack, string needle) {

    }

    vector<vector<int>> matrixReshape(vector<vector<int>> &nums, int r, int c) {
        int m = nums.size(), n = nums[0].size();
        if (m * n != r * c) return nums;
        vector<vector<int>> res(r, vector<int>(c));
        for (int i = 0; i < m * n; ++i) {
            res[i / c][i % c] = nums[i / n][i % n];
        }
        return res;
    }

    int findShortestSubArray(vector<int> &nums) {
        unordered_map<int, vector<int>> mp;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (mp.count(nums[i])) {
                mp[nums[i]][0]++;
                mp[nums[i]][2] = i;
            } else {
                mp[nums[i]] = {1, i, i};
            }
        }
        int maxNum = 0, minLen = 0;
        for (auto &m:mp) {
            if (maxNum < m.second[0]) {
                maxNum = m.second[0];
                minLen = m.second[2] - m.second[1] + 1;
            } else if (maxNum == m.second[0]) {
                minLen = min(minLen, m.second[2] - m.second[1] + 1);
            }
        }
        return minLen;
    }

    int minOperations(string s) {
        int n = s.size(), cnt1 = 0, cnt2 = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] % 2 != i % 2) cnt1++;
            else cnt2++;
        }
        return min(cnt1, cnt2);
    }

    int countHomogenous(string s) {
        size_t res = 0, r = 0;
        while (r < s.size()) {
            char cur = s[r];
            size_t t = 1;
            while (cur == s[r] && r < s.size()) {
                res += t;
                r++;
                t++;
            }
        }
        return res % 1000000007;
    }

    int minimumSize(vector<int> &nums, int maxOperations) {
        int l = 1, r = *max_element(nums.begin(), nums.end());
        int ans = 0;
        while (l <= r) {
            int y = (l + r) / 2;
            long long ops = 0;
            for (int x:nums) {
                ops += (x - 1) / y;
            }
            if (ops <= maxOperations) {
                ans = y;
                r = y - 1;
            } else {
                l = y + 1;
            }
        }
        return ans;
    }

    int minTrioDegree(int n, vector<vector<int>> &edges) {
        vector<vector<bool>> d(n, vector<bool>(n));
        vector<int> deg(n);
        for (auto &e:edges) {
            d[e[0] - 1][e[1] - 1] = d[e[1] - 1][e[0] - 1] = true;
            deg[e[0] - 1]++;
            deg[e[1] - 1]++;
        }
        int ans = INT_MAX;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (!d[i][j])
                    continue;
                for (int k = j + 1; k < n; ++k) {
                    if (d[i][k] && d[j][k]) {
                        ans = min(ans, deg[i] + deg[j] + deg[k] - 6);
                    }
                }
            }
        }
        return ans == INT_MAX ? -1 : ans;
    }

    int longestWPI(vector<int> &hours) {

    }

    bool isValid(string s) {
        stack<char> stk;
        for (char c:s) {
            if (c == 'a' || c == 'b') {
                stk.push(c);
            } else {
                if (stk.size() < 2) return false;
                if (stk.top() != 'b') return false;
                stk.pop();
                if (stk.top() != 'a') return false;
                stk.pop();
            }
        }
        return stk.empty();
    }

    bool find132pattern(vector<int> &nums) {

    }

    vector<int> nextGreaterElement(vector<int> &nums1, vector<int> &nums2) {
        unordered_map<int, int> mp;
        stack<int> stk;
        vector<int> res;
        for (auto &n:nums2) {
            if (stk.empty()) stk.push(n);
            else if (n < stk.top()) stk.push(n);
            else {
                while (!stk.empty() && stk.top() < n) {
                    int t = stk.top();
                    stk.pop();
                    mp[t] = n;
                }
                stk.push(n);
            }
        }
        while (!stk.empty()) {
            mp[stk.top()] = -1;
            stk.pop();
        }
        for (int i:nums1) {
            res.emplace_back(mp[i]);
        }
        return res;
    }

    vector<int> dailyTemperatures(vector<int> &T) {
        vector<int> res(T.size());
        stack<int> stk;
        for (int i = 0; i < T.size(); ++i) {
            while (!stk.empty() && T[stk.top()] < T[i]) {
                int idx = stk.top();
                res[idx] = i - idx;
                stk.pop();
            }
            stk.push(i);
        }
        return res;
    }

    vector<int> nextGreaterElements(vector<int> &nums) {
        vector<int> res(nums.size());
        stack<int> stk;
        for (int i = 2 * nums.size() - 1; i >= 0; --i) {
            while (!stk.empty() && nums[stk.top()] <= nums[i % nums.size()]) {
                stk.pop();
            }

        }
    }


    int longestSubarray(vector<int> &nums, int limit) {
        multiset<int> st;
        int l = 0, r = 0, len = 0;
        while (r < nums.size()) {
            st.insert(nums[r]);
            while (*st.rbegin() - *st.begin() > limit) {
                st.erase(st.find(nums[l]));
                l++;
            }
            len = max(len, r - l + 1);
            r++;
        }
        return len;
    }

    string reverseParentheses(string s) {
        stack<string> stk;
        string word = "";
        for (char c:s) {
            if (c == '(') {
                stk.push(word);
                word = "";
            } else if (c == ')') {
                reverse(word.begin(), word.end());
                word = stk.top() + word;
                stk.pop();
            } else {
                word += c;
            }
        }
        return word;
    }

    int calculate(string s) {

    }

    bool isToeplitzMatrix(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) {
                    return false;
                }
            }
        }
        return true;
    }

    string mergeAlternately(string word1, string word2) {
        string res = "";
        int ia = 0, ib = 0;
        string m = word1.size() < word2.size() ? word1 : word2;
        string t = word1.size() < word2.size() ? word2 : word1;
        for (int i = 0; i < 2 * m.size(); ++i) {
            if (i % 2) res += word1[ia];
            else res += word2[ib];
        }
        int len = word1.size() < word2.size() ? word2.size() - word1.size() : word1.size() - word2.size();
        res += t.substr(m.size(), len);
        return res;
    }

    vector<int> eventualSafeNodes(vector<vector<int>> &graph) {
        int n = graph.size();
        vector<int> color(n);
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            if (eventualSafeNodesDFS(i, color, graph)) {
                res.emplace_back(i);
            }
        }
        return res;
    }

    bool eventualSafeNodesDFS(int node, vector<int> &color, vector<vector<int>> &graph) {
        if (color[node] > 0) return color[node] == 2;
        color[node] = 1;
        for (int i:graph[node]) {
            if (color[node] == 2) {
                continue;
            }
            if (color[node] == 1 && !eventualSafeNodesDFS(i, color, graph)) {
                return false;
            }
        }
        color[node] = 2;
        return true;
    }

    bool findWhetherExistsPath(int n, vector<vector<int>> &graph, int start, int target) {

    }

    vector<bool> checkIfPrerequisite(int n, vector<vector<int>> &prerequisites, vector<vector<int>> &queries) {
        vector<vector<bool>> g(n, vector<bool>(n, false));
        for (auto i:prerequisites) {
            g[i[0]][i[1]] = true;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    if (g[j][i] && g[i][k]) {
                        g[j][k] = true;
                    }
                }
            }
        }
        vector<bool> res;
        for (auto &q : queries)
            if (g[q[0]][q[1]])
                res.push_back(true);
            else
                res.push_back(false);
        return res;
    }

    int getKth(int lo, int hi, int k) {

    }

    int maximalNetworkRank(int n, vector<vector<int>> &roads) {
        vector<unordered_set<int>> edges(n);
        vector<int> deg(n);
        for (const auto &r:roads) {
            edges[r[0]].insert(r[1]);
            edges[r[1]].insert(r[0]);
            ++deg[r[0]];
            ++deg[r[1]];
            int f = -1, s = -2;
            for (int i = 0; i < n; ++i) {

            }
        }
    }

    vector<vector<int>> transpose(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> res(n, vector<int>(m));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res[i][j] = matrix[j][i];
            }
        }
        return res;
    }
};

#endif //LEETCODEMAC_SOLUTION5_H
