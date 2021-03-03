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

    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *res = new ListNode(0), *t = res;
        int sum = 0, c = 0;
        while (l1 != nullptr || l2 != nullptr) {
            int x = l1 == nullptr ? 0 : l1->val;
            int y = l2 == nullptr ? 0 : l2->val;
            sum = x + y + c;
            ListNode *node = new ListNode(sum % 10);
            c = sum / 10;
            t->next = node;
            t = t->next;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        if (c) t->next = new ListNode(1);
        return res->next;
    }

    int findString(vector<string> &words, string s) {
        int l = 0, r = words.size();
        while (l < r) {
            int mid = (l + r) / 2;
            int tmp = mid;
            while (mid < r && words[mid] == "") {
                mid++;
            }
            if (mid == r) {
                r = tmp;
                continue;
            }
            if (words[mid] == s) {
                return mid;
            } else if (words[mid] > s) {
                r = mid;
            } else if (words[mid] < s) {
                l = mid + 1;
            }
        }
        return -1;
    }

    int smallestDifference(vector<int> &a, vector<int> &b) {

    }

    int buildTreeIdx = 0;

    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        unordered_map<int, int> mp;
        for (int i = 0; i < inorder.size(); ++i) {
            mp[inorder[i]] = i;
        }
        return buildTreeHelp(preorder, inorder, 0, inorder.size() - 1, mp);
    }

    TreeNode *buildTreeHelp(vector<int> &pre, vector<int> &in, int l, int r, unordered_map<int, int> &mp) {
        if (l > r) return nullptr;
        int val = pre[buildTreeIdx];
        TreeNode *root = new TreeNode(val);
        buildTreeIdx++;
        root->left = buildTreeHelp(pre, in, l, mp[val] - 1, mp);
        root->right = buildTreeHelp(pre, in, mp[val] + 1, r, mp);
        return root;
    }

    int cuttingRope(int n) {
        vector<int> dp(n + 1);
        dp[2] = 1;
        for (int i = 3; i <= n; ++i) {
            for (int j = 2; j < i; ++j) {
                dp[i] = max(dp[i], max(j * dp[i - j], j * (i - j)));
            }
        }
        return dp[n];
    }

    int minArray(vector<int> &numbers) {
        int l = 0, r = numbers.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
        }
    }

    TreeNode *mirrorTree(TreeNode *root) {
        if (root == nullptr) return nullptr;
        TreeNode *tmp = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(tmp);
        return root;
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
        stack <string> stk;
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

    vector<string> letterCombinations(string digits) {
        vector<string> v = {"abc", "def", "ghi", "jkl", "mno", "prqs", "tuv", "wxyz"};
        vector<string> res;
        string path = "";
        letterCombinationsBP(digits, path, res, v, 0);
        return res;
    }

    void letterCombinationsBP(string digits, string &path, vector<string> &res, vector<string> &v, int k) {
        if (path.size() == digits.size()) {
            res.emplace_back(path);
            return;
        }
        for (int i = k; i < digits.size(); ++i) {
            char d = digits[i];
            for (char c:v[d - '0']) {
                path += c;
                letterCombinationsBP(digits, path, res, v, i + 1);
                path.pop_back();
            }
        }
    }

    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string path;
        generateParenthesisBP(res, path, 0, 0, n);
        return res;
    }

    void generateParenthesisBP(vector<string> &res, string &path, int l, int r, int n) {
        if (path.size() == n * 2) {
            res.emplace_back(path);
            return;
        }
        if (l < n) {
            path.push_back('(');
            generateParenthesisBP(res, path, l + 1, r, n);
            path.pop_back();
        }
        if (r < l) {
            path.push_back(')');
            generateParenthesisBP(res, path, l, r + 1, n);
            path.pop_back();
        }
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
        unordered_map<int, unordered_set<int>> mp;
        vector<bool> vis(n, 0);
        for (int i = 0; i < graph.size(); ++i) {
            mp[graph[i][0]].insert(graph[i][1]);
        }
        return findWhetherExistsPathDFS(start, target, mp, vis);
    }

    bool
    findWhetherExistsPathDFS(int start, int target, unordered_map<int, unordered_set<int>> &mp, vector<bool> &vis) {
        if (start == target) return true;
        if (vis[start]) {
            return false;
        }
        vis[start] = true;
        for (const auto &nei:mp[start]) {
            if (findWhetherExistsPathDFS(nei, target, mp, vis)) {
                return true;
            }
        }
        return false;
    }

    TreeNode *sortedArrayToBST(vector<int> &nums) {
        if (nums.size() == 0) return nullptr;
        return sortedArrayToBSTHelp(nums, 0, nums.size() - 1);
    }

    TreeNode *sortedArrayToBSTHelp(vector<int> &nums, int l, int r) {
        if (l < 0 || r >= nums.size() || l > r) return nullptr;
        int mid = (l + r) / 2;
        TreeNode *n = new TreeNode(nums[mid]);
        n->left = sortedArrayToBSTHelp(nums, l, mid - 1);
        n->right = sortedArrayToBSTHelp(nums, mid + 1, r);
        return n;
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

    bool isValidSudoku(vector<vector<char>> &board) {

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

    bool validateStackSequences(vector<int> &pushed, vector<int> &popped) {
        stack<int> stk;
        int l = 0;
        for (int i = 0; i < pushed.size(); ++i) {
            stk.push(pushed[i]);
            while (!stk.empty() && stk.top() == popped[l]) {
                stk.pop();
                l++;
            }
        }
        return stk.empty();
    }

    vector<vector<int>> pathSum(TreeNode *root, int sum) {
        vector<vector<int>> res;
        vector<int> path;
        pathSumBP(root, sum, res, path);
        return res;
    }

    void pathSumBP(TreeNode *root, int sum, vector<vector<int>> &res, vector<int> &path) {
        if (root == nullptr) return;
        path.emplace_back(root->val);
        sum -= root->val;
        if (sum == 0 && root->left == nullptr && root->right == nullptr) {
            res.emplace_back(path);
        }
        pathSumBP(root->left, sum, res, path);
        pathSumBP(root->right, sum, res, path);
        path.pop_back();
    }

    bool increasingTriplet(vector<int> &nums) {
        int one = INT_MIN, two = INT_MIN;

    }

    int reversePairs(vector<int> &nums) {

    }

    bool isSymmetric(TreeNode *root) {
        if (root == nullptr) return true;
        return isSymmetricHelp(root->left, root->right);
    }

    bool isSymmetricHelp(TreeNode *p, TreeNode *q) {
        if (p == nullptr && q == nullptr) return true;
        if (p == nullptr || q == nullptr) return false;
        return p->val == q->val && isSymmetricHelp(p->left, q->right) && isSymmetricHelp(p->right, q->left);
    }

    vector<int> exchange(vector<int> &nums) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            if ((nums[l] & 1) != 0) {
                l++;
                continue;
            }
            if ((nums[r] & 1) != 1) {
                r--;
                continue;
            }
            swap(nums[l], nums[r]);
        }
        return nums;
    }

    int partition(vector<int> &nums, int l, int r) {
        int p = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= p) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
            swap(nums[i + 1], nums[r]);
            return i + 1;
        }
    }

    void randomized_select(vector<int> &arr, int l, int r, int k) {
        if (l >= r) return;
        int pos = partition(arr, l, r);
        int num = pos - l + 1;
        if (k == num) {
            return;
        } else if (k < num) {
            randomized_select(arr, l, pos - 1, k);
        } else {
            randomized_select(arr, pos + 1, r, k - num);
        }
    }

    vector<int> getLeastNumbers(vector<int> &arr, int k) {

    }

    vector<int> findSwapValues(vector<int> &array1, vector<int> &array2) {
        sort(array1.begin(), array1.end());
        sort(array2.begin(), array2.end());
        int s1 = accumulate(array1.begin(), array1.end(), 0);
        int s2 = accumulate(array1.begin(), array1.end(), 0);
        if (s2 > s1) {
            swap(array1, array2);
        }
        int i = 0, j = 0, tar = abs(s1 - s2) / 2;
        while (i < array1.size() && j < array2.size()) {
            int x = i >= array1.size() ? array1.back() : array1[i];
            int y = j >= array2.size() ? array2.back() : array2[j];
            if (x - y == tar) return vector<int>{x, y};
            if (x - y > tar) {
                j++;
            } else {
                i++;
            }
        }
        return vector<int>{};
    }

    int strToInt(string str) {
        int res = 0;
        int i = 0, sign = 1, n = str.size();
        if (n == 0) return 0;
        int edge = 214748364;
        while (str[i] == ' ') {
            ++i;
            if (i == n) return 0;
        }
        if (str[i] == '-') {
            sign = -1;
            i++;
        }
        if (str[i] == '+') {
            i++;
        }
        for (int j = i; j < n; ++j) {
            if (str[j] < '0' || str[j] > '9') break;
            res = res * 10 + (str[j] - '0');
            if (res >= edge) {
                return sign == 1 ? INT_MAX : INT_MIN;
            }
        }
        return sign * res;
    }

    vector<int> constructArr(vector<int> &a) {
        int n = a.size();
        vector<int> res(n, 1);
        int l = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = l;
            l *= a[i];
        }
        int r = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] *= r;
            r *= a[i];
        }
        return res;
    }

    ListNode *removeDuplicateNodes(ListNode *head) {
        if (head == nullptr) return head;
        unordered_set<int> st{head->val};
        ListNode *pos = head;
        while (pos->next != nullptr) {
            ListNode *cur = pos->next;
            if (!st.count(cur->val)) {
                st.insert(cur->val);
                pos = pos->next;
            } else {
                pos->next = pos->next->next;
            }
        }
        pos->next = nullptr;
        return head;
    }

    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if (slow == fast) {
                break;
            }
        }
        if (fast->next != nullptr && fast->next->next != nullptr) return nullptr;
        fast = head;
        while (fast != slow) {
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }

    vector<ListNode *> listOfDepth(TreeNode *tree) {
        vector<ListNode *> res;
        queue<TreeNode *> q;
        if (tree == nullptr) return res;
        q.push(tree);
        while (!q.empty()) {
            int n = q.size();
            ListNode *tmp = new ListNode(0), *tt = tmp;
            for (int i = 0; i < n; ++i) {
                TreeNode *t = q.front();
                q.pop();
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
                tt->next = new ListNode(t->val);
                tt = tt->next;
            }
            res.emplace_back(tmp->next);
        }
        return res;
    }

    bool isBalanced(TreeNode *root) {

    }

    int getHeight(TreeNode *root) {
        if (root == nullptr) return 0;
    }

    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
        if (root == p || root == q || root == nullptr) {
            return root;
        }
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (!left) return right;
        if (!right) return left;
        return root;
    }

    int insertBits(int N, int M, int i, int j) {
        for (int k = i; k <= j; k++) {
            N &= ~(1 << k);
        }
        return N + (M << i);
    }

    vector<vector<string>> groupAnagrams(vector<string> &strs) {
        vector<vector<string>> res;
        unordered_map<string, vector<string>> mp;
        for (string s:strs) {
            string t = s;
            sort(t.begin(), t.end());
            auto i = mp.find(t);
            if (i == mp.end()) {
                mp.insert(pair<string, vector<string>>{t, {s}});
            } else {
                i->second.emplace_back(s);
            }
        }
        for (auto i:mp) {
            res.emplace_back(i.second);
        }
        return res;
    }

    void wiggleSort(vector<int> &nums) {

    }

    vector<string> getValidT9Words(string num, vector<string> &words) {

    }

    int calculate(string s) {
        int n = s.size();
        stack<int> ss;
        int num = 0;
        char si = '+';
        int top = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] >= '0' && s[i] <= '9') {
                num = num * 10 + (s[i] - '0');
            } else {
                if (si == '+') {
                    ss.push(num);
                } else if (si == '-') {
                    ss.push(-num);
                } else if (si == '*') {
                    top = ss.top();
                    ss.pop();
                    ss.push(top * num);
                } else {
                    top = ss.top();
                    ss.pop();
                    ss.push(top / num);
                }
                num = 0;
                si = s[i];
            }
        }
        int res = 0;
        while (!ss.empty()) {
            res += ss.top();
            ss.pop();
        }
        return res;
    }

    int massage(vector<int> &nums) {
        int n = nums.size();
        if (!n) return 0;
        vector<vector<int>> dp(n, vector<int>(2, 0));
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        for (int i = 1; i < n; ++i) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }
        return max(dp[n - 1][0], dp[n - 1][1]);
    }

    vector<vector<int>> multiSearch(string big, vector<string> &smalls) {

    }

    vector<int> shortestSeq(vector<int> &big, vector<int> &small) {
        int n = big.size();
        vector<int> res;
        unordered_map<int, int> need;
        int minLen = n, diff = 0;
        for (auto &e:small) {
            need[e]++;
            diff++;
        }
        int l = 0, r = 0;
        while (r < n) {
            if (need.count(big[r]) && --need[big[r]] >= 0) --diff;
            while (diff == 0) {
                if (r - l < minLen) {
                    minLen = r - l;
                    res = {l, r};
                }
                if (need.count(big[l]) && ++need[big[l]] > 0) ++diff;
                ++l;
            }
            ++r;
        }
        return res;
    }

    bool isMatch(string s, string p) {
        vector<vector<bool>> dp(s.size() + 1, vector<bool>(p.size() + 1));
        dp[0][0] = true;
        for (int i = 1; i <= p.size(); ++i) {
            if (p[i - 1] == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i < dp.size(); ++i) {
            for (int j = 1; j < dp[0].size(); ++j) {
                if (s[i - 1] == p[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p[j - 1] == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 1] | dp[i - 1][j];
                }
            }
        }
        return dp[s.size()][p.size()];
    }

    bool canJump(vector<int> &nums) {
        int r = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (i > r) return false;
            r = max(r, i + nums[i]);
        }
        return true;
    }

    double myPow(double x, int n) {
        if (x == 0.0) return 0.0;
        long b = n;
        double res = 1;
        if (b < 0) {
            b = -b;
            x = 1 / x;
        }
        while (b != 0) {
            if ((b & 1) == 1) res *= x;
            x *= x;
            b = b >> 1;
        }
        return res;
    }

    int trap(vector<int> &height) {
        if (height.size() < 3) return 0;
        int l = 0, r = height.size() - 1;
        int leftMax = height[l], rightMax = height[r];
        int res = 0;
        while (l < r) {
            if (leftMax < rightMax) {
                res += leftMax - height[l++];
                leftMax = max(height[l], leftMax);
            } else {
                res += rightMax - height[r--];
                rightMax = max(height[r], rightMax);
            }
        }
        return res;
    }

    bool isMonotonic(vector<int> &A) {
        bool inc = true;
        bool dec = true;
        for (int i = 1; i < A.size(); ++i) {
            if (A[i] < A[i - 1])
                inc = false;
            if (A[i] > A[i - 1])
                dec = false;
            if (!inc && !dec)
                return false;
        }
        return true;
    }

    bool isFlipedString(string s1, string s2) {
        return s1.size() == s2.size() && (s1 + s1).find(s2) != -1;
    }

    vector<string> findLadders(string beginWord, string endWord, vector<string> &wordList) {

    }

    vector<int> findSquare(vector<vector<int>> &matrix) {

    }

    int reverse1(int x) {
        int res = 0;
        while (x) {
            int t = x % 10;
            x /= 10;
            if (res > INT_MAX / 10 || res == INT_MAX / 10 && t > 7) return 0;
            if (res < INT_MIN / 10 || res == INT_MIN / 10 && t < -8) return 0;
            res = res * 10 + t;
        }
        return res;
    }

    int lengthOfLongestSubstring(string s) {
        unordered_set<char> st;
        int res = 0;
        int l = 0, r = 0;
        while (r < s.size()) {
            while (st.count(s[r])) {
                st.erase(s[l]);
                l++;
            }
            st.insert(s[r]);
            res = max(res, r - l + 1);
            r++;
        }
        return res;
    }

    string longestPalindrome(string s) {
        int n = s.size();
        string ans = "";
        vector<vector<bool>> dp(n, vector<bool>(n));
        for (int l = 0; l < s.size(); ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][i] = true;
                } else if (l == 1) {
                    dp[i][j] = s[i] == s[j];
                } else {
                    dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.size()) {
                    ans = s.substr(i, l + 1);
                }
            }
        }
        return ans;
    }

    string longestCommonPrefix(vector<string> &strs) {
        string cur = strs[0];
        for (int i = 0; i < cur.size(); ++i) {
            char t = cur[i];
            for (const auto s:strs) {
                if (i == s.size()) return s.substr(i + 1);
                if (s[i] == t) {
                    if (i == 0) return "";
                    return s.substr(i);
                }
            }
        }
        return cur;
    }
};

class LRUCache1 {
public:
    LRUCache1(int capacity) {
        cap = capacity;
    }

    int get(int key) {
        auto it = mp.find(key);
        if (it == mp.end()) return -1;
        auto target = it->second;
        pair<int, int> n{target->first, target->second};
        cache.push_front(n);
        cache.erase(target);
        mp.erase(key);
        mp.emplace(key, cache.begin());
        return n.second;
    }

    void put(int key, int value) {
        auto it = mp.find(key);
        if (it != mp.end()) {
            cache.erase(it->second);
            mp.erase(key);
        }
        pair<int, int> n{key, value};
        cache.push_front(n);
        mp.emplace(key, cache.begin());
        if (cache.size() > cap) {
            mp.erase(cache.back().first);
            cache.pop_back();
        }
    }

private:
    size_t cap = 0;
    list<pair<int, int>> cache;
    unordered_map<int, list<pair<int, int>>::iterator> mp;
};

class WordsFrequency {
public:
    unordered_map<string, int> mp;

    WordsFrequency(vector<string> &book) {
        mp = unordered_map<string, int>();
        for (auto b:book) {
            if (mp.count(b)) {
                mp[b]++;
            } else {
                mp.insert(pair<string, int>{b, 1});
            }
        }
    }

    int get(string word) {
        if (mp.count(word)) {
            return mp[word];
        } else {
            return 0;
        }
    }
};

class NumArray1 {
public:
    vector<int> v;

    NumArray1(vector<int> &nums) {
        v = vector<int>(nums.size() + 1);
        v[0] = 0;
        for (int i = 1; i < v.size(); ++i) {
            v[i] = v[i - 1] + nums[i - 1];
        }
    }

    int sumRange(int i, int j) {
        return v[j + 1] - v[i];
    }
};

class NumMatrix {
public:
    vector<vector<int>> v;

    NumMatrix(vector<vector<int>> &matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) return;
        v = vector<vector<int>>(matrix.size() + 1, vector<int>(matrix[0].size() + 1));
        for (int i = 1; i < v.size(); ++i) {
            for (int j = 1; j < v[0].size(); ++j) {
                v[i][j] = v[i - 1][j] + v[i][j - 1] - v[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return v[row2 + 1][col2 + 1] - v[row2 + 1][col1] - v[row1][col2 + 1] + v[row1][col1];
    }
};


#endif //LEETCODEMAC_SOLUTION5_H
