#ifndef LEETCODEMAC_TSOLUTION_H
#define LEETCODEMAC_TSOLUTION_H

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

class TSolution {
public:
    void deleteNode(ListNode *node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }

    int maxDepth(TreeNode *root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left) + 1, maxDepth(root->right) + 1);
    }

    ListNode *reverseList(ListNode *head) {
        ListNode *tmp = new ListNode(0);
        while (head != nullptr) {
            ListNode *h = head->next;
            head->next = tmp->next;
            tmp->next = head;
            head = h;
        }
        return tmp->next;
    }

    int removeDuplicates(vector<int> &nums) {
        int i = 0;
        for (int j = 1; j < nums.size(); ++j) {
            if (nums[i] != nums[j]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        vector<int> dp(n);
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < n; ++i) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n - 1];
    }

    int characterReplacement(string s, int k) {
        vector<int> num(26);
        int n = s.size();
        int maxn = 0;
        int left = 0, right = 0;
        while (right < n) {
            num[s[right] - 'A']++;
            maxn = max(maxn, num[s[right] - 'A']);
            if (right - left + 1 - maxn > k) {
                num[s[left] - 'A']--;
                left++;
            }
            right++;
        }
        return right - left;
    }

    vector<vector<int>> subsets(vector<int> &nums) {
        vector<vector<int>> res;
        vector<int> path;
        subsetsBP(nums, res, path, 0);
        return res;
    }

    void subsetsBP(vector<int> &nums, vector<vector<int>> &res, vector<int> &path, int s) {
        res.emplace_back(path);
        for (int i = s; i < nums.size(); ++i) {
            path.emplace_back(nums[i]);
            subsetsBP(nums, res, path, i + 1);
            path.pop_back();
        }
    }

    vector<int> spiralOrder(vector<vector<int>> &matrix) {
        int l = 0, r = matrix[0].size() - 1, t = 0, b = matrix.size() - 1;
        vector<int> res;
        while (l <= r && t <= b) {
            for (int i = l; i <= r; ++i) {
                res.emplace_back(matrix[t][i]);
            }
            for (int i = t + 1; i <= b; ++i) {
                res.emplace_back(matrix[i][r]);
            }
            if (l < r && t < b) {
                for (int i = r - 1; i > l; i--) {
                    res.emplace_back(matrix[b][i]);
                }
                for (int i = b; i > t; --i) {
                    res.emplace_back(matrix[i][l]);
                }
            }
            l++;
            r--;
            t++;
            b--;
        }
        return res;
    }

    vector<vector<int>> permute(vector<int> &nums) {
        vector<vector<int>> res;
        vector<int> path;
        vector<bool> vis(nums.size());
        permuteBP(nums, res, path, vis);
        return res;
    }

    void permuteBP(vector<int> &nums, vector<vector<int>> &res, vector<int> &path, vector<bool> &vis) {
        if (path.size() == nums.size()) {
            res.emplace_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (!vis[i]) {
                vis[i] = true;
                path.emplace_back(nums[i]);
                permuteBP(nums, res, path, vis);
                path.pop_back();
                vis[i] = false;
            }
        }
    }

    int kthSmallest(TreeNode *root, int k) {
        int res = 0;
        vector<int> vec;
        kthSmallestHelp(root, vec);
        return vec[k - 1];
    }

    void kthSmallestHelp(TreeNode *root, vector<int> &vec) {
        if (!root) return;
        kthSmallestHelp(root->left, vec);
        vec.emplace_back(root->val);
        kthSmallestHelp(root->right, vec);
    }

    vector<int> productExceptSelf(vector<int> &nums) {
        int m = nums.size();
        vector<int> res(m);
        int k = 1;
        for (int i = 0; i < m; ++i) {
            res[i] = k;
            k *= nums[i];
        }
        k = 1;
        for (int i = m - 1; i >= 0; --i) {
            res[i] *= k;
            k *= nums[i];
        }
        return res;
    }

    vector<int> grayCode(int n) {
        vector<int> res(pow(2, n));
        for (int i = 0; i <= pow(2, n); ++i) {
            res[i] = i ^ (i >> 1);
        }
        return res;
    }

    ListNode *sortList(ListNode *head) {
        if (head == nullptr || head->next == nullptr) return head;
        ListNode *fast = head->next, *slow = head;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode *tmp = slow->next;
        slow->next = nullptr;
        ListNode *right = sortList(tmp);
        ListNode *left = sortList(head);
        ListNode *h = new ListNode(0);
        ListNode *res = h;
        while (left != nullptr && right != nullptr) {
            if (left->val < right->val) {
                h->next = left;
                left = left->next;
            } else {
                h->next = right;
                right = right->next;
            }
            h = h->next;
        }
        h->next = left != nullptr ? left : right;
        return res->next;
    }

    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
        if (root == nullptr || p == root || q == root) {
            return root;
        }
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (left == nullptr) return right;
        if (right == nullptr) return left;
        return root;
    }

    vector<double> medianSlidingWindow(vector<int> &nums, int k) {
        vector<double> res(nums.size() - k + 1);
        for (int i = 0; i < nums.size() - k; ++i) {
            vector<int> tmp(nums.begin() + i, nums.begin() + i + k);
            sort(tmp.begin(), tmp.end());
            if (k % 2) {
                res[i] = tmp[i + k / 2];
            } else {
                res[i] = (tmp[i + k / 2] + tmp[i + k / 2 - 1]) / 2.0f;
            }
        }
        return res;
    }

    int maxArea(vector<int> &height) {
        int res = -1;
        int l = 0, r = height.size() - 1;
        while (l < r) {
            int a = min(height[l], height[r]) * (r - l);
            res = max(res, a);
            if (height[l] <= height[r]) {
                l++;
            } else {
                r--;
            }
        }
        return res;
    }

    int uniquePaths(int m, int n) {
        long long ans = 1;
        for (int x = n, y = 1; y < m; ++x, ++y) {
            ans = ans * x / y;
        }
        return ans;
    }

    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    slow = slow->next;
                    fast = fast->next;
                }
                return fast;
            }
        }
        return nullptr;
    }

    int threeSumClosest(vector<int> &nums, int target) {

    }

    string multiply(string num1, string num2) {

    }

    string addStrings(string num1, string num2) {

    }

    vector<vector<int>> threeSum(vector<int> &nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for (int f = 0; f < n; ++f) {
            if (f > 0 && nums[f] == nums[f - 1]) {
                continue;
            }
            int third = n - 1;
            int target = -nums[f];
            for (int s = f + 1; s < n; ++s) {
                if (s > f + 1 && nums[s] == nums[s - 1]) {
                    continue;
                }
                while (s < third && nums[s] + nums[third] > target) {
                    --third;
                }
                if (s == third) {
                    break;
                }
                if (nums[s] + nums[third] == target) {
                    res.emplace_back(vector<int>{nums[f], nums[s], nums[third]});
                }
            }
        }
        return res;
    }

    ListNode *mergeKLists(vector<ListNode *> &lists) {
        if (lists.size() == 0) return nullptr;
        ListNode *res = lists[0];
        for (int i = 1; i < lists.size(); ++i) {
            res = mergeTwo(res, lists[i]);
        }
        return res;
    }

    ListNode *mergeTwo(ListNode *a, ListNode *b) {
        ListNode *res = new ListNode(0), *tmp = res;
        while (a != nullptr && b != nullptr) {
            if (a->val < b->val) {
                tmp->next = a;
                a = a->next;
            } else {
                tmp->next = b;
                b = b->next;
            }
            tmp = tmp->next;
        }
        tmp->next = a == nullptr ? b : a;
        return res->next;
    }

    int maxPathSum(TreeNode *root) {
        int res = 0;
        maxGain(root, res);
        return res;
    }

    int maxGain(TreeNode *node, int &ans) {
        if (node == nullptr) return 0;
        int left = max(maxGain(node->left, ans), 0);
        int right = max(maxGain(node->right, ans), 0);
        int path = node->val + left + right;
        ans = max(ans, path);
        return node->val + max(left, right);
    }

    void moveZeroes(vector<int> &nums) {
        if (nums.size() == 0) return;
        int j = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] != 0) {
                swap(nums[i], nums[j]);
                j++;
            }
        }
    }

    vector<int> findDisappearedNumbers(vector<int> &nums) {
        vector<int> res;
        vector<bool> tmp(nums.size() + 1);
        for (int i:nums) {
            tmp[i] = true;
        }
        for (int i = 1; i < tmp.size(); ++i) {
            if (!tmp[i]) res.emplace_back(i);
        }
        return res;
    }

    bool isSymmetric(TreeNode *root) {
        return isSymmetricCheck(root, root);
    }

    bool isSymmetricCheck(TreeNode *p, TreeNode *q) {
        if (!p && !q) return true;
        if (!p || !q) return false;
        return p->val == q->val && isSymmetricCheck(p->left, q->right) && isSymmetricCheck(p->right, q->left);
    }

    int diameterOfBinaryTree(TreeNode *root) {
        int res = 0;
        diameterOfBinaryTreeHelp(root, res);
        return res;
    }

    int diameterOfBinaryTreeHelp(TreeNode *root, int &res) {
        if (root == nullptr) return 0;
        if (root->left == nullptr && root->right == nullptr) return 1;
        int t = diameterOfBinaryTreeHelp(root->left, res) + diameterOfBinaryTreeHelp(root->right, res);
        res = max(t, res);
        return t + 1;
    }
};

class MedianFinder {
private:
    int count;
    priority_queue<int, vector<int>, greater<>> minHeap;
    priority_queue<int, vector<int>> maxHeap;

public:
    /** initialize your data structure here. */
    MedianFinder() {
        count = 0;
        minHeap = priority_queue<int, vector<int>, greater<>>();
        maxHeap = priority_queue<int, vector<int>>();
    }

    void addNum(int num) {
        count++;
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();
        if ((count & 1) != 0) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if ((count & 1) == 0) {
            return (double) (maxHeap.top() + minHeap.top()) / 2;
        } else {
            return (double) maxHeap.top();
        }
    }
};

class LRUCache {
public:
    LRUCache(int capacity) {

    }

    int get(int key) {

    }

    void put(int key, int value) {

    }
};

class MinStack {
public:
    stack<int> s1;
    stack<int> s2;

    /** initialize your data structure here. */
    MinStack() {
        s1 = stack<int>();
        s2 = stack<int>();
    }

    void push(int x) {
        s1.push(x);
        if (s2.empty()) {
            s2.push(x);
        } else {
            if (x <= s2.top()) {
                s2.push(x);
            }
        }
    }

    void pop() {
        if (s1.top() == s2.top()) {
            s2.pop();
        }
        s1.pop();
    }

    int top() {
        return s1.top();
    }

    int getMin() {
        return s2.top();
    }
};


#endif //LEETCODEMAC_TSOLUTION_H
