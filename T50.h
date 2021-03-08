#ifndef LEETCODEMAC_T50_H
#define LEETCODEMAC_T50_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>
#include "Solution5.h"

using namespace std;

class T50 {
public:
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *dummy = new ListNode(0);
        ListNode *cur = dummy;
        int sum = 0, c = 0;
        while (l1 != nullptr || l2 != nullptr) {
            int x = l1 == nullptr ? 0 : l1->val;
            int y = l2 == nullptr ? 0 : l2->val;
            sum = x + y + c;
            c = sum / 10;
            ListNode *tmp = new ListNode(sum % 10);
            cur->next = tmp;
            cur = tmp;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        if (c) {
            cur->next = new ListNode(1);
        }
        return dummy->next;
    }

    double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {

    }

    string longestPalindrome(string s) {
        int n = s.size();
        string res = "";
        vector<vector<bool>> dp(n, vector<bool>(n));
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][i] = true;
                } else if (l == 1) {
                    dp[i][j] = s[i] == s[j];
                } else {
                    dp[i][j] = s[i] == s[j] && dp[i + 1][j - 1];
                }
                if (dp[i][j] && l + 1 > res.size()) {
                    res = s.substr(i, l + 1);
                }
            }
        }
        return res;
    }

    int reverse1(int x) {
        int res = 0;
        while (x != 0) {
            int t = x % 10;
            if (res > INT_MAX / 10 || res == INT_MAX / 10 && t > INT_MAX % 10) return 0;
            if (res < INT_MIN / 10 || res == INT_MIN / 10 && t < INT_MIN % 10) return 0;
            res = res * 10 + t;
            x /= 10;
        }
        return res;
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

    int maxArea(vector<int> &height) {
        int l = 0, r = height.size() - 1;
        int res = INT_MIN;
        while (l < r) {
            res = max(res, min(height[l], height[r]) * (r - l));
            if (height[l] < height[r]) l++;
            else r--;
        }
        return res;
    }

    string longestCommonPrefix(vector<string> &strs) {
        if (strs.size() == 0) return "";
        string tar = strs[0];
        for (int i = 0; i < tar.size(); ++i) {
            for (string cur:strs) {
                if (i == cur.size()) {
                    return tar.substr(0, i);
                }
                if (cur[i] != tar[i]) {
                    return tar.substr(0, i);
                }
            }
        }
        return tar;
    }

    vector<vector<int>> threeSum(vector<int> &nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int r = nums.size() - 1;
            int tar = -nums[i];
            for (int j = i + 1; j < nums.size(); ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                while (j < r && nums[j] + nums[r] > tar) {
                    --r;
                }
                if (j == r) {
                    break;
                }
                if (nums[j] + nums[r] == tar) {
                    res.emplace_back(vector<int>{nums[i], nums[j], nums[r]});
                }
            }
        }
        return res;
    }

    int threeSumClosest(vector<int> &nums, int target) {
        sort(nums.begin(), nums.end());
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.size(); ++i) {
            int l = i + 1, r = nums.size() - 1;
            while (l < r) {

            }
        }
    }

    ListNode *mergeKLists(vector<ListNode *> &lists) {
        if (lists.size() == 0) return nullptr;
        ListNode *dummy = lists[0];
        for (int i = 1; i < lists.size(); ++i) {
            dummy = mergeTwo(dummy, lists[i]);
        }
        return dummy;
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

    int search(vector<int> &nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (target == nums[mid]) return mid;
            if (nums[mid] > nums[r]) {
                if (target > nums[l] && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (target > nums[mid] && target < nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        if (nums[l] != target) return -1;
        return l;
    }

    vector<vector<int>> permute(vector<int> &nums) {
        vector<vector<int>> res;
        vector<int> path;
        vector<bool> vis(nums.size());
        permuteBP(nums, res, vis, path);
        return res;
    }

    void permuteBP(vector<int> &nums, vector<vector<int>> &res, vector<bool> &vis, vector<int> &path) {
        if (path.size() == nums.size()) {
            res.emplace_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (!vis[i]) {
                vis[i] = true;
                path.push_back(nums[i]);
                permuteBP(nums, res, vis, path);
                path.pop_back();
                vis[i] = false;
            }
        }
    }

    int maxSubArray(vector<int> &nums) {
        vector<int> dp(nums);
        int res = dp[0];
        for (int i = 1; i < nums.size(); ++i) {
            dp[i] = max(dp[i], dp[i - 1] + nums[i]);
            res = max(res, dp[i]);
        }
        return res;
    }

    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res = vector<vector<int>>(n, vector<int>(n));
        int cur = 1;
        int l = 0, r = res.size() - 1, t = 0, b = res.size() - 1;
        while (true) {
            for (int i = l; i <= r; ++i) res[t][i] = cur++;
            if (++t > b) break;
            for (int i = t; i <= b; ++i) res[i][r] = cur++;
            if (--r < l) break;
            for (int i = r; i >= l; --i) res[b][i] = cur++;
            if (--b < t) break;
            for (int i = b; i >= t; --i) res[i][l] = cur++;
            if (++l > r) break;
        }
        return res;
    }

    ListNode *rotateRight(ListNode *head, int k) {
        ListNode *slow = head, *fast = head;
        for (int i = 0; i < k; ++i) {
            slow = slow->next;
            fast = fast->next;
        }
        while (fast->next != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        fast->next = head;
        ListNode *tmp = slow->next;
        slow->next = nullptr;
        return tmp;
    }

    vector<vector<int>> subsets(vector<int> &nums) {
        vector<int> path;
        vector<vector<int>> res;
        subsetsHelp(res, path, nums, 0);
        return res;
    }

    int res = 0;
    int count = 0;

    int kthSmallest(TreeNode *root, int k) {
        int t = k;
        kthSmallestHelp(root, t);
        return res;
    }

    void kthSmallestHelp(TreeNode *root, int &k) {
        if (root == nullptr) return;
        kthSmallestHelp(root->left, k);
        ++count;
        if (count == k) {
            res = root->val;
            return;
        }
        kthSmallestHelp(root->right, k);
    }

    void subsetsHelp(vector<vector<int>> &res, vector<int> &path, vector<int> &nums, int l) {
        res.emplace_back(path);
        for (int i = l; i < nums.size(); ++i) {
            path.emplace_back(nums[i]);
            subsetsHelp(res, path, nums, i + 1);
            path.pop_back();
        }
    }

    int res1 = INT_MIN;

    int maxPathSum(TreeNode *root) {
        maxPathSumHelp(root);
        return res1;
    }

    int maxPathSumHelp(TreeNode *node) {
        if (node == nullptr) return 0;
        int left = max(maxPathSumHelp(node->left), 0);
        int right = max(maxPathSumHelp(node->right), 0);
        res1 = max(res, right + left + node->val);
        return node->val + max(left, right);
    }

    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if (slow == fast) return true;
        }
        return false;
    }

    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
        if (root == nullptr || p == root || q == root) return root;
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);
        if (!left) return right;
        if (!right) return left;
        return root;
    }

    string reverseWords(string s) {
        string res = "";
        string cur = "";
        for (char c:s) {
            if (c == ' ') {
                reverse(cur.begin(), cur.end());
                res += cur + ' ';
                cur = "";
            } else {
                cur += c;
            }
        }
        reverse(cur.begin(), cur.end());
        res += cur;
        return res;
    }

    class LRUCache {
    public:
        LRUCache(int capacity) {
            cap = capacity;
        }

        int get(int key) {
            auto it = mp.find(key);
            if (it != mp.end()) {
                auto tmp = make_pair(key, it->second->second);
                l.erase(it->second);
                l.push_front(tmp);
                mp.erase(key);
                mp.insert(make_pair(key, l.begin()));
                return tmp.second;
            }
            return -1;
        }

        void put(int key, int value) {
            auto it = mp.find(key);
            if (it != mp.end()) {
                l.erase(it->second);
                mp.erase(key);
            }
            pair<int, int> t{key, value};
            l.push_front(t);
            mp.insert(make_pair<>(key, l.begin()));
            if (l.size() > cap) {
                mp.erase(l.back().first);
                l.pop_back();
            }
        }

        int cap;
        unordered_map<int, list<pair<int, int>>::iterator> mp;
        list<pair<int, int>> l;
    };

};

#endif //LEETCODEMAC_T50_H
