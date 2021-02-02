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

    }
};

#endif //LEETCODEMAC_TSOLUTION_H
