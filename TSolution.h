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
};

#endif //LEETCODEMAC_TSOLUTION_H
