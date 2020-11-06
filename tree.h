#ifndef LEETCODEMAC_TREE_H
#define LEETCODEMAC_TREE_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

#include "solution.h"

using namespace std;

class Node {
public:
    int val;
    vector<Node *> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node *> _children) {
        val = _val;
        children = _children;
    }
};

class TreeSolution {
public:
    bool isBalanced(TreeNode *root) {
        return isBalancedHelp(root) != -1;
    }

    int isBalancedHelp(TreeNode *root) {
        if (root == nullptr) return 0;
        int left = isBalancedHelp(root->left);
        if (left == -1) return -1;
        int right = isBalancedHelp(root->right);
        if (right == -1) return -1;
        return abs(left - right) < 2 ? max(left, right) + 1 : -1;
    }

    int maxDepth(TreeNode *root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }

    int minDepth(TreeNode *root) {
        if (root == nullptr) return 0;
        int m1 = minDepth(root->left);
        int m2 = minDepth(root->right);
        return root->left == nullptr || root->right == nullptr ? m1 + m2 + 1 : min(m1, m2) + 1;
    }

    int maxDepth(Node *root) {
        if (root == nullptr) return 0;
        int res = 0;
        for (auto &c:root->children) {
            if (c) res = max(res, maxDepth(c));
        }
        return res + 1;
    }

    vector<vector<int>> levelOrder(TreeNode *root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            vector<int> tmp(size);
            for (int i = 0; i < size; ++i) {
                auto node = q.front();
                q.pop();
                tmp[i] = node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            res.emplace_back(tmp);
        }
        return res;
    }

    vector<vector<int>> levelOrder(Node *root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<Node *> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            vector<int> tmp(size);
            for (int i = 0; i < size; ++i) {
                auto node = q.front();
                q.pop();
                tmp[i] = node->val;
                for (auto c:node->children) {
                    if (c) q.push(c);
                }
            }
            res.emplace_back(tmp);
        }
        return res;
    }

    vector<int> preorder(Node *root) {
        vector<int> res;
        if (!root) return res;
        preorderHelp(root, res);
        return res;
    }

    void preorderHelp(Node *root, vector<int> &res) {
        if (!root) return;
        res.emplace_back(root->val);
        for (auto c:root->children) {
            if (c) {
                preorderHelp(c, res);
            }
        }
    }

    vector<int> postorder(Node *root) {
        vector<int> res;
        if (!root) return res;
        postorderHelp(root, res);
        return res;
    }

    void postorderHelp(Node *root, vector<int> &res) {
        if (!root) return;
        for (auto c:root->children) {
            if (c) postorderHelp(c, res);
        }
        res.emplace_back(root->val);
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
                if (level % 2 == 0) tmp[i] = t->val;
                else tmp[size - i - 1] = t->val;
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
            level++;
            res.emplace_back(tmp);
        }
        return res;
    }

    int countPairs(TreeNode *root, int distance) {
        int ans = 0;
        countPairsHelp(root, distance, ans);
        return ans;
    }

    vector<int> countPairsHelp(TreeNode *root, int dis, int &ans) {
        if (!root) return {};
        if (root->left == nullptr && root->right == nullptr) return {0};

        vector<int> ret;
        auto left = countPairsHelp(root->left, dis, ans);
        for (auto &e:left) {
            if (++e > dis) continue;
            ret.emplace_back(e);
        }
        auto right = countPairsHelp(root->right, dis, ans);
        for (auto &e:right) {
            if (++e > dis) continue;
            ret.emplace_back(e);
        }
        for (auto l:left) {
            for (auto r:right) {
                if (l + r <= dis) {
                    ans++;
                }
            }
        }
        return ret;
    }

    TreeNode *constructMaximumBinaryTree(vector<int> &nums) {
        if (nums.size() == 0) return nullptr;
        return constructMaximumBinaryTreeHelp(nums, 0, nums.size() - 1);
    }

    TreeNode *constructMaximumBinaryTreeHelp(vector<int> &nums, int l, int r) {
        if (l > r) return nullptr;
        int idx = findMaxIdx(nums, l, r);
        auto *root = new TreeNode(nums[idx]);
        root->left = constructMaximumBinaryTreeHelp(nums, l, idx - 1);
        root->right = constructMaximumBinaryTreeHelp(nums, idx + 1, r);
        return root;
    }

    int findMaxIdx(vector<int> &nums, int l, int r) {
        int res = l;
        for (int i = l; i <= r; ++i) {
            if (nums[res] < nums[i]) res = i;
        }
        return res;
    }

    TreeNode *lcaDeepestLeaves(TreeNode *root) {
    }

    TreeNode *lcaDeepestLeavesHelp(TreeNode *root, int &height) {
        if (!root) {
            height = -1;
            return nullptr;
        }
        int leftH;
        int rightH;
        auto left = lcaDeepestLeavesHelp(root->left, leftH);
        auto right = lcaDeepestLeavesHelp(root->right, rightH);
        height = max(leftH, rightH) + 1;
        if (leftH == rightH) return root;
        if (leftH > rightH) return left;
        return right;
    }

    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {

    }
};

#endif //LEETCODEMAC_TREE_H
