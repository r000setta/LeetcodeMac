#ifndef LEETCODEMAC_OFFER_H
#define LEETCODEMAC_OFFER_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class Offer {
public:
    int findRepeatNumber(vector<int> &nums) {
        unordered_set<int> st;
        for (int n:nums) {
            if (st.count(n)) {
                return n;
            } else {
                st.insert(n);
            }
        }
        return -1;
    }

    bool findNumberIn2DArray(vector<vector<int>> &matrix, int target) {
        int i = 0, j = matrix[0].size() - 1;
        while (i < matrix.size() && j >= 0) {
            if (target == matrix[i][j]) {
                return true;
            } else if (target > matrix[i][j]) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }

    string replaceSpace(string s) {
        string res = "";
        for (char c:s) {
            if (c != ' ') {
                res += c;
            } else {
                res += "%20";
            }
        }
        return res;
    }

    vector<int> reversePrint(ListNode *head) {
        vector<int> res;
        while (head != nullptr) {
            res.emplace_back(head->val);
        }
        reverse(res.begin(), res.end());
        return res;
    }

    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {

    }

    int fib(int n) {
        vector<size_t> dp(n + 1);
        size_t mod = 1000000007;
        if (n == 0) return 0;
        if (n == 1) return 1;
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % mod;
        }
        return dp[n];
    }

    int numWays(int n) {
        int mod = 1000000007;
        vector<int> dp(n + 1);
        if (n == 0) return 1;
        if (n == 1) return 1;
        if (n == 2) return 2;
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; ++i) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % mod;
        }
        return dp[n];
    }

    int minArray(vector<int> &numbers) {
        int l = 0, r = numbers.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (numbers[mid] > numbers[r]) {
                l = mid + 1;
            } else if (numbers[mid] < numbers[r]) {
                r = mid;
            } else {
                while (numbers[r] == numbers[mid]) {
                    r--;
                }
            }
        }
        return l;
    }

    bool exist(vector<vector<char>> &board, string word) {
        int m = board.size(), n = board[0].size();
        vector<vector<bool>> vis(m, vector<bool>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == word[0]) {
                    vis[i][j] = true;
                    if (existDFS(board, word, 0, vis, i, j)) {
                        return true;
                    }
                    vis[i][j] = false;
                }
            }
        }
        return false;
    }

    int dir[4][2] = {{-1, 0},
                     {1,  0},
                     {0,  -1},
                     {0,  1}};

    bool existDFS(vector<vector<char>> &board, string word, int i, vector<vector<bool>> &vis, int x, int y) {
        if (i == word.size() - 1) return true;
        for (int i = 0; i < 4; ++i) {
            int nx = x + dir[i][0];
            int ny = y + dir[i][1];
            if (nx >= 0 && nx < board.size() && ny >= 0 && ny < board[0].size() && !vis[nx][ny] &&
                board[nx][ny] == word[i + 1]) {
                vis[nx][ny] = true;
                existDFS(board, word, i + 1, vis, nx, ny);
                vis[nx][ny] = false;
            }
        }
        return false;
    }

    int movingCount(int m, int n, int k) {

    }

    int cuttingRope(int n) {
        vector<int> dp(n + 1);
        dp[2] = 1;
        for (int i = 3; i <= n; ++i) {
            for (int k = 2; k <= i; ++k) {
                dp[i] = max(dp[i], max(k * (i - k), k * dp[i - k]));
            }
        }
        return dp[n];
    }

    int hammingWeight(uint32_t n) {
        int res = 0;
        while (n != 0) {
            n = n & (n - 1);
            res++;
        }
        return res;
    }

    double myPow(double x, int n) {
        double res = 1;
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        while (n > 0) {
            if ((n & 1) == 1) res *= x;
            x *= x;
            n >>= 1;
        }
        return res;
    }

    vector<int> printNumbers(int n) {
        int tar = pow(10, n);
        vector<int> res(tar - 1);
        for (int i = 0; i < tar - 1; ++i) {
            res[i] = i + 1;
        }
        return res;
    }

    ListNode *deleteNode(ListNode *head, int val) {
        ListNode *dummy = new ListNode(0);
        ListNode *res = dummy;
        dummy->next = head;
        while (dummy->next != nullptr && dummy->next->val != val) {
            dummy = dummy->next;
        }
        dummy->next = dummy->next->next;
        return res->next;
    }

    bool isNumber(string s) {

    }

    ListNode *getKthFromEnd(ListNode *head, int k) {
        ListNode *fast = head, *slow = head;
        for (int i = 0; i < k; ++i) {
            fast = fast->next;
        }
        while (fast->next != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        return slow->next;
    }

    vector<int> exchange(vector<int> &nums) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            while (l < r && nums[l] % 2 == 1) l++;
            while (l < r && nums[r] % 2 == 0) r--;
            swap(nums[l], nums[r]);
        }
        return nums;
    }

    ListNode *reverseList(ListNode *head) {
        ListNode *res = new ListNode(0);
        ListNode *cur = res;
        while (head != nullptr) {
            ListNode *t = head->next;
            head->next = cur->next;
            cur->next = head;
            head = t;
        }
        return cur->next;
    }

    bool isSubStructure(TreeNode *A, TreeNode *B) {
        if (A == nullptr || B == nullptr) return false;
        return isSameTree(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }

    bool isSameTree(TreeNode *A, TreeNode *B) {
        if (A == nullptr && B == nullptr) return true;
        if (A == nullptr && B != nullptr) return false;
        if (A != nullptr && B == nullptr) return true;
        return A->val == B->val && isSameTree(A->right, B->right) || isSameTree(A->right, B->right);
    }

    TreeNode *mirrorTree(TreeNode *root) {
        if (root == nullptr) return root;
        TreeNode *l = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(l);
        return root;
    }

    bool isSymmetric(TreeNode *root) {
        if (root == nullptr) return true;
        return isSymmetricHelp(root->left, root->right);
    }

    bool isSymmetricHelp(TreeNode *l, TreeNode *r) {
        if (l == nullptr && r == nullptr) return true;
        if (l == nullptr || r == nullptr) return false;
        return l->val == r->val && isSymmetricHelp(l->left, r->right) && isSymmetricHelp(l->right, r->left);
    }

    vector<int> spiralOrder(vector<vector<int>> &matrix) {
        int l = 0, r = matrix[0].size() - 1, t = 0, b = matrix.size() - 1;
        vector<int> res((r + 1) * (b + 1));
        int k = 0;
        while (true) {
            for (int i = l; i <= r; ++i) res[k++] = matrix[t][i];
            if (++t > b) break;
            for (int i = t; i <= b; ++i) res[k++] = matrix[i][r];
            if (l > --r) break;
            for (int i = r; i >= l; --i) res[k++] = matrix[b][i];
            if (t > --b) break;
            for (int i = b; i >= t; --i) res[k++] = matrix[i][l];
            if (++l > r) break;
        }
        return res;
    }

    bool validateStackSequences(vector<int> &pushed, vector<int> &popped) {
        stack<int> st;
        int k = 0;
        for (int i = 0; i < pushed.size(); ++i) {
            st.push(pushed[i]);
            while (!st.empty() && st.top() == popped[k]) {
                st.pop();
                k++;
            }
        }
        return st.empty();
    }

    vector<int> levelOrder(TreeNode *root) {
        vector<int> res;
        if (root == nullptr) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int n = q.size();
            for (int i = 0; i < n; ++i) {
                TreeNode *cur = q.front();
                q.pop();
                res.emplace_back(cur->val);
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
        }
        return res;
    }

    vector<vector<int>> levelOrder2(TreeNode *root) {
        vector<vector<int>> res;
        if (root == nullptr) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int n = q.size();
            vector<int> v(n);
            for (int i = 0; i < n; ++i) {
                TreeNode *cur = q.front();
                q.pop();
                v[i] = cur->val;
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res.emplace_back(std::move(v));
        }
        return res;
    }

    vector<vector<int>> levelOrder3(TreeNode *root) {
        vector<vector<int>> res;
        int t = 0;
        if (root == nullptr) return res;
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            int n = q.size();
            vector<int> v(n);
            for (int i = 0; i < n; ++i) {
                TreeNode *cur = q.front();
                q.pop();
                v[i] = cur->val;
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            if (t % 2 == 1) {
                reverse(v.begin(),v.end());
                res.emplace_back(std::move(v));
            }else{
                res.emplace_back(std::move(v));
            }
        }
        return res;
    }



    class MinStack {
    public:
        stack<int> st = stack<int>();
        stack<int> st2 = stack<int>();

        /** initialize your data structure here. */
        MinStack() {

        }

        void push(int x) {
            st.push(x);
            if (st2.empty()) {
                st2.push(x);
            } else {
                if (st2.top() >= x) {
                    st2.push(x);
                }
            }
        }

        void pop() {
            int t = st.top();
            if (t == st2.top()) {
                st2.pop();
            }
            st.pop();
        }

        int top() {
            return st.top();
        }

        int min() {
            return st2.top();
        }
    };


    class CQueue {
    public:
        stack<int> s1 = stack<int>();
        stack<int> s2 = stack<int>();

        CQueue() {

        }

        void appendTail(int value) {
            s1.push(value);
        }

        int deleteHead() {
            if (s2.empty()) {
                while (!s1.empty()) {
                    int t = s1.top();
                    s1.pop();
                    s2.push(t);
                }
            }
            if (s2.empty()) {
                return -1;
            }
            int res = s2.top();
            s2.pop();
            return res;
        }
    };
};


#endif //LEETCODEMAC_OFFER_H