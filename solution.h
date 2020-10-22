#ifndef LEETCODEMAC_SOLUTION_H
#define LEETCODEMAC_SOLUTION_H

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    int numSubmat(vector<vector<int>> &mat) {
        int n = mat.size();
        int m = mat[0].size();
        vector<vector<int>> row(n, vector<int>(m, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (j == 0) {
                    row[i][j] = mat[i][j];
                } else if (mat[i][j]) {
                    row[i][j] = row[i][j - 1] + 1;
                } else {
                    row[i][j] = 0;
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int col = row[i][j];
                for (int k = i; k >= 0 && col; --k) {
                    col = min(col, row[k][j]);
                    ans += col;
                }
            }
        }
        return ans;
    }

    int knightDialer(int n) {
        const int m = 1000000007;
        vector<vector<unsigned long>> dp(n, vector<unsigned long>(10, 1));
        for (int i = 1; i < n; i++) {
            dp[i][1] = (dp[i - 1][6] + dp[i - 1][8]) % m;
            dp[i][2] = (dp[i - 1][7] + dp[i - 1][9]) % m;
            dp[i][3] = (dp[i - 1][4] + dp[i - 1][8]) % m;
            dp[i][4] = (dp[i - 1][3] + dp[i - 1][9] + dp[i - 1][0]) % m;
            dp[i][5] = 0;
            dp[i][6] = (dp[i - 1][1] + dp[i - 1][7] + dp[i - 1][0]) % m;
            dp[i][7] = (dp[i - 1][2] + dp[i - 1][6]) % m;
            dp[i][8] = (dp[i - 1][1] + dp[i - 1][3]) % m;
            dp[i][9] = (dp[i - 1][4] + dp[i - 1][2]) % m;
            dp[i][0] = (dp[i - 1][6] + dp[i - 1][4]) % m;
        }
        int res = 0;
        for (int i = 0; i < 10; i++) {
            res = (res + dp[n - 1][i]) % m;
        }
        return res;
    }

    int findNumberOfLIS(vector<int> &nums) {
        if (!nums.size()) {
            return 0;
        }
        int n = nums.size();
        vector<int> dp(n, 1);
        vector<int> counter(n, 1);
        int maxtmp = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = max(dp[i], dp[j] + 1);
                        counter[i] = counter[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        counter[i] += counter[j];
                    }
                }
            }
            maxtmp = max(maxtmp, dp[i]);
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == maxtmp)
                res += counter[i];
        }
        return res;
    }

    vector<int> findMode(TreeNode *root) {
        unordered_map<int, int> map{};
        vector<int> res{};
        findModeHelp(map, root);
        vector<pair<int, int>> vec(map.begin(), map.end());
        sort(vec.begin(), vec.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
            return a.second > b.second;
        });
        for (int i = 0; i < vec.size(); i++) {
            if (vec[i].second == vec[0].second) {
                res.push_back(vec[i].first);
            }
        }
        return res;
    }

    void findModeHelp(unordered_map<int, int> &map, TreeNode *root) {
        if (root == nullptr)
            return;
        findModeHelp(map, root->left);
        map[root->val]++;
        findModeHelp(map, root->right);
    }

    bool canPartitionKSubsets(vector<int> &nums, int k) {
        int maxNum = 0;
        auto sum = 0;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            maxNum = max(maxNum, nums[i]);
        }
        if (sum % k != 0 || sum / k < maxNum) {
            return false;
        }
        vector<bool> used(n, false);
        return canPartitionKSubsetsBP(nums, k, sum / k, 0, 0, used);
    }

    bool canPartitionKSubsetsBP(vector<int> &nums, int k, int target, int cur, int start, vector<bool> &used) {
        if (k == 0) return true;
        if (cur == target) {
            return canPartitionKSubsetsBP(nums, k - 1, target, 0, 0, used);
        }
        for (int i = start; i < nums.size(); i++) {
            if (!used[i] && cur + nums[i] <= target) {
                used[i] = true;
                if (canPartitionKSubsetsBP(nums, k, target, cur + nums[i], i + 1, used)) return true;
                used[i] = false;
            }
        }
        return false;
    }

    bool delst(vector<int> l, int s, int t) {
        int k = 0;
        if (l.size() == 0 || s >= t) {
            return false;
        }
        for (int i = 0; i < l.size(); ++i) {
            if (l[i] >= s && l[i] <= t) {
                k++;
            } else {
                l[i - k] = l[i];
            }
        }
        return true;
    }

    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast != slow) {
            if (fast == nullptr || fast->next == nullptr) return nullptr;
            fast = fast->next->next;
            slow = slow->next;
        }
        fast = head;
        while (fast != slow) {
            fast = fast->next;
            slow = slow->next;
        }
        return fast;
    }

    vector<int> sequentialDigits(int low, int high) {
        vector<int> ans;
        for (int i = 1; i <= 9; i++) {
            int num = i;
            for (int j = i + 1; j <= 9; ++j) {
                num = num * 10 + j;
                if (num >= low && num <= high) {
                    ans.push_back(num);
                }
            }
        }
        sort(ans.begin(), ans.end());
        return ans;
    }

    vector<vector<int>> permuteUnique(vector<int> &nums) {
        vector<vector<int>> ans;
        vector<int> path;
        vector<bool> visit(nums.size());
        sort(nums.begin(), nums.end());
        permuteUniqueBP(nums, ans, path, visit);
        return ans;
    }

    void permuteUniqueBP(vector<int> &nums, vector<vector<int>> &ans, vector<int> &path, vector<bool> &visit) {
        if (path.size() == nums.size()) {
            ans.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (visit[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !visit[i - 1]) continue;
            visit[i] = true;
            path.push_back(nums[i]);
            permuteUniqueBP(nums, ans, path, visit);
            path.pop_back();
            visit[i] = false;
        }
    }

    int getMaximumGoldDir[4][2] = {{-1, 0},
                                   {1,  0},
                                   {0,  1},
                                   {0,  -1}};

    int getMaximumGold(vector<vector<int>> &grid) {
        int ans = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] != 0) {
                    ans = max(ans, getMaximumGoldDFS(grid, i, j));
                }
            }
        }
        return ans;
    }

    int getMaximumGoldDFS(vector<vector<int>> &grid, int x, int y) {
        int tmax = 0;
        int tmp = grid[x][y];
        grid[x][y] = 0;
        for (auto d:getMaximumGoldDir) {
            int newX = x + d[0];
            int newY = y + d[1];
            if (newX < 0 || newX >= grid.size() || y < 0 || y >= grid.size() || grid[x][y] == 0) return 0;
            tmax = max(tmax, getMaximumGoldDFS(grid, newX, newY));
        }
        grid[x][y] = tmp;
        return tmax + grid[x][y];
    }

    int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 1;
        vector<int> dp(n + 1);
        dp[0] = 1;
        dp[1] = 10;
        for (int i = 2; i <= n; ++i) {
            int tmp = 9, k = 9;
            for (int j = 1; j < i; j++) {
                tmp *= k;
                k--;
            }
            dp[i] = tmp + dp[i - 1];
        }
        return dp[n];
    }

    vector<int> diffWaysToCompute(string input) {
        vector<int> res;
        int i = 0;
        int num = 0;
        while (i < input.size() && !diffWaysToComputeOP(input[i])) {
            num = num * 10 + (input[i] - '0');
            i++;
        }
        if (i == res.size()) {
            res.push_back(num);
            return res;
        }
        for (int i = 0; i < input.size(); ++i) {
            if (diffWaysToComputeOP(input[i])) {
                vector<int> left = diffWaysToCompute(input.substr(0, i));
                vector<int> right = diffWaysToCompute(input.substr(i, input.size()));
                for (int j:left) {
                    for (int k:right) {
                        res.push_back(diffWaysToComputeCal(input[i], j, k));
                    }
                }
            }
        }
        return res;
    }

    int diffWaysToComputeCal(char op, int a, int b) {
        switch (op) {
            case '+':
                return a + b;
                break;
            case '-':
                return a - b;
                break;
            case '*':
                return a * b;
                break;
            case '/':
                return a / b;
        }
    }

    bool diffWaysToComputeOP(char o) {
        return o == '+' || o == '-' || o == '*' || o == '/';
    }

    int getMinimumDifference(TreeNode *root) {
        vector<int> path;
        getMinimumDifferenceHelp(root, path);
        int ans = INT_MAX;
        for (int i = 1; i < path.size(); ++i) {
            int t = abs(path[i] - path[i - 1]);
            ans = min(ans, t);
        }
        return ans;
    }

    void getMinimumDifferenceHelp(TreeNode *root, vector<int> &ans) {
        if (root == nullptr) return;
        getMinimumDifferenceHelp(root->left, ans);
        ans.push_back(root->val);
        getMinimumDifferenceHelp(root->right, ans);
    }

    ListNode *swapPairs(ListNode *head) {
        ListNode *pre = new ListNode(0);
        pre->next = head;
        ListNode *tmp = pre;
        while (tmp->next != nullptr && tmp->next->next != nullptr) {
            ListNode *start = tmp->next;
            ListNode *end = tmp->next->next;
            tmp->next = end;
            start->next = end->next;
            end->next = start;
            tmp = start;
        }
        return pre->next;
    }

    class Node {
    public:
        int val;
        Node *left;
        Node *right;

        Node() {}

        Node(int _val) {
            val = _val;
            left = NULL;
            right = NULL;
        }

        Node(int _val, Node *_left, Node *_right) {
            val = _val;
            left = _left;
            right = _right;
        }
    };

    Node *treeToDoublyList(Node *root) {
        if (root == nullptr) return root;
        Node *pre, *head;
        treeToDoublyListHelp(root, pre, head);
        head->left = pre;
        pre->right = head;
        return head;
    }

    void treeToDoublyListHelp(Node *root, Node *pre, Node *head) {
        if (root == nullptr) return;
        treeToDoublyListHelp(root->left, pre, head);
        if (pre != nullptr) pre->right = root;
        else head = root;
        root->left = pre;
        pre = root;
        treeToDoublyListHelp(root->right, pre, head);
    }

    vector<vector<int>> kClosest(vector<vector<int>> &points, int K) {
        vector<vector<int>> res(K);
        auto cmp = [this](const vector<int> &a, const vector<int> &b) {
            return this->kClosestDist(a) < this->kClosestDist(b);
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> q(cmp);
        for (auto &a:points) {
            q.push(a);
            if (q.size() > K) q.pop();
        }
        for (int i = 0; i < K; ++i) {
            res[i] = q.top();
            q.pop();
        }
        return res;
    }

    int kClosestDist(const vector<int> &p) {
        return p[0] * p[0] + p[1] * p[1];
    }

    vector<int> smallestK(vector<int> &arr, int k) {
        vector<int> res(k);
        priority_queue<int> q;
        for (int a:arr) {
            q.push(a);
            if (q.size() > k) q.pop();
        }
        while (!q.empty()) {
            res.push_back(q.top());
            q.pop();
        }
        return res;
    }

    bool searchMatrix(vector<vector<int>> &matrix, int target) {
        if (matrix.size() == 0 || matrix[0].size() == 0) return false;
        int row = 0, col = matrix[0].size() - 1;
        while (row != matrix.size() && col != -1) {
            if (matrix[row][col] > target) {
                col -= 1;
            } else if (matrix[row][col] < target) {
                row += 1;
            } else {
                return true;
            }
        }
        return false;
    }

    ListNode *mergeKLists(vector<ListNode *> &lists) {
        if (lists.size() == 0) return nullptr;
        return mergeKListsHelp(lists, 0, lists.size() - 1);
    }

    ListNode *mergeKListsHelp(vector<ListNode *> &lists, int left, int right) {
        if (left >= right) return lists[left];
        int mid = (left + right) / 2;
        ListNode *l1 = mergeKListsHelp(lists, left, mid);
        ListNode *l2 = mergeKListsHelp(lists, mid + 1, right);
        return mergeKListsTwo(l1, l2);
    }

    ListNode *mergeKListsTwo(ListNode *l1, ListNode *l2) {
        auto *res = new ListNode(0);
        ListNode *tmp = res;
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                res->next = l1;
                l1 = l1->next;
            } else {
                res->next = l2;
                l2 = l2->next;
            }
            res = res->next;
        }
        if (l1 == nullptr) res->next = l2;
        if (l2 == nullptr) res->next = l1;
        return tmp->next;
    }

    int maxCoins(vector<int> &nums) {
        if (nums.size() == 0) return 0;
        vector<int> tmp(nums.size() + 2);
        tmp[0] = 1;
        for (int i = 1; i <= nums.size(); ++i) {
            tmp[i] = nums[i - 1];
        }
        tmp[nums.size() + 1] = 1;
        vector<vector<int>> cache(tmp.size(), vector<int>(tmp.size()));
        return maxCoinsDP(tmp, 0, tmp.size() - 1, cache);
    }

    int maxCoinsDP(vector<int> &nums, int begin, int end, vector<vector<int>> &cache) {
        if (begin == end - 1) return 0;
        if (cache[begin][end] != 0) return cache[begin][end];
        int fmax = 0;
        for (int i = begin + 1; i < end; i++) {
            int tmp = maxCoinsDP(nums, begin, i, cache) + maxCoinsDP(nums, i, end, cache) +
                      nums[i] * nums[begin] * nums[end];
            fmax = max(fmax, tmp);
        }
        cache[begin][end] = fmax;
        return fmax;
    }

    vector<int> sortedSquares(vector<int> &A) {
        int n = A.size();
        vector<int> ans(n);
        for (int i = 0, j = n - 1, pos = n - 1; i <= j;) {
            if (A[i] * A[i] > A[j] * A[j]) {
                ans[pos] = A[i] * A[i];
                ++i;
            } else {
                ans[pos] = A[j] * A[j];
                --j;
            }
            --pos;
        }
        return ans;
    }

    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<int> path;
        vector<bool> col(n);
        vector<bool> main(2 * n - 1);
        vector<bool> sub(2 * n - 1);
        solveNQueensBP(res, n, path, col, main, sub, 0);
        return res;
    }

    void solveNQueensBP(vector<vector<string>> &res, int n, vector<int> &path, vector<bool> &col, vector<bool> &main,
                        vector<bool> &sub, int row) {
        if (row == n) {
            auto board = NQconvert(path, n);
            res.emplace_back(board);
            return;
        }
        for (int j = 0; j < n; ++j) {
            if (!col[j] && !main[row + j] && !sub[row - j + n - 1]) {
                path.push_back(j);
                col[j] = true;
                main[row + j] = true;
                sub[row - j + n - 1] = true;
                solveNQueensBP(res, n, path, col, main, sub, row + 1);
                sub[row - j + n - 1] = false;
                main[row + j] = false;
                col[j] = false;
                path.pop_back();
            }
        }
    }

    vector<string> NQconvert(vector<int> &path, int n) {
        vector<string> res;
        for (auto i:path) {
            string row(n, '.');
            row[i] = 'Q';
            res.emplace_back(row);
        }
        return res;
    }

    int totalNQueens(int n) {
        int res;
        vector<int> path;
        vector<bool> col(n);
        vector<bool> main(2 * n - 1);
        vector<bool> sub(2 * n - 1);
        solveNQueensBP2(res, n, path, col, main, sub, 0);
        return res;
    }

    void solveNQueensBP2(int &res, int n, vector<int> &path, vector<bool> &col, vector<bool> &main,
                         vector<bool> &sub, int row) {
        if (row == n) {
            res++;
            return;
        }
        for (int j = 0; j < n; ++j) {
            if (!col[j] && !main[row + j] && !sub[row - j + n - 1]) {
                path.push_back(j);
                col[j] = true;
                main[row + j] = true;
                sub[row - j + n - 1] = true;
                solveNQueensBP2(res, n, path, col, main, sub, row + 1);
                sub[row - j + n - 1] = false;
                main[row + j] = false;
                col[j] = false;
                path.pop_back();
            }
        }
    }

    static constexpr int TARGET = 24;
    static constexpr double EPSILON = 1e-6;
    static constexpr int ADD = 0, MUL = 1, SUB = 2, DIV = 3;

    bool judgePoint24(vector<int> &nums) {
        vector<double> l;
        for (int n:nums) {
            l.emplace_back(static_cast<double>(n));
        }
        return judgePoint24Help(l);
    }

    bool judgePoint24Help(vector<double> &l) {
        if (l.empty()) {
            return false;
        }
        if (l.size() == 1) {
            return fabs(l[0] - TARGET) < EPSILON;
        }
        int size = l.size();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; ++j) {
                if (i != j) {
                    vector<double> list2 = vector<double>();
                    for (int k = 0; k < size; ++k) {
                        if (k != i && k != j) {
                            list2.emplace_back(l[k]);
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (k < 2 && i > j) {
                            continue;
                        }
                        if (k == ADD) {
                            list2.emplace_back(l[i] + l[j]);
                        } else if (k == MUL) {
                            list2.emplace_back(l[i] * l[j]);
                        } else if (k == SUB) {
                            list2.emplace_back(l[i] - l[j]);
                        } else if (k == DIV) {
                            if (fabs(l[j]) < EPSILON) {
                                continue;
                            }
                            list2.emplace_back(l[i] / l[j]);
                        }
                        if (judgePoint24Help(list2)) {
                            return true;
                        }
                        list2.pop_back();
                    }
                }
            }
        }
        return false;
    }

    bool rowUsed[9][10]{false};
    bool colUsed[9][10]{false};
    bool boxUsed[3][3][10]{false};

    void solveSudoku(vector<vector<char>> &board) {
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[0].size(); ++j) {
                int num = board[i][j] - '0';
                if (num >= 1 && num <= 9) {
                    rowUsed[i][num] = true;
                    colUsed[j][num] = true;
                    boxUsed[i / 3][j / 3][num] = true;
                }
            }
        }
        solveSudokuBP(board, 0, 0);
    }

    bool solveSudokuBP(vector<vector<char>> &board, int row, int col) {
        if (col == board[0].size()) {
            col = 0;
            row++;
            if (row == board.size()) {
                return true;
            }
        }
        if (board[row][col] == '.') {
            for (int i = 1; i <= 9; ++i) {
                if (!rowUsed[row][i] && !colUsed[col][i] && !boxUsed[row / 3][col / 3][i]) {
                    rowUsed[row][i] = true;
                    colUsed[col][i] = true;
                    boxUsed[row / 3][col / 3][i] = true;
                    board[row][col] = (char) (i + '0');
                    if (solveSudokuBP(board, row, col + 1)) {
                        return true;
                    }
                    rowUsed[row][i] = false;
                    colUsed[col][i] = false;
                    boxUsed[row / 3][col / 3][i] = false;
                    board[row][col] = '.';
                }
            }
        } else {
            return solveSudokuBP(board, row, col + 1);
        }
        return false;
    }

    ListNode *removeNthFromEnd(ListNode *head, int n) {
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        ListNode *p = dummy;
        ListNode *q = dummy;
        for (int i = 0; i < n + 1; i++)
            q = q->next;
        while (q) {
            p = p->next;
            q = q->next;
        }
        ListNode *del = p->next;
        p->next = p->next->next;
        delete del;

        ListNode *ret = dummy->next;
        delete dummy;
        return ret;
    }

    ListNode *oddEvenList(ListNode *head) {
        if (head == nullptr) return nullptr;
        ListNode *odd = head, *even = head->next, *evenHead = even;
        while (even != nullptr && even->next != nullptr) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = evenHead;
        return head;
    }

    bool isValidSudoku(vector<vector<char>> &board) {
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[0].size(); ++j) {
                int num = board[i][j] - '0';
                if (num >= 1 && num <= 9) {
                    if (rowUsed[i][num] || colUsed[j][num] || boxUsed[i / 3][j / 3][num])
                        return false;
                    rowUsed[i][num] = true;
                    colUsed[j][num] = true;
                    boxUsed[i / 3][j / 3][num] = true;
                }
            }
        }
        return true;
    }

    void setZeroes(vector<vector<int>> &matrix) {
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[0].size(); ++j) {
                if (matrix[i][j] == 0) {
                    for (int k = 0; k < matrix[0].size(); ++k) {
                        if (matrix[i][k] != 0) matrix[i][k] = -10000;
                    }
                    for (int k = 0; k < matrix.size(); ++k) {
                        if (matrix[i][k] != 0) matrix[k][j] = -10000;
                    }
                }
            }
        }
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[0].size(); ++j) {
                if (matrix[i][j] == -10000) matrix[i][j] = 0;
            }
        }
    }

    vector<vector<string>> groupAnagrams(vector<string> &strs) {
        vector<vector<string>> res;
        map<string, int> m;
        int idx = 0;
        for (int i = 0; i < strs.size(); ++i) {
            string tmp = strs[i];
            sort(tmp.begin(), tmp.end());
            if (m.find(tmp) == m.end()) {
                m[tmp] = idx++;
                vector<string> vec;
                vec.emplace_back(strs[i]);
                res.emplace_back(vec);
            } else {
                res[m[tmp]].emplace_back(strs[i]);
            }
        }
        return res;
    }

    ListNode *ReverseList(ListNode *pHead) {
        ListNode *dummy = new ListNode(0);
        ListNode *cur = pHead;
        ListNode *nex = nullptr;
        while (cur != nullptr) {
            nex = cur->next;
            cur->next = dummy->next;
            dummy->next = cur;
            cur = nex;
        }
        return dummy->next;
    }

    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }

    vector<vector<int> > threeOrders(TreeNode *root) {
        vector<int> pre;
        vector<int> in;
        vector<int> post;
        threeOrdersPre(root, pre);
        threeOrdersIn(root, in);
        threeOrdersPost(root, post);
        vector<vector<int>> res{pre, in, post};
        return res;
    }

    void threeOrdersPre(TreeNode *root, vector<int> &pre) {
        if (root == nullptr) return;
        pre.emplace_back(root->val);
        threeOrdersPre(root->left, pre);
        threeOrdersPre(root->right, pre);
    }

    void threeOrdersIn(TreeNode *root, vector<int> &pre) {
        if (root == nullptr) return;
        threeOrdersIn(root->left, pre);
        pre.emplace_back(root->val);
        threeOrdersIn(root->right, pre);
    }

    void threeOrdersPost(TreeNode *root, vector<int> &pre) {
        if (root == nullptr) return;
        threeOrdersPost(root->left, pre);
        threeOrdersPost(root->right, pre);
        pre.emplace_back(root->val);
    }

    bool backspaceCompare(string S, string T) {
        return backspaceCompareHelp(S) == backspaceCompareHelp(T);
    }

    string backspaceCompareHelp(string s) {
        string res;
        for (char ch:s) {
            if (ch != '#') {
                res.push_back(ch);
            } else if (!res.empty()) {
                res.pop_back();
            }
        }
        return res;
    }

    void reorderList(ListNode *head) {
        if (head == nullptr) return;
        ListNode *slow = head, *fast = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode *newHead = slow->next;
        slow->next = nullptr;
        newHead = reverseList(newHead);
        while (newHead != nullptr) {
            ListNode *tmp = newHead->next;
            newHead->next = head->next;
            head->next = newHead;
            head = newHead->next;
            newHead = tmp;
        }
    }

    ListNode *reverseList(ListNode *head) {
        ListNode *res = new ListNode(0);
        ListNode *cur = head, *nex = nullptr;
        while (cur != nullptr) {
            nex = cur->next;
            cur->next = res->next;
            res->next = cur;
            cur = nex;
        }
        return res->next;
    }

    int minDep(TreeNode *root) {
        if (root == nullptr) return 0;
        int left = minDep(root->left);
        int right = minDep(root->right);
        if (left * right != 0) {
            return (left > right ? right : left) + 1;
        } else {
            return (left > right ? left : right) + 1;
        }
    }

    int evalRPN(vector<string> &tokens) {
        stack<int> stk;
        for (const auto &t:tokens) {
            if (!evalRPNisOp(t)) {
                stk.push(stoi(t));
            } else {
                if (t == "+") {
                    int a1 = stk.top();
                    stk.pop();
                    int a2 = stk.top();
                    stk.pop();
                    stk.push(a1 + a2);
                } else if (t == "-") {
                    int a1 = stk.top();
                    stk.pop();
                    int a2 = stk.top();
                    stk.pop();
                    stk.push(a1 - a2);
                } else if (t == "*") {
                    int a1 = stk.top();
                    stk.pop();
                    int a2 = stk.top();
                    stk.pop();
                    stk.push(a1 * a2);
                } else if (t == "/") {
                    int a1 = stk.top();
                    stk.pop();
                    int a2 = stk.top();
                    stk.pop();
                    stk.push(a1 / a2);
                }
            }
        }
        return stk.top();
    }

    bool evalRPNisOp(string c) {
        return c == "+" || c == "+" || c == "*" || c == "/";
    }

    struct Point {
        int x;
        int y;
    };

    int maxPoints(vector<Point> &points) {
        int size = points.size();
        if (size <= 2) return size;
        int res = 0;
        for (int i = 0; i < size; ++i) {
            int d = 1;
            map<float, int> tmp;
            for (int j = i + 1; j < size; ++j) {
                if (points[i].x == points[j].x) {
                    if (points[i].y == points[j].y) {
                        d++;
                    } else {
                        tmp[INT_MAX]++;
                    }
                } else {
                    float k = (points[i].y - points[j].y) * 1.0 / (points[i].x - points[j].x);
                    tmp[k]++;
                }
            }
            res = max(res, d);
            for (auto it:tmp) {
                res = max(res, it.second + d);
            }
        }
        return res;
    }

    ListNode *sortList(ListNode *head) {
        if (head == nullptr || head->next == nullptr) return head;
        ListNode *slow = head, *fast = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode *newHead = slow->next;
        slow->next = nullptr;
        ListNode *left = sortList(head);
        ListNode *right = sortList(newHead);
        return mergeList(left, right);
    }

    ListNode *mergeList(ListNode *a, ListNode *b) {
        ListNode *res = new ListNode(0);
        ListNode *tmp = res;
        while (a != nullptr && b != nullptr) {
            if (a->val < b->val) {
                tmp->next = new ListNode(a->val);
                a = a->next;
            } else {
                tmp->next = new ListNode(b->val);
                b = b->next;
            }
            tmp = tmp->next;
        }
        tmp->next = (a == nullptr) ? b : a;
        return res->next;
    }

    ListNode *insertionSortList(ListNode *head) {
        if (head == nullptr || head->next == nullptr) return head;
        ListNode *tmp = new ListNode(0);
        ListNode *p;
        ListNode *q;
        ListNode *t;
        while (head) {
            p = tmp;
            q = p->next;
            t = head;
            head = head->next;
            while (q && q->val < t->val) {
                p = p->next;
                q = q->next;
            }
            t->next = q;
            p->next = t;
        }
        return tmp->next;
    }

    bool isLongPressedName(string name, string typed) {
        int i = 0, j = 0;
        while (j < typed.length()) {
            if (i < name.length() && name[i] == typed[j]) {
                i++;
                j++;
            } else if (j > 0 && typed[j] == typed[j - 1]) {
                j++;
            } else {
                return false;
            }
        }
        return i == name.length();
    }

    void reorderList2(ListNode *head) {
        if (head == nullptr) return;
        ListNode *fast = head, *slow = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode *newHead = slow->next;
        slow->next = nullptr;
        newHead = reverseList(newHead);
        while (newHead) {
            ListNode *tmp = newHead->next;
            newHead->next = head->next;
            head->next = newHead;
            head = newHead->next;
            newHead = tmp;
        }
    }

    ListNode *detectCycle2(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (true) {
            if (fast == nullptr || fast->next == nullptr) {
                return nullptr;
            }
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) break;
        }
        fast = head;
        while (fast != slow) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }

    bool wordBreak(string s, unordered_set<string> &dict) {
        int len = s.length();
        vector<bool> dp(len + 1);
        dp[0] = true;
        for (int i = 1; i <= len; ++i) {
            for (int j = i - 1; j >= 0; --j) {
                if(dp[j]&&dict.count(s.substr(j,i-j))){
                    dp[i]=true;
                    break;
                }
            }
        }
        return dp[len];
    }
};

#endif //LEETCODEMAC_SOLUTION_H
