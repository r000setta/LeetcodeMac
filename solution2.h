#ifndef LEETCODEMAC_SOLUTION2_H
#define LEETCODEMAC_SOLUTION2_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution2 {
public:
    bool findNumberIn2DArray(vector<vector<int>> &matrix, int target) {
        int x = matrix.size() - 1, y = 0;
        while (x >= 0 && y < matrix[0].size()) {
            if (matrix[x][y] == target) {
                return true;
            } else if (target < matrix[x][y]) {
                x--;
            } else {
                y++;
            }
        }
        return false;
    }

    vector<int> findSquare(vector<vector<int>> &matrix) {

    }

    bool validMountainArray(vector<int> &A) {
        int n = A.size(), i = 0;

        while (i + 1 < n && A[i] < A[i + 1]) {
            i++;
        }
        if (i == 0 || i == n - 1) return false;
        while (i + 1 < n && A[i] > A[i + 1]) {
            i++;
        }
        return i == n - 1;
    }

    int numSubmat(vector<vector<int>> &mat) {
        vector<vector<int>> dp(mat.size(), vector<int>(mat[0].size()));
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                if (j == 0) {
                    dp[i][j] = mat[i][j];
                } else if (mat[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i][j - 1] + 1;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < mat.size(); ++i) {
            for (int j = 0; j < mat[0].size(); ++j) {
                int val = dp[i][j];
                for (int k = i; k >= 0 && val; --k) {
                    val = min(val, dp[k][j]);
                    res += val;
                }
            }
        }
        return res;
    }

    int existDir[4][2] = {{1,  0},
                          {-1, 0},
                          {0,  1},
                          {0,  -1}};

    bool exist(vector<vector<char>> &board, string word) {
        bool res = false;
        vector<vector<bool>> vis(board.size(), vector<bool>(board[0].size()));
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); ++j) {
                if (board[i][j] == word[0]) {
                    vis.assign(board.size(), vector<bool>(board[0].size(), false));
                    res = res || existDFS(board, word, 0, i, j, vis);
                    if (res) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool existDFS(vector<vector<char>> &board, string &word, int cur,
                  int x, int y, vector<vector<bool>> &vis) {
        if (cur == word.size() - 1) return true;
        vis[x][y] = true;
        bool res = false;
        for (const auto &dir:existDir) {
            int tx = x + dir[0];
            int ty = y + dir[1];
            if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() || vis[tx][ty] == true ||
                word[cur + 1] != board[tx][ty]) {
                continue;
            } else {
                res = res || existDFS(board, word, cur + 1, tx, ty, vis);
                if (res) {
                    return true;
                }
            }
        }
        return false;
    }

    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int over = 0;
        if (!(E >= C || G <= A || F >= D || H <= B)) {
            int x1 = max(A, E);
            int y1 = max(B, F);
            int x2 = min(C, G);
            int y2 = min(D, H);
            over = (x2 - x1) * (y2 - y1);
        }

        return (C - A) * (D - B) + ((G - E) * (H - F) - over);
    }

    int maximalRectangle(vector<vector<char>> &matrix) {
        int res = 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == 0) {
                    dp[i][j] = matrix[i][j] - '0';
                } else if (matrix[i][j] == '1') {
                    dp[i][j] = dp[i][j - 1] + 1;
                } else {
                    dp[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int val = dp[i][j];
                for (int k = i; k >= 0 && val; --k) {
                    val = min(val, dp[k][j]);
                    res = max(res, val * (i - k + 1));
                }
            }
        }
        return res;
    }

    int maximalSquare(vector<vector<char>> &matrix) {
        if (matrix.size() == 0) return 0;
        int res = 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) dp[i][j] = matrix[i][j] - '0';
                else if (matrix[i][j] == '0') dp[i][j] = 0;
                else {
                    dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
                res = max(res, dp[i][j]);
            }
        }
        return res * res;
    }

    int countSquares(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0) {
                    dp[i][j] = matrix[i][j];
                } else if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = min(min(dp[i][j - 1], dp[i - 1][j]),
                                   dp[i - 1][j - 1]) + 1;
                }
                res += dp[i][j];
            }
        }
        return res;
    }


    vector<int> spiralOrder(vector<vector<int>> &matrix) {
        if (matrix.size() == 0) return {};
        int l = 0, r = matrix[0].size() - 1, t = 0, b = matrix.size() - 1, x = 0;
        vector<int> res((r + 1) * (b + 1));
        while (true) {
            for (int i = l; i <= r; ++i) res[x++] = matrix[t][i];
            if (++t > b) break;
            for (int i = t; i <= b; i++) res[x++] = matrix[i][r];
            if (l > --r) break;
            for (int i = r; i >= l; i--) res[x++] = matrix[b][i];
            if (t > --b) break;
            for (int i = b; i >= t; i--) res[x++] = matrix[i][l];
            if (++l > r) break;
        }
        return res;
    }

    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n));
        int l = 0, t = 0, r = n - 1, b = n - 1, k = 1;
        while (true) {
            for (int i = l; i <= r; ++i) {
                res[t][i] = k;
                k++;
            }
            if (++t > b) break;
            for (int i = t; i <= b; i++) {
                res[i][r] = k;
                k++;
            }
            if (l > --r) break;
            for (int i = r; i >= l; --i) {
                res[b][i] = k;
                k++;
            }
            if (--b < t) break;
            for (int i = b; i >= t; --i) {
                res[i][l] = k;
                k++;
            }
            if (++l > r) break;
        }
        return res;
    }

    void setZeroes(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<bool> rows(m);
        vector<bool> cols(n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    if (!rows[i]) {
                        rows[i] = true;
                    }
                    if (!cols[j]) {
                        cols[j] = true;
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            if (rows[i]) {
                for (int j = 0; j < n; ++j) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (cols[i]) {
                for (int j = 0; j < m; ++j) {
                    matrix[j][i] = 0;
                }
            }
        }
    }

    vector<vector<int>> transpose(vector<vector<int>> &A) {
        int m = A.size(), n = A[0].size();
        vector<vector<int>> res(n, vector<int>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res[j][i] = A[i][j];
            }
        }
        return res;
    }

    vector<vector<int>> matrixBlockSum(vector<vector<int>> &mat, int K) {
        int m = mat.size(), n = mat[0].size();
        vector<vector<int>> res(m, vector<int>(n));
        vector<vector<int>> pre(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + mat[i - 1][j - 1];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res[i][j] = matrixBlockSumGet(pre, m, n, i + K + 1, j + K + 1)
                            - matrixBlockSumGet(pre, m, n, i + K + 1, j - K) -
                            matrixBlockSumGet(pre, m, n, i - K, j + K + 1)
                            + matrixBlockSumGet(pre, m, n, i - K, j - K);
            }
        }
        return res;
    }

    int matrixBlockSumGet(vector<vector<int>> &pre, int m, int n, int i, int j) {
        int x = max(min(i, m), 0);
        int y = max(min(j, n), 0);
        return pre[x][y];
    }

    int numMagicSquaresInside(vector<vector<int>> &grid) {
        if (grid.size() < 3 || grid[0].size() < 3) return 0;
        int res = 0;
        for (int i = 0; i < grid.size() - 2; ++i) {
            for (int j = 0; j < grid[0].size() - 2; ++j) {
                if (numMagicSquaresInsideCheck(grid, i, j)) {
                    res++;
                }
            }
        }
        return res;
    }

    bool numMagicSquaresInsideCheck(vector<vector<int>> &grid, int x, int y) {
        if (grid[x + 1][y + 1] != 5) return false;
        unordered_set<int> tmp;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int num = grid[x + i][y + j];
                if (num > 9 || num < 1) return false;
                tmp.insert(num);
            }
        }
        if (tmp.size() != 9) return false;
        int tar = grid[x][y] + grid[x + 1][y + 1] + grid[x + 2][y + 2];
        for (int i = 0; i < 3; ++i) {
            int res = 0;
            for (int j = 0; j < 3; ++j) {
                int cur = grid[x + i][y + j];
                res += cur;
            }
            if (res != tar) return false;
        }
        for (int i = 0; i < 3; ++i) {
            int res = 0;
            for (int j = 0; j < 3; ++j) {
                res += grid[x + j][y + i];
            }
            if (res != tar) return false;
        }

        return grid[x + 2][y] + grid[x + 1][y + 1] + grid[x][y + 2] == tar;
    }

    vector<int> luckyNumbers(vector<vector<int>> &matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<int> rmin(m, INT_MAX);
        vector<int> cmax(n, 0);
        vector<int> ans;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                rmin[i] = min(rmin[i], matrix[i][j]);
                cmax[j] = max(cmax[j], matrix[i][j]);
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == rmin[i] && matrix[i][j] == cmax[j]) {
                    ans.push_back(matrix[i][j]);
                }
            }
        }
        return ans;
    }

    vector<vector<int>> diagonalSort(vector<vector<int>> &mat) {
        int n = mat.size(), m = mat[0].size();
        unordered_map<int, vector<int>> vs;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                vs[i - j].emplace_back(mat[i][j]);
            }
        }
        for (auto &v:vs) sort(v.second.begin(), v.second.end());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                mat[i][j] = vs[i - j].back();
                vs[i - j].pop_back();
            }
        }
        return mat;
    }

    int diagonalSum(vector<vector<int>> &mat) {
        int m = mat.size();
        int res = 0;
        for (int i = 0; i < m; ++i) {
            res += mat[i][i];
            res += mat[i][m - i - 1];
        }
        if (m % 2) {
            return res - mat[m / 2][m / 2];
        } else {
            return res;
        }
    }

    int longestIncreasingPath(vector<vector<int>> &matrix) {
        if (matrix.size() == 0) return 0;
        int res = 0;
        vector<vector<int>> memo(matrix.size(), vector<int>(matrix[0].size()));
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[0].size(); ++j) {
                res = max(res, longestIncreasingPathDFS(matrix, memo, i, j));
            }
        }
        return res;
    }

    int longestIncreasingPathDFS(vector<vector<int>> &matrix, vector<vector<int>> &memo, int x, int y) {
        if (memo[x][y] != 0) return memo[x][y];
        memo[x][y]++;
        if (x >= 1 && matrix[x][y] < matrix[x - 1][y])
            memo[x][y] = max(memo[x][y], longestIncreasingPathDFS(matrix, memo, x - 1, y) + 1);
        if (x + 1 < matrix.size() && matrix[x][y] < matrix[x + 1][y])
            memo[x][y] = max(memo[x][y], longestIncreasingPathDFS(matrix, memo, x + 1, y) + 1);
        if (y - 1 >= 0 && matrix[x][y] < matrix[x][y - 1])
            memo[x][y] = max(memo[x][y], longestIncreasingPathDFS(matrix, memo, x, y - 1) + 1);
        if (y + 1 < matrix[0].size() && matrix[x][y] < matrix[x][y + 1])
            memo[x][y] = max(memo[x][y], longestIncreasingPathDFS(matrix, memo, x, y + 1) + 1);
        return memo[x][y];
    }

    int maxProductPath(vector<vector<int>> &grid) {
        int MOD = 1e9 + 7;
        int m = grid.size(), n = grid[0].size();
        vector<vector<vector<long long>>> dp(m, vector<vector<long long>>(n, vector<long long>(2)));
        //0:max,1:min
        dp[0][0][0] = dp[0][0][1] = grid[0][0];
        for (int i = 1; i < m; ++i) {
            dp[i][0][0] = dp[i][0][1] = dp[i - 1][0][0] * grid[i][0];
        }
        for (int i = 1; i < n; ++i) {
            dp[0][i][0] = dp[0][i][1] = dp[0][i - 1][0] * grid[0][i];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j][0] = max(dp[i - 1][j][0] * grid[i][j], max(dp[i - 1][j][1] * grid[i][j],
                                                                    max(dp[i][j - 1][0] * grid[i][j],
                                                                        dp[i][j - 1][1] * grid[i][j])));
                dp[i][j][1] = min(dp[i - 1][j][0] * grid[i][j], min(dp[i - 1][j][1] * grid[i][j],
                                                                    min(dp[i][j - 1][0] * grid[i][j],
                                                                        dp[i][j - 1][1] * grid[i][j])));
            }
        }
        return dp[m - 1][n - 1][0] >= 0 ? dp[m - 1][n - 1][0] % MOD : -1;
    }

    string frequencySort(string s) {
        unordered_map<char, int> ump;
        for (auto &c:s) {
            ++ump[c];
        }
        vector<string> vec(s.size() + 1);
        for (auto &u:ump) {
            vec[u.second].append(u.second, u.first);
        }
        string res;
        for (int i = s.size(); i >= 0; i--) {
            if (!vec[i].empty()) {
                res.append(vec[i]);
            }
        }
        return res;
    }

    string longestPalindrome(string s) {
        string res;
        for (int i = 0; i < s.size(); ++i) {
            string s1 = Palindrome(s, i, i);
            string s2 = Palindrome(s, i, i + 1);
            res = res.size() >= s1.size() ? res : s1;
            res = res.size() >= s2.size() ? res : s2;
        }
        return res;
    }

    string Palindrome(string &s, int l, int r) {
        while (l >= 0 && r < s.size() && s[l] == s[r]) {
            l--;
            r++;
        }
        return s.substr(l + 1, r - l - 1);
    }

    bool isPalindrome(ListNode *head) {
        if (head == nullptr) return false;
        if (head->next == nullptr) return true;
        ListNode *fast = head->next->next, *slow = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode *newNode = slow->next;
        slow->next = nullptr;
    }

    int trap(vector<int> &height) {
        int res = 0;
        int n = height.size();
        vector<int> lmax(n), rmax(n);
        lmax[0] = height[0];
        rmax[n - 1] = height[n - 1];
        for (int i = 1; i < n; ++i) {
            lmax[i] = max(lmax[i - 1], height[i]);
        }
        for (int i = n - 2; i >= 0; --i) {
            rmax[i] = max(rmax[i + 1], height[i]);
        }
        for (int i = 1; i < n - 1; ++i) {
            res += min(lmax[i], rmax[i]) - height[i];
        }
        return res;
    }

    vector<int> corpFlightBookings(vector<vector<int>> &bookings, int n) {
        vector<int> res(n);
        vector<int> diff(n);
        for (const auto &b:bookings) {
            diff[b[0] - 1] += b[2];
            if (b[1] < n) {
                diff[b[1]] -= b[2];
            }
        }
        res[0] = diff[0];
        for (int i = 1; i < n; ++i) {
            res[i] = res[i - 1] + diff[i];
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

    int search3(vector<int> &nums, int target) {
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

    vector<int> searchRange(vector<int> &nums, int target) {
        vector<int> res = {std::move(searchRangeLower(nums, target)), std::move(searchRangeUpper(nums, target))};
    }

    int searchRangeLower(vector<int> &nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (target > nums[mid]) l = mid + 1;
            else if (target < nums[mid]) r = mid - 1;
            else r = mid;
        }
        if (nums[l] != target) return -1;
        return l;
    }

    int searchRangeUpper(vector<int> &nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r + 1) / 2;
            if (target > nums[mid]) l = mid + 1;
            else if (target < nums[mid]) r = mid - 1;
            else l = mid;
        }
        if (nums[l] != target) return -1;
        return l;
    }

    int findBestValue(vector<int> &arr, int target) {
        sort(arr.begin(), arr.end());
        int n = arr.size();
        int l = 0, r = arr.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            int sum = arr[mid] * n;
            if (sum < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return arr[l];
    }

    int minEatingSpeed(vector<int> &piles, int h) {
        int l = 1, r = *max_element(piles.begin(), piles.end());
        while (l < r) {
            int mid = (l + r) / 2;
            if (minEatingSpeedCheck(piles, h, mid)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    bool minEatingSpeedCheck(vector<int> &p, int h, int s) {
        int t = 0;
        for (int i:p) {
            if (i % s == 0) {
                t = t + i / s;
            } else {
                t = t + i / s + 1;
            }
        }
        return t <= h;
    }

    int minDays(vector<int> &bloomDay, int m, int k) {
        int l = 1, r = 1e9;
        while (l < r) {
            int mid = (l + r) / 2;

        }
    }

    int maxDistance(vector<int> &position, int m) {
        sort(position.begin(), position.end());
        int l = 1, r = position.back() - position.front();
        if (m == 2) {
            return position.back() - position.front();
        }
        while (l < r) {
            int mid = (l + r) / 2;
            if (maxDistanceCheck(position, mid, m)) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }

    bool maxDistanceCheck(vector<int> &pos, int mid, int m) {
        int cnt = 0;
        int tar = pos[0] + mid;
        for (int i = 1; i < pos.size() - 1; ++i) {
            if (pos[i] < tar && pos[i + 1] >= tar) {
                cnt++;
                tar = pos[i + 1] + mid;
            }
        }
        return cnt > m - 1;
    }

    int eraseOverlapIntervals(vector<vector<int>>& intervals) {

    }

    int findMinArrowShots(vector<vector<int>> &points) {
        if(points.size()==0) return 0;
        sort(points.begin(), points.end(), [](const auto &a, const auto &b) {
            return a[0] < b[0];
        });
        vector<int> cur = points[0];
        int cnt = 1;
        for (int i = 1; i < points.size(); ++i) {
            if (points[i][0] > cur[1]) {
                cnt++;
                cur = points[i];
            } else {
                cur[0] = max(cur[0], points[i][0]);
                cur[1] = min(cur[1], points[i][1]);
            }
        }
        return cnt;
    }

    int missingNumber(vector<int> &nums) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > mid) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }
};

#endif //LEETCODEMAC_SOLUTION2