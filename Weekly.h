#ifndef LEETCODEMAC_WEEKLY_H
#define LEETCODEMAC_WEEKLY_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class WeekSolution {
public:
    char slowestKey(vector<int> &releaseTimes, string keysPressed) {
        char res = ' ';
        int vmax = INT_MIN;
        for (int i = 0; i < releaseTimes.size(); ++i) {
            int cur = 0;
            if (i == 0) {
                cur = releaseTimes[i];
            } else {
                cur = releaseTimes[i] - releaseTimes[i - 1];
            }
            if (cur >= vmax) {
                if (cur == vmax) {
                    res = res > keysPressed[i] ? res : keysPressed[i];
                } else {
                    res = keysPressed[i];
                }
                vmax = cur;
            }
        }
        return res;
    }

    vector<bool> checkArithmeticSubarrays(vector<int> &nums, vector<int> &l, vector<int> &r) {
        int n = l.size();
        vector<bool> res(n);
        for (int i = 0; i < n; ++i) {
            if (r[i] - l[i] < 2) {
                res[i] = true;
                continue;
            }
            vector<int> tmp(nums.begin() + l[i], nums.begin() + r[i] + 1);
            sort(tmp.begin(), tmp.end());
            res[i] = checkHelp(tmp);
        }
        return res;
    }

    bool checkHelp(vector<int> &nums) {
        int diff = nums[1] - nums[0];
        for (int i = 2; i < nums.size(); ++i) {
            if (nums[i] - nums[i - 1] != diff) return false;
        }
        return true;
    }

    int minimumEffortPath(vector<vector<int>> &heights) {
        int l = 0, r = 1e6, mid, ans = 0;
        while (l <= r) {
            mid = l + ((r - l) >> 1);
            vector<vector<bool>> vis(heights.size(), vector<bool>(heights[0].size()));
            if (minimumEffortPathDFS(heights, vis, 0, 0, mid)) {
                r = mid - 1;
                ans = mid;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    int minimumEffortPathDir[4][2]{{1,  0},
                                   {0,  1},
                                   {0,  -1},
                                   {-1, 0}};

    bool minimumEffortPathDFS(vector<vector<int>> &heights, vector<vector<bool>> &visit, int x, int y, int d) {
        if (x == heights.size() - 1 && y == heights[0].size() - 1) {
            return true;
        }
        visit[x][y] = true;
        for (const auto &dir:minimumEffortPathDir) {
            int newX = x + dir[0];
            int newY = y + dir[1];
            if (newX >= 0 && newX < heights.size() && newY >= 0 && newY < heights[0].size()
                && !visit[newX][newY] && abs(heights[x][y] - heights[newX][newY]) <= d) {
                if (minimumEffortPathDFS(heights, visit, newX, newY, d)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool canFormArray(vector<int> &arr, vector<vector<int>> &pieces) {
        for (int j = 0; j < pieces.size(); ++j) {
            int len = pieces[j].size();
            int idx = 0;
            for (int i = 0; i < arr.size(); ++i) {
                if (arr[i] == pieces[j][idx]) {
                    idx++;
                    if (idx == len) {
                        break;
                    }
                }
            }
            if (idx != len) return false;
        }
        return true;
    }

    int countVowelStrings(int n) {
        vector<vector<int>> dp(n + 1, vector<int>(5, 1));
        for (int i = 2; i <= n; ++i) {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = dp[i][0] + dp[i - 1][1];
            dp[i][2] = dp[i][1] + dp[i - 1][2];
            dp[i][3] = dp[i][2] + dp[i - 1][3];
            dp[i][4] = dp[i][3] + dp[i - 1][4];
        }
        return dp[n][0] + dp[n][1] + dp[n][2] + dp[n][3] + dp[n][4];
    }

    int furthestBuilding(vector<int> &heights, int bricks, int ladders) {
        int n = heights.size();
        priority_queue<int, vector<int>, greater<int>> q;
        int sum = 0;
        for (int i = 1; i < n; ++i) {
            int delta = heights[i] - heights[i - 1];
            if (delta > 0) {
                q.push(delta);
                if (q.size() > ladders) {
                    sum += q.top();
                    q.pop();
                }
                if (sum > bricks) {
                    return i - 1;
                }
            }
        }
        return n - 1;
    }

    int boardDir[8][2] = {{0,  1},
                          {1,  0},
                          {0,  -1},
                          {-1, 0},
                          {1,  1},
                          {1,  -1},
                          {-1, 1},
                          {-1, -1}};

    vector<vector<char>> updateBoard(vector<vector<char>> &board, vector<int> &click) {
        int x = click[0];
        int y = click[1];
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
        } else {
            updateBoardDFS(board, x, y);
        }
        return board;
    }

    void updateBoardDFS(vector<vector<char>> &board, int x, int y) {
        int cnt = 0;
        for (int i = 0; i < 8; ++i) {
            int tx = x + boardDir[i][0];
            int ty = y + boardDir[i][1];
            if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size()) {
                continue;
            }
            cnt += board[tx][ty] == 'M';
        }
        if (cnt > 0) {
            board[x][y] = cnt + '0';
        } else {
            board[x][y] = 'B';
            for (int i = 0; i < 8; ++i) {
                int tx = x + boardDir[i][0];
                int ty = y + boardDir[i][1];
                if (tx < 0 || tx >= board.size() || ty < 0 || ty >= board[0].size() || board[tx][ty] != 'E') {
                    continue;
                }
                updateBoardDFS(board, tx, ty);
            }
        }
    }

    vector<vector<int>> insert(vector<vector<int>> &intervals, vector<int> &newInterval) {
        int left = newInterval[0];
        int right = newInterval[1];
        bool flag = false;
        vector<vector<int>> ans;
        for (const auto &interval:intervals) {
            if (left > interval[1]) {
                ans.emplace_back(interval);
            } else if (right < interval[0]) {
                if (!flag) {
                    vector<int> tmp{left, right};
                    ans.emplace_back(std::move(tmp));
                    flag = true;
                }
                ans.emplace_back(interval);
            } else {
                left = min(left, interval[0]);
                right = max(right, interval[1]);
            }
        }
        if (!flag) {
            vector<int> tmp{left, right};
            ans.emplace_back(std::move(tmp));
        }
        return ans;
    }

    int ladderLength(string beginWord, string endWord, vector<string> &wordList) {
        unordered_set<string> words(wordList.begin(), wordList.end());
        if (words.empty() || words.find(endWord) == words.end()) return 0;
        words.erase(beginWord);
        queue<string> que;
        que.push(beginWord);
        unordered_set<string> visited;
        visited.insert(beginWord);
        int step = 1;
        while (!que.empty()) {
            int n = que.size();
            while (n--) {
                string curWord = que.front();
                que.pop();
                for (int i = 0; i < curWord.size(); ++i) {
                    char oriChar = curWord[i];
                    for (int j = 0; j < 26; ++j) {
                        if (char('a' + j) == oriChar) continue;
                        curWord[i] = (char) ('a' + j);
                        if (words.find(curWord) != words.end() && visited.find(curWord) == visited.end()) {
                            if (curWord == endWord) return step + 1;
                            else {
                                que.push(curWord);
                                visited.insert(curWord);
                            }
                        }
                    }
                    curWord[i] = oriChar;
                }
            }
            ++step;
        }
        return 0;
    }

    int countRangeSum(vector<int> &nums, int lower, int upper) {
        int m = nums.size();
        vector<int> pre(m + 1);
        for (int i = 0; i < m; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = i; j < m; ++j) {
                int sum = pre[j + 1] - pre[i];
                if (sum <= upper && sum >= lower) res++;
            }
        }
        return res;
    }

    int maxProfit(vector<int> &prices) {
        if (prices.size() == 0) return 0;
        int res = 0;
        int mval = prices[0];
        for (int i = 1; i < prices.size(); ++i) {
            if (prices[i] < mval) mval = prices[i];
            else res = max(res, prices[i] - mval);
        }
        return res;
    }

    int maxProfit2(vector<int> &prices) {
        if (prices.size() == 0) return 0;
        int res = 0;
        for (int i = 0; i < prices.size() - 1; ++i) {
            if (prices[i] < prices[i + 1]) res += prices[i + 1] - prices[i];
        }
        return res;
    }

    int maxProfit(vector<int> &prices, int fee) {
        int m = prices.size();
        if (m == 0) return 0;
        vector<vector<int>> dp(m, vector<int>(2));
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < m; ++i) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[m - 1][0];
    }

    int maxProfit3(vector<int> &prices) {
        int m = prices.size();
        if (m == 0) return 0;
        vector<vector<int>> dp(m, vector<int>(3));
        dp[0][0] = 0;
        dp[0][1] = 0;
        dp[0][2] = -prices[0];
        for (int i = 1; i < m; ++i) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] + prices[i]);
            dp[i][1] = dp[i - 1][0];
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
        }
        return dp[m - 1][0];
    }

    int maxProfit2(int k, vector<int> &prices) {
        int m = prices.size();
        if (m == 0) return 0;
        vector<vector<vector<int>>> dp(m, vector<vector<int>>(3, vector<int>(2)));
        dp[0][0][0] = 0;
        dp[0][0][1] = -prices[0];
        dp[0][1][0] = 0;
        dp[0][1][1] = -prices[0];
        dp[0][2][0] = 0;
        dp[0][2][1] = -prices[0];
        for (int i = 1; i < m; ++i) {
            dp[i][0][0] = dp[i - 1][0][0];
            dp[i][0][1] = max(dp[i - 1][0][1], dp[i - 1][0][0] - prices[i]);
            dp[i][1][0] = max(dp[i - 1][1][0], dp[i - 1][0][1] + prices[i]);
            dp[i][1][1] = max(dp[i - 1][1][1], dp[i - 1][1][0] - prices[i]);
            dp[i][2][0] = max(dp[i - 1][2][0], dp[i - 1][1][1] + prices[i]);
        }
        return dp[m - 1][2][0];
    }

    int findKthLargest(vector<int> &nums, int k) {
        int len = nums.size();
        int left = 0;
        int right = len - 1;
        int tar = len - k;
        while (true) {
            int idx = findKthLargestPar(nums, left, right);
            if (idx == tar) return nums[idx];
            else if (idx < tar) left = idx + 1;
            else right = idx - 1;
        }
    }

    int findKthLargestPar(vector<int> &nums, int l, int r) {
        int pivot = nums[l];
        int j = l;
        for (int i = l + 1; i <= r; ++i) {
            if (nums[i] < pivot) {
                j++;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[j], nums[l]);
        return j;
    }

    void sortColors(vector<int> &nums) {
        int size = nums.size();
        if (size < 2) return;
        int zero = 0;
        int two = size;
        int i = 0;
        while (i < two) {
            if (nums[i] == 0) {
                swap(nums[zero], nums[i]);
                zero++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                two--;
                swap(nums[i], nums[two]);
            }
        }
    }

    static bool topKFrequentCmp(pair<int, int> &m, pair<int, int> &n) {
        return m.second > n.second;
    }

    vector<int> topKFrequent(vector<int> &nums, int k) {
        unordered_map<int, int> mp;
        for (int n:nums) {
            mp[n]++;
        }
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&topKFrequentCmp)> q(topKFrequentCmp);
        for (auto &m:mp) {
            if (q.size() == k) {
                if (q.top().second < m.second) {
                    q.pop();
                    q.emplace(m.first, m.second);
                }
            } else {
                q.emplace(m.first, m.second);
            }
        }
        vector<int> res(k);
        for (int i = 0; i < k; ++i) {
            res[i] = q.top().first;
            q.pop();
        }
        return res;
    }

    void nextPermutation(vector<int> &nums) {
        if (nums.size() == 0 || nums.size() == 1) return;
        int i = 0;
        int m = nums.size();
        for (i = m - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                break;
            }
        }
        if (i == -1) {
            reverse(nums.begin(), nums.end());
        } else {
            for (int j = m - 1; j >= i + 1; j--) {
                if (nums[i] < nums[j]) {
                    swap(nums[i], nums[j]);
                    reverse(nums.begin() + i + 1, nums.end());
                    break;
                }
            }
        }
    }

    vector<int> sortArrayByParityII(vector<int> &A) {
        stack<int> s1;
        stack<int> s2;
        for (int i = 0; i < A.size(); ++i) {
            if (i % 2 == 0 && A[i] % 2 != 0) {
                s1.push(i);
            } else if (i % 2 == 1 && A[i] % 2 != 1) {
                s2.push(i);
            }
        }
        while (!s1.empty() && !s2.empty()) {
            swap(A[s1.top()], A[s2.top()]);
            s1.pop();
            s2.pop();
        }
        return A;
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

    vector<int> relativeSortArray(vector<int> &arr1, vector<int> &arr2) {
        int upper = *max_element(arr1.begin(), arr1.end());
        vector<int> frequency(upper + 1);
        for (int x: arr1) {
            ++frequency[x];
        }
        vector<int> ans;
        for (int x: arr2) {
            for (int i = 0; i < frequency[x]; ++i) {
                ans.push_back(x);
            }
            frequency[x] = 0;
        }
        for (int x = 0; x <= upper; ++x) {
            for (int i = 0; i < frequency[x]; ++i) {
                ans.push_back(x);
            }
        }
        return ans;
    }

    string removeKdigits(string num, int k) {
        vector<char> stk;
        for (auto &d:num) {
            while (stk.size() > 0 && stk.back() > d && k) {
                stk.pop_back();
                k--;
            }
            stk.push_back(d);
        }
        for (; k > 0; --k) {
            stk.pop_back();
        }
        string ans = "";
        bool flag = true;
        for (auto &digit:stk) {
            if (flag && digit == '0') {
                continue;
            }
            flag = false;
            ans += digit;
        }
        return ans == "" ? "0" : ans;
    }

    vector<vector<int>> reconstructQueue(vector<vector<int>> &people) {
        sort(people.begin(), people.end(), [](const auto &p1, const auto &p2) {
            return p1[0] == p2[0] ? p1[1] < p2[1] : p1[0] > p2[0];
        });
        vector<vector<int>> ans;
        for (auto &p:people) {
            ans.insert(ans.begin() + p[1], p);
        }
        return ans;
    }

    int allCellsDistOrderDir[4][2] = {{0,  1},
                                      {0,  -1},
                                      {1,  0},
                                      {-1, 0}};

    vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {
        vector<vector<int>> res;
        vector<vector<bool>> vis(R, vector<bool>(C));
        queue<pair<int, int>> q;
        q.push({r0, c0});
        while (!q.empty()) {
            auto p = q.front();
            q.pop();
            int x = p.first, y = p.second;
            if (vis[x][y]) continue;
            vis[x][y] = true;
            vector<int> a{x, y};
            res.emplace_back(a);
            for (const auto &dir:allCellsDistOrderDir) {
                int tx = x + dir[0];
                int ty = y + dir[1];
                if (tx < 0 || tx >= R || ty < 0 || ty >= C || vis[tx][ty]) {
                    continue;
                } else {
                    q.push({tx, ty});
                }
            }
        }
        return res;
    }

    int canCompleteCircuit(vector<int> &gas, vector<int> &cost) {
        int n = gas.size();
        int i = 0;
        while (i < n) {
            int sgas = 0, scost = 0;
            int cnt = 0;
            while (cnt < n) {
                int j = (i + cnt) % n;
                sgas += gas[j];
                scost += cost[j];
                if (scost > sgas) {
                    break;
                }
                cnt++;
            }
            if (cnt == n) {
                return i;
            } else {
                i = i + cnt + 1;
            }
        }
        return -1;
    }

    bool closeStrings(string word1, string word2) {
        int n1 = word1.length(), n2 = word2.length();
        if (n1 != n2) {
            return false;
        }
        vector<int> v1(26), v2(26);
        for (int i = 0; i < n1; ++i) {
            ++v1[word1[i] - 'a'];
            ++v2[word2[i] - 'a'];
        }
        for (int i = 0; i < 26; ++i) {
            if ((v1[i] == 0) ^ (v2[i] == 0)) return false;
        }
        sort(v1.begin(), v1.end());
        sort(v2.begin(), v2.end());
        for (int i = 0; i < 26; ++i) {
            if (v1[i] != v2[i]) {
                return false;
            }
        }
        return true;
    }

    int minOperations(vector<int> &nums, int x) {
        unordered_map<int, int> um;
        int ans = INT_MAX;
        um[0] = -1;
        int sum = 0;
        for (int i = 0; i < nums.size(); ++i) {
            sum += nums[i];
            um[sum] = i;
        }
        sum = 0;
        if (um.find(x) != um.end()) {
            ans = min(ans, um[x] + 1);
        }
        for (int i = nums.size() - 1; i >= 0; --i) {
            sum += nums[i];
            if (um.find(x - sum) != um.end()) {
                int cnt = nums.size() - i + um[x - sum] + 1;
                if (cnt > nums.size()) continue;
            }
        }
    }

    vector<int> decrypt(vector<int> &code, int k) {
        int m = code.size();
        vector<int> res(m);
        if (k == 0) {
            return res;
        }
        for (int i = 0; i < m; ++i) {
            int sum = 0;
            for (int j = 0; j < abs(k); ++j) {
                if (k > 0) {
                    sum += code[(i + j + 1) % m];
                } else {
                    sum += code[(i - j - 1 + m) % m];
                }
            }
            res[i] = sum;
        }
        return res;
    }

    void moveZeroes(vector<int> &nums) {
        if (nums.empty()) return;
        int j = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] != 0) {
                nums[j++] = nums[i];
            }
        }
        for (int i = j; i < nums.size(); ++i) {
            nums[i] = 0;
        }
    }

    int minimumDeletions(string s) {
        int n = s.length();
        if (n == 0) return 0;
        vector<int> a(n);
        vector<int> b(n);
        if (s[0] == 'a') a[0]++;
        for (int i = 1; i < n; ++i) {
            a[i] = s[i] == 'a' ? a[i - 1] + 1 : a[i - 1];
        }
        if (s[n - 1] == 'b') b[n - 1]++;
        for (int i = n - 2; i >= 0; --i) {
            b[i] = s[i] == 'b' ? b[i + 1] + 1 : b[i + 1];
        }
        int mmax = 0;
        for (int i = 0; i < n; ++i) {
            mmax = max(mmax, b[i] + a[i]);
        }
        return n - mmax;
    }
};

#endif //LEETCODEMAC_WEEKLY_H
