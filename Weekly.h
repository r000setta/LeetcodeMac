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

};

#endif //LEETCODEMAC_WEEKLY_H
