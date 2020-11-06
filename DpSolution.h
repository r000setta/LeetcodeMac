#ifndef LEETCODEMAC_DPSOLUTION_H
#define LEETCODEMAC_DPSOLUTION_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

class DPSolution {
public:
    string longestDiverseString(int a, int b, int c) {
        vector<vector<char>> v;
        v.push_back({(char) a, 'a'});
        v.push_back({(char) b, 'b'});
        v.push_back({(char) c, 'c'});
        string res;
        while (res.size() < a + b + c) {
            sort(v.rbegin(), v.rend());
            if ((res.size() > 0) && (res.back() == v[0][1])) {
                if (v[1][0]-- > 0) res.push_back(v[1][1]);
                else return res;
            } else {
                if (v[0][0]-- > 0) res.push_back(v[0][1]);
                if (v[0][0]-- > 0) res.push_back(v[0][1]);
            }
        }
        return res;
    }

    int maxSumAfterPartitioning(vector<int> &arr, int k) {
        int n = arr.size();
        vector<int> dp(n + 1);
        for (int i = 0; i <= n; ++i) {
            int curMax = 0;
            for (int j = i - 1; (i - j) < k && j >= 0; --j) {
                curMax = max(curMax, arr[j]);
                dp[i] = max(dp[i], dp[i] + (i - j) * curMax);
            }
        }
        return dp[n];
    }

    int splitArray(vector<int> &nums, int m) {
        int n = nums.size();
        vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, LLONG_MAX));
        vector<long long> sub(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            sub[i + 1] = sub[i] + nums[i];
        }
        dp[0][0] = 0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= min(i, m); ++j) {
                for (int k = 0; k < i; ++k) {
                    dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sub[i] - sub[k]));
                }
            }
        }
        return (int) dp[n][m];
    }
};

#endif //LEETCODEMAC_DPSOLUTION_H
