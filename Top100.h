#ifndef LEETCODEMAC_TOP100_H
#define LEETCODEMAC_TOP100_H

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

class Top100 {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> path;
        partitionBP(s, res, path, 0, 0);
        return res;
    }

    void partitionBP(string s, vector<vector<string>> &res, vector<string> &path, int r, int len) {
        if (len == s.size()) {
            res.emplace_back(path);
            return;
        }
        for (int i = r; i < s.size(); ++i) {
            string cur = s.substr(r, i - r + 1);
            if (partitionCheck(cur)) {
                path.emplace_back(cur);
                partitionBP(s, res, path, i + 1, len + cur.size());
                path.pop_back();
            }
        }
    }

    bool partitionCheck(string s) {
        int l = 0, r = s.size() - 1;
        while (l < r) {
            if (s[l] != s[r]) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    ListNode *partition(ListNode *head, int x) {
        ListNode *less = new ListNode(0), *t1 = less;
        ListNode *more = new ListNode(0), *t2 = more;
        while (head != nullptr) {
            if (head->val < x) {
                less->next = head;
                less = less->next;
            } else {
                more->next = head;
                more = more->next;
            }
            head = head->next;
        }
        more->next = nullptr;
        less->next = t2->next;
        return t1->next;
    }

    vector<vector<int>> largeGroupPositions(string s) {
        vector<vector<int>> res;
        int l = 0, r = 0;
        while (r < s.size()) {
            if (s[l] != s[r]) {
                if (r - l + 1 > 3) {
                    res.emplace_back(vector<int>{l, r - 1});
                }
                l = r;
            }
            if (r - l + 1 > 3) {
                res.emplace_back(vector<int>{l, r - 1});
            }
            r++;
        }
        return res;
    }

    int findCircleNum(vector<vector<int>> &isConnected) {
        int n = isConnected.size();
        vector<int> p(n);
        for (int i = 0; i < n; ++i) {
            p[i] = i;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (i == j) continue;
                if (isConnected[i][j] == 1) {
                    findCircleNumUnion(p, i, j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < p.size(); ++i) {
            if (p[i] != i) {
                res++;
            }
        }
        return n - res;
    }

    void findCircleNumUnion(vector<int> &p, int a, int b) {
        int pa = findCircleNumFind(p, a);
        int pb = findCircleNumFind(p, b);
        if (pa != pb) {
            p[pa] = pb;
        }
    }

    int findCircleNumFind(vector<int> &p, int x) {
        if (p[x] == x) {
            return x;
        }
        return p[x] = findCircleNumFind(p, p[x]);
    }

    void rotate(vector<int> &nums, int k) {
        int n = nums.size();
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k % n);
        reverse(nums.begin() + k % n, nums.end());
    }

    vector<string> summaryRanges(vector<int> &nums) {
        vector<string> res;
        int i = 0;
        int n = nums.size();
        while (i < n) {
            int l = i;
            i++;
            while (i < n && nums[i] == nums[i - 1] + 1) {
                i++;
            }
            int r = i - 1;
            if (l < r) {
                string t = to_string(nums[l]) + "->" + to_string(nums[r]);
                res.emplace_back(t);
            } else {
                string t = to_string(nums[l]);
                res.emplace_back(t);
            }
        }
        return res;
    }

    int smallestStringWithSwapsFind(vector<int> &p, int x) {
        if (p[x] == x) {
            return x;
        }
        return p[x] = smallestStringWithSwapsFind(p, p[x]);
    }

    void smallestStringWithSwapsUnion(vector<int> &p, int x, int y) {
        int px = smallestStringWithSwapsFind(p, x);
        int py = smallestStringWithSwapsFind(p, y);
        if (px != py) {
            p[px] = py;
        }
    }

    string smallestStringWithSwaps(string s, vector<vector<int>> &pairs) {
        vector<int> p(s.size());

    }

    vector<bool> prefixesDivBy5(vector<int> &A) {

    }

    int removeStonesFind(vector<int> &p, int x) {
        if (p[x] == x) {
            return x;
        }
        return p[x] = removeStonesFind(p, p[x]);
    }

    void removeStonesUnion(vector<int> &p, int x, int y) {
        int px = removeStonesFind(p, x);
        int py = removeStonesFind(p, y);
        if (px != py) {
            p[px] = py;
        }
    }

    int removeStones(vector<vector<int>> &stones) {
        int n = stones.size();
        UnionFind u(n);
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                if (i == j) continue;
                if (stones[i][0] == stones[j][0]
                    || stones[i][1] == stones[j][1]) {
                    if (u.find(i) != u.find(j)) {
                        u.merge(i, j);
                    }
                }
            }
        }
        return n - u.count;
    }

    double slice(const string &s) {
        int cnt = 0;
        int l = 0;
        int n = s.size();
        while (l < n) {
            char cur = s[l];
            l++;
            while (l < n && s[l] == cur) {
                l++;
            }
            cnt++;
        }
        double res = static_cast<float>(s.size()) / cnt;
        return res;
    }

    class UnionFind {
    public:
        vector<int> p;
        int n;
        int count;

        UnionFind(int n) {
            p = vector<int>(n);
            for (int i = 0; i < n; ++i) {
                p[i] = i;
            }
            count = n;
        }

        int find(int x) {
            if (p[x] == x) return x;
            return p[x] = find(p[x]);
        }

        void merge(int x, int y) {
            int px = find(x);
            int py = find(y);
            if (px != py) {
                count--;
                p[px] = py;
            }
        }
    };

    bool checkStraightLine(vector<vector<int>> &coordinates) {
        if (coordinates.size() <= 2) return true;
        int x = coordinates[0][0], y = coordinates[0][1];
        int x1 = coordinates[1][0] - x, y1 = coordinates[1][1] - y;
        for (int i = 2; i < coordinates.size(); ++i) {
            int x2 = coordinates[i][0] - x, y2 = coordinates[i][1] - y;
            if (x1 * y2 - x2 * y1 != 0) return false;
        }
        return true;
    }

    vector<vector<string>> accountsMerge(vector<vector<string>> &accounts) {
        vector<vector<string>> res;
        unordered_map<string, int> mp;
        int n = accounts.size();
        UnionFind u(n);
        for (int i = 0; i < accounts.size(); ++i) {
            int m = accounts[i].size();
            for (int j = 1; j < m; ++j) {
                string s = accounts[i][j];
                if (mp.find(s) == mp.end()) {
                    mp[s] = i;
                } else {
                    u.merge(i, mp[s]);
                }
            }
        }

        unordered_map<int, vector<string>> mp2;
        for (auto &p:mp) {
            mp2[u.find(p.second)].emplace_back(p.first);
        }
        for (auto &p:mp2) {
            sort(p.second.begin(), p.second.end());
        }
    }

    vector<int> addToArrayForm(vector<int> &A, int K) {
        vector<int> res;
        int n = A.size();
        int sum = 0, car = 0;
        for (int i = n - 1; i >= 0; --i) {
            sum = A[i] + K % 10 + car;
            K /= 10;
            car = sum / 10;
            res.emplace_back(sum % 10);
        }
        if (car) K++;
        while (K) {
            res.emplace_back(K % 10);
            K /= 10;
        }
        reverse(res.begin(), res.end());
        return res;
    }

    int profitableSchemes(int n, int minProfit, vector<int> &group, vector<int> &profit) {
        vector<bool> vis(group.size());
        int res = 0;
        profitableSchemesHelp(n, minProfit, group, profit, res, 0);
        return res;
    }

    int MOD = 1e9 + 7;

    void
    profitableSchemesHelp(int n, int minProfit, vector<int> &group, vector<int> &profit, int &res, int s) {
        if (minProfit <= 0) {
            res = (res + 1) % MOD;
        }
        for (int i = s; i < group.size(); ++i) {
            if (n >= group[i]) {
                profitableSchemesHelp(n - group[i], minProfit - profit[i], group, profit, res, i + 1);
            }
        }
    }
};

#endif //LEETCODEMAC_TOP100_H
