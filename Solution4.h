#ifndef LEETCODEMAC_SOLUTION4_H
#define LEETCODEMAC_SOLUTION4_H

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

using namespace std;;

#endif //LEETCODEMAC_SOLUTION4_H

class Solution4 {
public:
    vector<int> findBall(vector<vector<int>> &grid) {
        int m = grid.size(), n = grid[0].size();
        vector<int> res(n);
        for (int i = 0; i < n; ++i) {
            res[i] = findBallHelp(grid, i, 0);
        }
        return res;
    }

    int findBallHelp(vector<vector<int>> &grid, int x, int y) {
        if (y == grid.size()) return x;
        if (x == 0 && grid[y][x] == -1) return -1;
        if (x == grid[0].size() - 1 && grid[y][x] == 1) return -1;
        if (grid[y][x] == 1 && grid[y][x + 1] == -1) return -1;
        if (grid[y][x] == -1 && grid[y][x - 1] == 1) return -1;
        return findBallHelp(grid, x + grid[y][x], y + 1);
    }

    vector<double>
    calcEquation(vector<vector<string>> &equations, vector<double> &values, vector<vector<string>> &queries) {
        int n = 0;
        unordered_map<string, int> vars;
        int m = equations.size();
        for (int i = 0; i < m; ++i) {
            if (vars.find(equations[i][0]) == vars.end()) {
                vars[equations[i][0]] = n++;
            }
            if (vars.find(equations[i][1]) == vars.end()) {
                vars[equations[i][1]] = n++;
            }
        }
        vector<vector<double>> graph(n, vector<double>(n, -1.0));
        for (int i = 0; i < n; ++i) {
            int va = vars[equations[i][0]], vb = vars[equations[i][1]];
            graph[va][vb] = values[i];
            graph[vb][va] = 1.0 / values[i];
        }
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (graph[i][k] > 0 && graph[k][j] > 0) {
                        graph[i][j] = graph[i][k] * graph[k][j];
                    }
                }
            }
        }
        vector<double> ret;
        for (const auto &q:queries) {
            double res = -1.0;
            if (vars.find(q[0]) != vars.end() && vars.find(q[1]) != vars.end()) {
                int ia = vars[q[0]], ib = vars[q[1]];
                if (graph[ia][ib] > 0) {
                    res = graph[ia][ib];
                }
            }
            ret.emplace_back(res);
        }
        return ret;
    }

    int countStudents(vector<int> &students, vector<int> &sandwiches) {
        int one = 0, zero = 0;
        for (int i:students) {
            if (i == 0) zero++;
            else one++;
        }
        for (int i:sandwiches) {
            if (i == 0) {
                if (zero <= 0) {
                    break;
                }
                zero--;
            } else {
                if (one <= 0) {
                    break;
                }
                one--;
            }
        }
        return one + zero;
    }

    double averageWaitingTime(vector<vector<int>> &customers) {
        double res = 0;
        int cur = 0;
        for (const auto &c:customers) {
            cur = max(cur, c[0]);
            cur += c[1];
            res += cur - c[0];
        }
        return res / customers.size();
    }

    int findCircleNumFind(vector<int> &p, int x) {
        if (p[x] == x) {
            return x;
        } else {
            return p[x] = findCircleNumFind(p, p[x]);
        }
    }

    void findCircleNumUnion(vector<int> &p, vector<int> &rank, int x, int y) {
        int px = findCircleNumFind(p, x);
        int py = findCircleNumFind(p, y);
        if (px == py) return;
        if (rank[px] < rank[py]) {
            p[px] = py;
            rank[px] += rank[py];
        } else {
            p[py] = px;
            rank[py] += rank[px];
        }
    }

    int findCircleNum(vector<vector<int>> &isConnected) {
        int m = isConnected.size();
        vector<int> p(m), rank(m, 1);
        for (int i = 0; i < m; ++i) {
            p[i] = i;
        }
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                if (isConnected[i][j]) {
                    findCircleNumUnion(p, rank, i, j);
                }
            }
        }
        int tmp = 0;
        for (int i = 0; i < m; ++i) {
            if (p[i] == i) {
                tmp++;
            }
        }
        return tmp;
    }

    int accountsMergeFind(vector<int> &f, int x) {
        if (f[x] == x) {
            return x;
        }
        return f[x] = accountsMergeFind(f, f[x]);
    }

    void accountsMergeUnion(vector<int> f, int x, int y) {
        int px = accountsMergeFind(f, x);
        int py = accountsMergeFind(f, y);
        if (px != py) {
            f[px] = py;
        }
    }

    vector<vector<string>> accountsMerge(vector<vector<string>> &accounts) {
        int m = accounts.size();
        unordered_set<string> s;
        unordered_map<string, int> father;
        vector<int> f(m);
        vector<vector<string>> res;
        for (int i = 0; i < m; ++i) f[i] = i;
        for (int i = 0; i < m; ++i) {
            for (int j = 1; j < m; ++j) {
                if (!s.count(accounts[i][j])) {
                    s.insert(accounts[i][j]);
                    father[accounts[i][j]] = i;
                } else {
                    accountsMergeUnion(f, father[accounts[i][j]], i);
                }
            }
        }
        unordered_map<int, set<string>> acc;
        for (int i = 0; i < m; ++i) {
            int t = accountsMergeFind(f, i);
            int len = accounts[i].size();]
            for (int j = 1; j < len; ++j) {
                acc[t].insert(accounts[i][j]);
            }
        }

    }

    void rotate(vector<int> &nums, int k) {
        int m = nums.size();
        vector<int> res(m);
        for (int i = 0; i < m; ++i) {
            res[i] = nums[(i + m - k) % m];
        }
        for (int i = 0; i < m; ++i) {
            nums[i] = res[i];
        }
    }

    int equationsPossibleFind(vector<int> &f, int x) {
        if (f[x] == x) return x;
        return f[x] = equationsPossibleFind(f, f[x]);
    }

    void equationsPossibleUnion(vector<int> &f, vector<int> &h, int x, int y) {
        int px = equationsPossibleFind(f, x);
        int py = equationsPossibleFind(f, y);
        if (h[px] < h[py]) {
            f[px] = py;
            h[px] += h[py];
        } else {
            f[py] = px;
            h[py] += h[px];
        }
    }

    bool equationsPossible(vector<string> &equations) {
        sort(equations.begin(), equations.end(), [](const auto &a, const auto &b) {
            return a[1] > b[1];
        });
        vector<int> f(26), h(26, 1);
        for (int i = 1; i < 26; ++i) f[i] = i;
        for (const auto &e:equations) {
            int x = e[0] - 'a';
            int y = e[3] - 'a';
            if (e[1] == '=') {
                equationsPossibleUnion(f, h, x, y);
            } else {
                if (equationsPossibleFind(f, x) == equationsPossibleFind(f, y)) {
                    return false;
                }
            }
        }
        return true;
    }

    struct Edge {
        int a, b, w;

        bool operator<(const Edge &e) const {
            return w < e.w;
        }
    };

    int minCostConnectPointsFind(vector<int> &p, int x) {
        if (p[x] == x) return x;
        return p[x] = minCostConnectPointsFind(p, p[x]);
    }

    void minCostConnectPointsUnion(vector<int> &p, int x, int y) {
        int px = minCostConnectPointsFind(p, x);
        int py = minCostConnectPointsFind(p, y);
        p[px] = py;
    }

    int minCostConnectPoints(vector<vector<int>> &points) {
        int m = points.size();
        vector<Edge> vec;
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                vec.push_back({i, j, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])});
            }
        }
        vector<int> p(m);
        for (int i = 0; i < m; ++i) p[i] = i;
        sort(vec.begin(), vec.end());
        int res = 0;
        for (const auto &e:vec) {
            if (minCostConnectPointsFind(p, e.a) != minCostConnectPointsFind(p, e.b)) {
                minCostConnectPointsUnion(p, e.a, e.b);
                res += e.w;
            }
        }
        return res;
    }
};