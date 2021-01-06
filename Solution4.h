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
};