#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <limits>
#include "solution.h"
#include "top.h"
#include "Weekly.h"
#include "solution2.h"
#include "Solution3.h"
#include "DpSolution.h"
#include "Solution4.h"
#include "Week2.h"
#include "TSolution.h"

using namespace std;

int main() {
    Solution3 s3;
    TSolution ts;
    vector<int> v1{1, 3, -1, -3, -5, 3, 6, 7};

    TreeNode node1 = TreeNode(2);
    TreeNode node2 = TreeNode(3);
    TreeNode node3 = TreeNode(1);
    node1.left = &node2;
    node2.left = &node3;
    vector<string> v2{"blw", "bwl", "wlb"};
    vector<vector<int>> v3{{3,  10, 9, 5,  5,  7},
                           {0,  1,  7, 3,  8,  1},
                           {9,  3,  0, 6,  1,  6},
                           {10, 2,  9, 10, 10, 7}};
    vector<int> v4{4, 2, 1};
    vector<int> v5{9, 4, 2, 10, 7, 8, 8, 1, 9};
    vector<int> v6{3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4};
    s3.totalFruit(v6);
//    ts.countBalls(1, 10);
//    s3.stoneGameVII(v1);
}