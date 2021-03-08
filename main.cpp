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
#include "Solution5.h"
#include "Offer.h"
#include "T50.h"
#include "Top100.h"

using namespace std;

int main() {
    Solution3 s3;
    Solution2 s2;
    Solution5 s5;
    TSolution ts;
    Offer o;
    Top100 top100;
    vector<int> v1{1, 3, -1, -3, -5, 3, 6, 7};
    auto t = v1.data();
    TreeNode node1 = TreeNode(2);
    TreeNode node2 = TreeNode(3);
    TreeNode node3 = TreeNode(1);
    node1.left = &node2;
    node2.left = &node3;
    vector<string> v2{"blw", "bwl", "wlb"};
    vector<vector<int>> v3{{0, 0},
                           {0, 1},
                           {1, 0},
                           {1, 2},
                           {2, 1},
                           {2, 2}};
    vector<vector<int>> vt{{0, 1},
                           {1, 0}};
    vector<int> v4{3, 9, 20, 15, 7};
    vector<int> v5{9, 3, 15, 20, 7};
    vector<int> v6{3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4};
    vector<int> v7{1, 5, 6, 7, 8, 10, 6, 5, 6};
    vector<int> v8{0, 1, 2, 4, 5, 7};

    vector<int> g1 = {2, 2};
    vector<int> g2 = {2, 3};
    top100.profitableSchemes(5, 3, g1, g2);
}