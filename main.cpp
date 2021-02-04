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
    ts.diameterOfBinaryTree(&node1);
//    s3.stoneGameVII(v1);
}