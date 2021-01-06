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
#include "Week2.h"

using namespace std;

int main() {
    Solution3 s3;
    vector<vector<int>> v1{{5,  1,  9,  11},
                           {2,  4,  8,  10},
                           {13, 3,  6,  7},
                           {15, 14, 12, 16}};
    s3.rotate(v1);
//    s3.stoneGameVII(v1);
}