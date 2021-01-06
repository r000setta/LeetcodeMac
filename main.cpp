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
#include "Solution4.h"

using namespace std;

int main() {
    Solution3 s3;
    Solution4 s4;
    vector<int> vec{1, 2, 2, 2, 5, 0};
    vector<vector<int>> v2{{1,  1,  1,  -1, -1},
                           {1,  1,  1,  -1, -1},
                           {-1, -1, -1, 1,  1},
                           {1,  1,  1,  1,  -1},
                           {-1, -1, -1, -1, -1}};
    s4.findBall(v2);
}