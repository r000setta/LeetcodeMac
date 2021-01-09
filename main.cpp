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

using namespace std;

int main() {
    Solution4 s4;
    vector<vector<string>> v1{{"John", "johnsmith@mail.com", "john_newyork@mail.com"},
                              {"John", "johnsmith@mail.com", "john00@mail.com"},
                              {"Mary", "mary@mail.com"},
                              {"John", "johnnybravo@mail.com"}};
    s4.accountsMerge(v1);

//    s3.stoneGameVII(v1);
}