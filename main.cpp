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
#include "DpSolution.h"
#include "Week2.h"

using namespace std;

int main() {
    WeekSolution2 ws;
    vector<int> v1{1, 2, 3, 4};
//    ws.hasAllCodes("0110", 2);
    vector<string> v2{"leetcoder", "leetcode", "od", "hamlet", "am"};
    ws.stringMatching(v2);
}