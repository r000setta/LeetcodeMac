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

using namespace std;

int main() {
    DPSolution dps;

    WeekSolution ws;
    //ws.removeKdigits("3002",1);
    vector<int> v1{1, 1};
    dps.findShortestSubArray(v1);
}