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
    vector<int> vv{2, 4, 7, 1};
    DPSolution dps;
    dps.canPartition(vv);
}