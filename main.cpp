#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <limits>
#include "solution.h"
#include "top.h"
#include "Weekly.h"

using namespace std;

int main() {
    Solution solution;
    //solution.isMatch2("aab","c*a*b");

    TopSolution s;
    //s.lengthOfLongestSubstring("abcabcbb");
    vector<int> v{1, 2, 2, 1, 1, 3};
    //solution.uniqueOccurrences(v);
    //s.longestPalindrome("cbbd");

    WeekSolution ws;
    vector<vector<int>> vec{{1, 2, 2},
                            {3, 8, 2},
                            {5, 3, 5}};
    vector<int> v1{-12, -9, -3, -12, -6, 15, 20, -25, -20, -15, -10};
    vector<int> v2{0, 1, 6, 4, 8, 7};
    vector<int> v3{4, 4, 9, 7, 9, 10};
    //ws.checkArithmeticSubarrays(v1, v2, v3);

    cout<<solution.strStr("aaaaa", "aaa");
}