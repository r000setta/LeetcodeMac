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

    //cout<<solution.strStr("aaaaa", "aaa");
    vector<int> v4{1, 2, 3};

    vector<string> v5{"cat", "banana", "dog", "nana", "walk", "walker", "dogwalker"};
    solution.longestWord(v5);

    vector<vector<char>> v7{{'A', 'B', 'C', 'E'},
                            {'S', 'F', 'C', 'S'},
                            {'A', 'D', 'E', 'E'}};

    vector<vector<char>> v8{{'C', 'A', 'A'},
                            {'A', 'A', 'A'},
                            {'B', 'C', 'D'}};
    Solution2 s2;
    vector<int> v6{0, 3, 2, 1};

    vector<vector<char>> v9{{'1', '0', '1', '0', '0'},
                            {'1', '0', '1', '1', '1'},
                            {'1', '1', '1', '1', '1'},
                            {'1', '0', '0', '1', '0'}};

    // s2.maximalRectangle(v9);
    vector<int> va{1, 1, 1, 2, 2, 3, 3, 3};
    ws.topKFrequent(va, 2);
}