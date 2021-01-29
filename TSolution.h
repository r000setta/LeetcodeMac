#ifndef LEETCODEMAC_TSOLUTION_H
#define LEETCODEMAC_TSOLUTION_H

#include <vector>

#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>
#include <numeric>
#include "solution.h"

class TSolution {
public:
    void deleteNode(ListNode *node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};

#endif //LEETCODEMAC_TSOLUTION_H
