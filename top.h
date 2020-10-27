#ifndef LEETCODEMAC_TOP_H
#define LEETCODEMAC_TOP_H

#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class TopSolution {
public:
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *res = new ListNode(0);
        ListNode *tres = res;
        ListNode *t1 = l1, *t2 = l2;
        int sum = 0;
        int carry = 0;
        while (t1 != nullptr || t2 != nullptr) {
            int v1 = t1 == nullptr ? 0 : t1->val;
            int v2 = t2 == nullptr ? 0 : t2->val;
            sum = v1 + v2 + carry;
            ListNode *tmp = new ListNode(sum % 10);
            carry = sum / 10;
            tres->next = tmp;
            tres = tmp;
            if (t1) t1 = t1->next;
            if (t2) t2 = t2->next;
        }
        if (carry != 0) {
            tres->next = new ListNode(1);
        }
        return res->next;
    }

    int lengthOfLongestSubstring(string s) {
        unordered_set<char> cnt;
        int left = 0, right = 0, res = 1;
        while (left < s.size()) {
            if (!cnt.count(s[left])) {
                cnt.insert(s[left]);
                left++;
            } else {
                res = max(left - right + 1, res);
                while (cnt.count(s[left])) {
                    right++;
                    cnt.erase(s[left]);
                }
                cnt.insert(s[left]);
            }
        }
        res = max(res, left - right + 1);
        return res;
    }
};

#endif //LEETCODEMAC_TOP_H
