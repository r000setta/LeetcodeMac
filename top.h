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
        int left = 0, right = 0, res = 0;
        while (left < s.size()) {
            if (!cnt.count(s[left])) {
                cnt.insert(s[left]);
                res = max(left - right + 1, res);
                left++;
            } else {
                while (cnt.count(s[left])) {
                    cnt.erase(s[right]);
                    right++;
                }
                cnt.insert(s[left]);
                left++;
            }
        }
        return res;
    }

    double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {

    }

    string longestPalindrome(string s) {
        int len = s.size();
        if (len < 2) return s;
        int maxLen = 1;
        int begin = 0;
        vector<vector<bool>> dp(len, vector<bool>(len));
        for (int i = 0; i < len; ++i) dp[i][i] = true;
        for (int j = 1; j < len; ++j) {
            for (int i = 0; i < j; ++i) {
                if (s[i] != s[j]) {
                    dp[i][j] = false;
                    continue;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substr(begin, begin + maxLen);
    }

    int lhs(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size();
        vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1));
        for (int i = 1; i <= n1; ++i) {
            for (int j = 1; j <= n2; ++j) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n1][n2];
    }

    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > INT_MAX / 10 || (rev == INT_MAX / 10 && pop > 7)) return 0;
            if (rev < INT_MIN / 10 || (rev == INT_MIN / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

    int myAtoi(string s) {

    }

    int maxArea(vector<int> &height) {
        int l = 0, r = height.size() - 1, res = 0;
        while (l < r) {
            res = max((r - l * min(height[l], height[r])), res);
            if (height[l] > height[r]) {
                r--;
            } else {
                l++;
            }
        }
        return res;
    }

    string longestCommonPrefix(vector<string> &strs) {
        if (!strs.size()) return "";
        int len = strs[0].size();
        int count = strs.size();
        for (int i = 0; i < len; ++i) {
            char c = strs[0][i];
            for (int j = 1; j < count; ++j) {
                if (i == strs[j].size() || strs[j][i] != c) {
                    return strs[0].substr(0, i);
                }
            }
        }
        return strs[0];
    }

    vector<string> letters{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) return {};
        vector<string> ans;
        string path;
        letterCombinationsBP(digits, path, ans, 0);
        return ans;
    }

    void letterCombinationsBP(string &digits, string &path, vector<string> &ans, int idx) {
        if (idx == digits.size()) {
            ans.emplace_back(path);
            return;
        } else {
            int d = digits[idx] - '0';
            const string letter = letters[d - 2];
            for (const char &l:letter) {
                path.push_back(l);
                letterCombinationsBP(digits, path, ans, idx + 1);
                path.pop_back();
            }
        }
    }

    ListNode *removeNthFromEnd(ListNode *head, int n) {
        ListNode *dummy = new ListNode(0);
        dummy->next = head;
        ListNode *slow = dummy, *fast = dummy;
        for (int i = 0; i <= n; i++) {
            fast = fast->next;
        }
        while (fast != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        return dummy->next;
    }

    bool isValidKuo(string s) {
        stack<char> stk;
        for (const char &c:s) {
            if (c == '{' || c == '[' || c == '(') {
                stk.push(c);
            } else {
                if (stk.empty()) return false;
                switch (c) {
                    case '}':
                        if (stk.top() != '{')
                            return false;
                        break;
                    case ']':
                        if (stk.top() != '[')
                            return false;
                        break;
                    case ')':
                        if (stk.top() != '(')
                            return false;
                        break;
                }
                stk.pop();
            }
        }
        return stk.empty();
    }

    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        ListNode *res = new ListNode(0);
        ListNode *rtmp = res;
        ListNode *t1 = l1, *t2 = l2;
        while (t1 != nullptr && t2 != nullptr) {
            if (t1->val < t2->val) {
                rtmp->next = t1;
                t1 = t1->next;
            } else {
                rtmp->next = t2;
                t2 = t2->next;
            }
            rtmp = rtmp->next;
        }
        if (t1 == nullptr) rtmp->next = t2;
        if (t2 == nullptr) rtmp->next = t1;
        return res->next;
    }

    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string path;
        generateParenthesisBP(n, res, path, 0, 0);
        return res;
    }

    void generateParenthesisBP(int n, vector<string> &res, string &path, int left, int right) {
        if (path.size() == 2 * n) {
            res.emplace_back(path);
            return;
        }
        if (left < n) {
            path.push_back('(');
            generateParenthesisBP(n, res, path, left + 1, right);
            path.pop_back();
        }
        if (right < left) {
            path.push_back(')');
            generateParenthesisBP(n, res, path, left, right + 1);
            path.pop_back();
        }
    }

    ListNode *mergeKLists(vector<ListNode *> &lists) {
        if (lists.size() == 0) return nullptr;
        return mergeKListsHelp(lists, 0, lists.size() - 1);
    }

    ListNode *mergeKListsHelp(vector<ListNode *> &lists, int left, int right) {
        if (left >= right) return lists[left];
        int mid = (left + right) / 2;
        ListNode *l = mergeKListsHelp(lists, left, mid);
        ListNode *r = mergeKListsHelp(lists, mid + 1, right);
        return mergeTwoLists(l, r);
    }
};

#endif //LEETCODEMAC_TOP_H
