#ifndef LEETCODEMAC_MYCALENDAR_H
#define LEETCODEMAC_MYCALENDAR_H

#include <map>

using namespace std;

class MyCalendar {
public:
    map<int, int> ans;

    MyCalendar() {
        ans.clear();
    }

    bool book(int start, int end) {
        --end;
        auto p = ans.lower_bound(start);
        if (p != ans.end() && p->second <= end) {
            return false;
        }
        ans[end] = start;
        return true;
    }
};

class MyCalendarTwo {
public:
    MyCalendarTwo() {

    }

    bool book(int start, int end) {

    }
};

#endif //LEETCODEMAC_MYCALENDAR_H
