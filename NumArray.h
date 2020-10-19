#ifndef LEETCODEMAC_NUMARRAY_H
#define LEETCODEMAC_NUMARRAY_H

#include <vector>

using namespace std;

class NumArray {
public:
    vector<int> tree;
    int n;

    void buildTree(vector<int> &nums) {
        for (int i = n, j = 0; i < 2 * n; ++i, ++j) {
            tree[i] = nums[j];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }

    NumArray(vector<int> &nums) {
        if (nums.size() >= 0) {
            n = nums.size();
            tree = vector<int>(n * 2);
            buildTree(nums);
        }
    }

    void update(int i, int val) {
        i += n;
        tree[i] = val;
        while (i > 0) {
            int left = i;
            int right = i;
            if (i % 2 == 0) {
                right = i + 1;
            } else {
                left = i - 1;
            }
            tree[i / 2] = tree[left] + tree[right];
            i /= 2;
        }
    }

    int sumRange(int i, int j) {
        i += n;
        j += n;
        int sum = 0;
        while (i <= j) {
            if ((i % 2) == 1) {
                sum += tree[i];
                i++;
            }
            if ((j % 2) == 0) {
                sum += tree[j];
                j--;
            }
            i /= 2;
            j /= 2;
        }
        return sum;
    }
};

#endif //LEETCODEMAC_NUMARRAY_H
