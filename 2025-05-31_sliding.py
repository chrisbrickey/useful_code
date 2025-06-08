# SLIDING PRACTICE

from collections import Counter

def sliding_average(nums, span, threshold):
    count, window_sum = 0, 0

    for right, num in enumerate(nums):
        window_sum += num

        if right < span - 1:
            continue

        # comparison between float and int (ok)
        if window_sum / span >= threshold:
            count += 1

        left = right - span + 1
        dropped = nums[left]
        window_sum -= dropped

    return count

def find_indices(word, pattern):
    # Dicts can be compared shallowly with == ; Time:O(1) if constrain type of characters
    indices, span = [], len(pattern)
    w_store, p_store = Counter(), Counter(pattern)

    if len(word) < span:
        return indices

    for right, char in enumerate(word):
        if char in p_store:
            w_store[char] += 1

        if right < span - 1:
            continue

        left = right - span + 1
        # Optimization: keep track of points and compare integers to integers to ensure constant time comparison
        if p_store == w_store:
            indices.append(left)

        dropped = word[left]
        if dropped in p_store:
            w_store[dropped] -= 1

    return indices

def min_window_substring(word, pattern):
    min_size, min_string = float('inf'), ""
    w_store, p_store = Counter(), Counter(pattern)
    w_points, p_points = 0, len(pattern)

    left = 0
    for right, char in enumerate(word):
        if char in p_store:
            w_store[char] += 1
            if w_store[char] == p_store[char]:
                w_points += p_store[char]

        # We have to use points instead of comparing dicts because duplicates in the output is accepted.
        while left <= right and p_points == w_points:
            size = right - left + 1
            if size < min_size:
                min_size, min_string = size, word[left: right + 1]

            dropped = word[left]
            left += 1
            if dropped in p_store:
                w_store[dropped] -= 1
                if w_store[dropped] == p_store[dropped] - 1:
                    w_points -= p_store[dropped]

    return min_string

def max_window_no_duplicates(phrase):
    if len(phrase) < 2:
        return len(phrase)

    window_set, largest_size = set(), float('-inf')
    left = 0
    for right, char in enumerate(phrase):
        # No boundary check required
        while char in window_set:
            dropped = phrase[left]
            left += 1
            window_set.remove(dropped)

        # no dups at this point
        window_set.add(char)
        largest_size = max(largest_size, (right - left + 1))

    return largest_size

# REDACTED
def rolling_sum_of_time_windows(logs, second_range):
    pass

def subarray_sum_to_k(nums, target):
    count, window_sum = 0, 0
    store = Counter()
    store[0] = 1

    for num in nums:
        window_sum += num
        diff_from_target = window_sum - target

        # prefix count that sums to diff
        count += store[diff_from_target]

        store[window_sum] += 1

    return count

def find_products_beneath_limit(nums, limit):
    count, running_product = 0, 1

    left = 0
    for right, num in enumerate(nums):

        # if this number can never be included; advance right and left beyond it
        if num >= limit:
            running_product = 1
            left = right + 1
            continue

        running_product *= num

        # drop from left while losing; stop short of right because that subarray of length 1 was already counted
        while left < right and running_product >= limit:
            dropped = nums[left]
            left += 1
            running_product /= dropped

        # winning now so increment count
        # count of subarrays within a range is the range itself!
        count += (right - left + 1)

    return count


# TESTS ============

average_tests = [
        [[1, 2, 3], 2, 2, 1],
        [[5, 3], 1, 10, 0],
        [[2, 2, 2, 2, 5, 5, 5, 8], 3, 4, 3],
        [[11, 13, 17, 23, 29, 31, 7, 5, 2, 3], 3, 5, 6],
    ]

for test in average_tests:
    result = sliding_average(test[0], test[1], test[2])
    assert result == test[3], f"actual: {result}; expected: {test[3]}"

sliding_pattern_tests = [
    ["a", "b", []],
    ["a", "a", [0]],
    ["abab", "ab", [0, 1, 2]],
    ["cbaebabacd", "abc", [0, 6]],
    ["cbbaebababcd", "abbc", [0, 7]], # doubles in pattern!
    ["xy", "abc", []]
]

for test in sliding_pattern_tests:
    result = find_indices(test[0], test[1])
    assert result == test[2], f"actual: {result}; expected: {test[2]}"

substring_tests = [  # s1, s2, expectation
    ["a", "b", ""],
    ["a", "a", "a"],
    ["a", "aa", ""],  # use s2 as guide; s1 must be at least as long as s2
    ["CZYKBAACDAZZM", "ABC", "BAAC"], # !! You might have more than the requirement in the output - you just need to get the minimum.
    ["ADOBCEBCNA", "ABC", "BCNA"],
    ["ADOBECODEBANC", "ABC", "BANC"],
    ["ADOBE", "E", "E"],  # result is at least as large as s2
    ["ADOBE", "OC", ""],
    ["ADOBE", "OA", "ADO"],
    ["ABACDDAC", "AA", "ABA"],
    ["ABCDDEFZGABCDEABCDDEF", "DDBBA", "BCDEABCD"]
    # order does not matter, but we have to consider the non-relevant characters that make substrings  larger than the pattern
]

for test in substring_tests:
    result2 = min_window_substring(test[0], test[1])
    assert result2 == test[2], f"actual: {result2}; expected: {test[2]}"

no_dups_tests = [
    ['', 0],
    ['a', 1],
    ['aa', 1],
    ['Aa', 2],
    ['AaA', 2],
    ['zxyzxyz', 3],
    ['zzzz', 1],
    ['abcabcbb', 3],
    ['pwwkew', 3]
]

for test in no_dups_tests:
    result = max_window_no_duplicates(test[0])
    assert result == test[1],f"actual: {result}; expected: {test[1]}"


input00 = [] # REDACTED
expected00 = [3, -4]
result00 = rolling_sum_of_time_windows(input00, 3)
assert result00 == expected00,f"actual: {result00}; expected: {expected00}"

input0 = [] # REDACTED
expected0 = [3, -4, 2, 1, 2]
result0 = rolling_sum_of_time_windows(input0, 3)
assert result0 == expected0,f"actual: {result0}; expected: {expected0}"

tests = [
    [[9], -2, 0],
    [[9], 9, 1], # [9]


    # 0  -3
    [[-3, 2], -1, 1], # [-3, 2]

    # {2: 2, 1: 1, }
    # 0  2  1  2
    [[2,-1, 1, 2], 2, 4], #  [2], [2,-1,1], [-1,1,2], [2]
    [[4,4,4,4,4,4],4, 6],
]

for test in tests:
    result = subarray_sum_to_k(test[0], test[1])
    assert result == test[2],f"actual: {result}; expected: {test[2]}"


tests = [
    [[4, 2, 1], 3, 3],
    [[4, 2, 1], 2, 1],
    [[3, 4, 5], 0, 0],
    [[10], 3, 0],
    [[10], 10, 0],
    [[10], 11, 1],
    [[10, 5, 2, 6], 100, 8],
    [[10, 200, 2, 6], 100, 4],
]

for test in tests:
    result = find_products_beneath_limit(test[0], test[1])
    assert result == test[2],f"actual: {result}; expected: {test[2]}"


# def sliding_average(nums, span, threshold):
#     pass
#
# def find_indices(word, pattern):
#     pass
#
# def min_window_substring(word, pattern):
#     pass
#
# def max_window_no_duplicates(phrase):
#     pass
#
# def rolling_sum_of_time_windows(logs, second_range):
#     pass
#
#
# def subarray_sum_to_k(nums, target):
#     pass
#
# def find_products_beneath_limit(nums, limit):
#     pass