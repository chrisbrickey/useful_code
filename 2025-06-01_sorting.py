# SORTING PRACTICE

# USE TOP 3 IF ALL YOU CARE ABOUT IS SPACE: O(1) because pointers
# ....but time: O(N^2)

# Looks at entire array and moves smaller numbers to index0 (until smallest has been moved there). Then looks at entire array (except index0) and moves smallest to index1. Repeats until comes to end of array.
# Time: O(N^2) always; Space: O(1) using pointers
def selection_sort(l):
    for i in range(0, len(l) - 1):
        for j in range(i, len(l)):
            if l[i] > l[j]:
                l[i], l[j] = l[j], l[i]

    return l

# Moves from left to right looking at consecutive numbers and swapping. After a swap it backtracks to swap all the way back toward the 0th index (as needed) to insure order behind the outer loop.
# Time: O(N^2) worst case; Space: O(1) using pointers
def insertion_sort(l):
    for i in range(0, len(l) - 1):
        while i >= 0 and l[i] > l[i + 1]:
            l[i], l[i+1] = l[i+1], l[i]
            i -= 1

    return l

# Use window that is 2 items long. Traverse array making swaps over and over until no more swapping required.
# Time: O(N^2); This only reduces to O(N) if the array is already sorted
# Space: O(1) using pointers
def bubble_sort(l):
    for i in range(len(l)):
        swapped = False

        for j in range(len(l) - 1 - i):
            if l[j] > l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
                swapped = True

        if not swapped:
            break

    return l

# ALWAYS NlogN time but N space
# Split the list. Recursively sort left and right sides. After recursion, merge the left and right side in an iterative sub-process.
# Time: O(N*logN); There are logN recursive calls. At each stack frame, every element has to be visited (by the compare_merge helper). So logN * N = NlogN.
# Space: O(N); Each element eventually is held in a new array in the compare_merge helper function. This dominates the number of stack frames (only logN). Technically it is N + logN.
def merge_sort(l, pivot):
    if len(l) < 2:
        return l

    # include pivot
    left, right = l[0:pivot], l[pivot:]
    sorted_left = merge_sort(left, len(left) // 2)
    sorted_right = merge_sort(right, len(right) // 2)
    return merge_two(sorted_left, sorted_right)

def merge_two(l1, l2):
    merged = []

    while l1 and l2:
        if l1[0] < l2[0]:
            merged.append(l1.pop(0))
        else:
            merged.append(l2.pop(0))

    if l1:
        merged.extend(l1)
    elif l2:
        merged.extend(l2)

    return merged

# Time: N^2 worst case (NlogN avg) BUT... Space: logN (stack frames)
# Choose index as the pivot. Rearrange the array so that elements less than the pivot are on the left and elements greater are on the right. Recursion: Make sub-arrays by repeating the above steps. Combine the sorted sub-arrays.
# Time: O(N^2) worst case (sorted arr); Otherwise this generally runs O(N*logN).
# Space: O(logN) b/c stack frames IF we did this with swapping in place  (instead of creating new arrays); On average we cutting the array in half for each recursive call.
# Actual Space of below: O(N) because not swapping in place. All elements stored in supplemental array. O(N + logN) = O(N)
def quick_sort(l):
    if len(l) < 2:
        return l

    # pivot excluded
    pivot_value = l[0]
    left, right = [], []
    for num in l[1:]:
        if num < pivot_value:
            left.append(num)
        else:
            right.append(num)

    return quick_sort(left) + [pivot_value] + quick_sort(right)

# !! Can only be used when the range of elements in the array is small (e.g. 0-10_000)
# Time: O(N) even in worst case scenario; super efficient but range of numbers must be small
# Space: O(range of elements) because we store a count in an array for each unique element
def bucket_sort(l, min, max):
    span = max - min + 1
    buckets = [0] * span

    for num in l:
        bucket_index = num - min
        buckets[bucket_index] += 1

    insertion_index = 0
    for b_index, count in enumerate(buckets):
        value = b_index + min
        for _ in range(count):
            l[insertion_index] = value
            insertion_index += 1

    return l

# HEAPSORT (if use heapify to build from array: Time: O(N) ; Space: O(1) because python implements with swapping in place)
# import heapq
# heapq.heapify(my_list); Time: O(N), Space: O(1); This only works when you feed heapify an existing array; subsequent additions to the heap take logN for every element added
# my_list is now a min heap; index 0 and len(my_list) work as expected (no dummy index)
# to make a max heap, negate numbers on the way in and on the way out
#
# If we built manually as a binary tree with sorted property:
#     build: O(NlogN); logN for each insertion assuming you add every node; if you only keep the top k elements in the heap then NlogK to build which reduces to N if K is constant
#     index 0 is a dummy index! start at 1
#     left_child, right_child = 2*parent, 2*parent + 1
#     parent = child // 2

# heapq.heappop(my_list) removes minimum from the heap;  O(logN) b/c always balanced and only need to work up or down the height of the tree once
# heapq.heappush(my_list, element_to_add); O(logN) b/c always balanced and only need to work up or down the height of the tree once
# heapq.nsmallest(my_list, k) returns list of kth smallest numbers; O(k*logN)
# heapq.nlargest(my_list, k) returns list of kth largest numbers; O(k*logN)

# Time: O(NlogN) because sorting first
# Space: O(N) worst case all intervals included in results
def merge_intervals(intervals):
    intervals.sort(key = lambda x: x[0])
    merged = [intervals[0]]

    # [[[1, 4], [4, 5]], [[1, 5]]],
    for current in intervals[1:]:
        last_end = merged[-1][1]
        current_start, current_end = current[0], current[1]

        if current_start <= last_end:
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])

    return merged

# Time: O(N); Space: O(1)
def trap_water(heights):
    total_water = 0
    if not heights:
        return total_water

    left, right = 0, len(heights) - 1
    max_left, max_right = heights[left], heights[right]

    # from lower side, advance pointer, calculate new max on that side
    while left < right:
        if heights[left] < heights[right]:
            left += 1
            max_left = max(heights[left], max_left)

            # This will never be negative b/c recalculated max above to consider current position
            total_water += max_left - heights[left]

        else:
            right -= 1
            max_right = max(heights[right], max_right)

            # This will never be negative b/c recalculated max above to consider current position
            total_water += max_right - heights[right]

    return total_water

# days until a lower number is reached (equal ok) BEHIND
def create_flashback(nums):
    results = [0] * len(nums)

    stack = []
    for i in range(len(nums) - 1, -1, -1):
        while stack and nums[i] < nums[stack[-1]]:
            popped_index = stack.pop()
            day_diff = popped_index - i # moving from high to low indices
            results[popped_index]  = day_diff

        stack.append(i)
    return results

# days until a lower number is reached (equal ok) AHEAD
def create_flashforward(nums):
    results = [0] * len(nums)

    stack = []
    for i, num in enumerate(nums):
        while stack and num < nums[stack[-1]]:
            popped_index = stack.pop()
            day_diff = i - popped_index # moving from low to high indices
            results[popped_index] = day_diff

        stack.append(i)
    return results

# Time: O(N): Iterating 3 times but sequential. Operations within all three loops are constant or inner loops are limited.
# Space: O(N): Creating about 5 data structures of size N
def rob_bank(guards, span):
    flashback = create_flashback(guards)
    flashforward = create_flashforward(guards)

    good_days = []
    for i, count in enumerate(guards):
        # left_span, right_span = span, span

        # bound check on i
        if i - span < 0 or i + span >= len(guards):
            continue
        # if i - span < 0:
        #     left_span = i
        # if i + span >= len(guards):
        #     right_span = len(guards) - 1 - i

        # BUT... it's ok to look out of bounds on flashback and flashforward.
        backward, forward = flashback[i], flashforward[i]
        if (forward == 0 or forward >= span) and (backward == 0 or backward >= span):
            good_days.append(i)

    return good_days


# Given an array of unique strings, return the pairs of indices indicating the possible combinations that make palindromes
# Time: O(N × K²); N = length of words array; K = average length of a word
# Space: O(N^2) worst case if every pair forms a palindrome
def find_palindromes(phrases):
    store = {word: i for i, word in enumerate(phrases)}

    results  = set()
    for j, word in enumerate(phrases):
        reverse = word[::-1]
        if reverse in store and store[reverse] != j:
            results.add((j, store[reverse]))

        for k in range(len(word)):
            prefix, suffix = word[0:k], word[k:]
            reverse_prefix, reverse_suffix = prefix[::-1], suffix[::-1]

            if reverse_prefix in store and store[reverse_prefix] != j:
                results.add((j, store[reverse_prefix]))

            if reverse_suffix in store and store[reverse_suffix] != j:
                results.add((store[reverse_suffix], j))

    return list(list(pair) for pair in results)

if __name__ == "__main__":
    test = [9, -200, 1000, 0, -8, 2]
    expected = [-200, -8, 0, 2, 9, 1000]
    sort_functions = [
        selection_sort,
        insertion_sort,
        bubble_sort,
        merge_sort,
        quick_sort,
        bucket_sort,
    ]

    for f in sort_functions:
        print(f"Testing {f.__name__}")
        fresh_test = test.copy()  # avoids reusing an already sorted array
        if f == merge_sort:
            result = f(fresh_test, len(fresh_test) // 2)  # requires split_index to start
        elif f == bucket_sort:
            result = f(fresh_test, min(fresh_test), max(fresh_test))
        else:
            result = f(fresh_test)
        assert result == expected, f"actual; {result}, expected: {expected}"

tests = [
    [[[0, 5]], [[0, 5]]],
    [[[1, 6], [7, 9]], [[1, 6], [7, 9]]],
    [[[1, 4], [4, 5]], [[1, 5]]],
    [[[1, 6], [5, 9]], [[1, 9]]],
    [[[1,3],[2,6],[8,10],[15,18]], [[1,6],[8,10],[15,18]]],
    [[[15,18],[1,3],[8,10],[2,6]], [[1,6],[8,10],[15,18]]], # not sorted
    [[[0, 6], [0, 6], [1, 2], [2, 3], [3, 8], [4, 5], [4, 6], [4, 7], [5, 6]], [[0, 8]]], # multiple inner intervals
]

for test in tests:
    result = merge_intervals(test[0])
    assert result == test[1],f"actual: {result}; expected: {test[1]}"

tests = [
    [[1, 0, 1], 1],
    [[0, 2, 0, 3, 1, 0, 1, 3, 2, 1], 9],
    [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6],
    [[0, 1, 2, 3, 2, 1], 0],
    [[6, 6, 6, 6, 6, 6], 0],
]

for test in tests:
    result = trap_water(test[0])
    assert result == test[1], f"actual: {result}; expected: {test[1]}"

tests = [
    [[1, 1, 1], 1, [1]],
    [[1, 1, 0, 1, 1], 2, [2]],
    [[5, 3, 3, 3, 5, 6, 2], 2, [2, 3]],
    [[1, 1, 1, 1, 1], 0, [0, 1, 2, 3, 4]],
    [[1, 2, 3, 4, 5, 6], 2, []],
]

for test in tests:
    result = rob_bank(test[0], test[1])
    assert result == test[2],f"actual: {result}; expected: {test[2]}"


# Constraint: no duplicates! Allows us to avoid double loop.
palindrome_tests = [
    [['x', 'z'], []],
    [['xxxxxxxx', 'x'], [[0, 1], [1, 0]]],
    [['ab', 'a'], [[0, 1]]],
    [['ab', 'ba'], [[0, 1], [1, 0]]],
    [['lls', 's'], [[1, 0]]],
    [['lls', 's', 'xox'], [[1, 0]]],
]

for test in palindrome_tests:
    result = find_palindromes(test[0])
    assert result == test[1],f"actual: {result}, expected: {test[1]}"


# def selection_sort(l):
#     return l

# def insertion_sort(l):
#     return l

# def bubble_sort(l):
#     return l

# def merge_sort(l, pivot):
#     return l

# def quick_sort(l):
#     return l

# def bucket_sort(l):
#     return l

# HEAPSORT (if use heapify to build from array: Time: ; Space:  because...)
#
# def merge_intervals(intervals):
#     pass
#
# def trap_water(heights):
#     pass
#
#
# def rob_bank(guards, span):
#     pass
#
# Given an array of unique strings, return the pairs of indices indicating the possible combinations that make palindromes
# def find_palindromes(phrases):
#   pass