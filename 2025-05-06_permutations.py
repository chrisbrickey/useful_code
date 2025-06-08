# PERMUTATIONS PRACTICE

# DOES NOT HANDLE DUPLICATE VALUES IN INPUT ARRAY!!
# iterate over every element; for each subset already in results: append that element
# Time: O(N*2^N); Outer loop = N; Inner loop changes. By the end of the problem the inner loop is as large as the final collection of all subsets: (2^N); You could use sets and tuples (instead of arrays and subarrays) to prevent duplicates but that would increase time complexity to N^2*2^N. There is a better way.
# Space: O(N*2^N); results array holds all the possible subsets; There are 2^N subsets and each subset can be up to size N (though the average size is N/2)
# Logic: Why are there 2^N subsets? Decision tree: As you work through each element in the array (height of tree) you decide whether or not to add that element to the in-process subset. That's 2 choices at each point in the decision tree: branching factor = 2, height of tree = N, Therefore O(2^N) decisions/options.
def find_subsets_iterative(nums):
    results = [[]]
    for num in nums:
        holding = []
        for result_sub in results:
            interim_sub = result_sub.copy()
            interim_sub.append(num)
            holding.append(interim_sub)

        results.extend(holding)
    return results

# DOES NOT HANDLE DUPLICATE VALUES IN INPUT ARRAY!!
# Each path through the decision tree (one line of dfs) represents a consideration of every index - a yes/no decision on whether or not to include it.
# Time: O(2^N) The function does O(1) work for each subset (add or don't add an element). So in the end, only the number of subsets generated (2^N) drives the time complexity.
# Space: O(N*2^N); results array holds all the possible subsets and dominates the space consumption; There are 2^N subsets and each subset can be up to size N (though the average size is N/2)
# .... if we disregard the size of the results array, there are two remaining considerations:
#         a) O(N): the number of stack frames used is N (height of tree)
#         b) O(N): the size of the interim arrays (at a given point in time) can only be of size N or smaller
def find_subsets_recursive(nums):
    results = []

    def dfs(i, interim_sub):
        if i >= len(nums):
            results.append(interim_sub.copy())
            return

        interim_sub.append(nums[i])
        dfs(i + 1, interim_sub.copy())

        interim_sub.pop()
        dfs(i + 1, interim_sub.copy())

    dfs(0, [])
    return results

#  HANDLES DUPLICATE ITEMS IN ARRAY by sorting and counting duplicate elements and then using an additional inner loop that loops for each duplicate value
#  Time: O(N*2^N); 2^N work because generating 2^N subsets at most. Every loop we copy the interim_subset which grows to the size of input array (so we multiply by N).
#       There is an additional NlogN for sorting the array but that term is dominated by N*2^N.
#  Space: O(N*2^N): Size of results array (2^N) * size of the subarrays in those results (largest will be size N)
def find_all_subsets_deduped_iterative(nums):
    nums.sort()
    results = [[]]

    i = 0
    while i < len(nums):

        # count duplicates within this loop, ok to reset at top of loop
        dup_count = 1
        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            dup_count += 1
            i += 1

        # i is sitting at end of duplicate span
        # Loop over the results that exist at this point without mutating underlying array
        holding = []
        for result_sub in results:
            interim_sub = result_sub.copy()
            inner_holding = []

            # repeat once for each duplicate
            # this loop continues to build up interim_sub and appends copies of it for each iteration
            for _ in range(dup_count):
                interim_sub.append(nums[i])
                inner_holding.append(interim_sub.copy())

            holding.extend(inner_holding)
        results.extend(holding)

        # move i past the span of duplicates
        i += 1
    return results



    return results

# Replicate stack frames with an array; track start_index and interim_combo in the tuples of the stack
# Time: O(combo_size*2^N) This algo does the work of finding all subsets regardless of size so the worst case time complexity is the number of all possible subsets: 2^N
# Space: O(2^combo_size) the space consumed by the results array
def find_combos_iterative(nums, combo_size):
    results = []

    stack = [(0, [])]
    while stack:
        i, interim_combo = stack.pop() # current_index, interim_combo

        # base case 1: interim combination is finally the correct length
        if len(interim_combo) == combo_size:
            results.append(interim_combo.copy())
            continue # skip logic below (we don't want it to grow larger)

        # base case 2: we are out of desired range so skip this branch but let stack play out
        if i >= len(nums):
            continue
            # must stop executing here to prevent index out of range below
            # nothing added to the stack so stack will expire eventually; no need to return

        # decision branch where current_val is EXCLUDED
        # To maintain LIFO, add this to the stack first so that it is processed second/after the branch where current_val is included.
        stack.append((i + 1, interim_combo.copy()))

        # decision branch where current_val is INCLUDED
        interim_combo.append(nums[i])
        stack.append((i + 1, interim_combo.copy()))

    return results

    # THIS WORKS BUT O(N*2N)
    # results = [[]]
    #
    # for num in nums:
    #     holding = []
    #     for result_sub in results:
    #         interim_sub = result_sub.copy()
    #         interim_sub.append(num)
    #         if len(interim_sub) == combo_size:
    #             holding.append(interim_sub)
    #
    #     results.extend(holding)
    # return results

# Time: O(K * possible_combos_of_k_size) = O(K * (N!/(K!(N-K)!))
# In a naive recursive solution, the time complexity is O(K*2^N) because it does as much work as generating all possible combos/subsets (regardless of length).
# BUT when we are generated combos of a given length we should be able to approach: O(K*possible_number_of_combos) which is smaller than (K*2^N).
#    The possible number of combinations of length K is:  (N!/(K!(N-K)!) which you do NOT need to know. But it is smaller than 2^N.
# Space: The results array dominates the space complexity: (N^combo_size).
def find_combos_recursive(nums, combo_size):
    results = []

    def dfs(i, interim_combo):
        if len(interim_combo) == combo_size:
            results.append(interim_combo.copy())
            return

        if i >= len(nums):
            return

        interim_combo.append(nums[i])
        dfs(i + 1, interim_combo.copy())

        interim_combo.pop()
        dfs(i + 1, interim_combo.copy())

    dfs(0, [])
    return results

# DOES NOT WORK IF DUPLICATES IN THE ARRAY; loops over every element in array; loops over every existing result array; inserts the number from input array into all possible indices
# Time: O(N^2 * N!); generally the number of nodes in the decision tree
#      O(N^2) because we must capture the work to build out each permutation; Every time we add an element we are inserting into the middle of an array (O(N)) operation. All subarrays are of length N so we do that O(N) operation N times. = N^2
#      O(N!) because there are always N! possible permutations and we must process each one of those.
# Space: O(N * N!) storage space for result: There are N! permutations and they are all of length N.
def find_perms_iterative(nums):
    results = [[]]

    for num in nums:
        augmented_results = []

        # insert current element at each possible location in interim_perm
        # insert_index extends one beyond the length of the base we are adding to in order to add the next number to the end
        for result_perm in results:
            inner_holding = []
            interim_perm = result_perm.copy()

            for insertion_index in range(len(interim_perm) + 1):
                growing_perm = interim_perm.copy()
                growing_perm.insert(insertion_index, num)
                inner_holding.append(growing_perm)

            augmented_results.extend(inner_holding)

            # reassignment, not extension. We had to start results with a starter structure [] but that is not part of the final collection
        results = augmented_results

    return results

# Time: O(N^2 * N!); generally the number of nodes in the decision tree
#      O(N^2) because we must capture the work to build out each permutation; Every time we add an element we are inserting into the middle of an array (O(N)) operation. All subarrays are of length N so we do that O(N) operation N times. = N^2
#      O(N!) because there are always N! possible permutations and we must process each one of those.
# Space: O(N * N!)
#       O(N) stack frames: We decrease the input by 1 on every recursive call (i + 1) so there are roughly N stack frames
#       O(N * N!) storage space for result: There are N! permutations and they are all of length N.
def find_perms_recursive(nums):
    results = []

    def dfs(i, interim_perm):
        if len(interim_perm) == len(nums):
            results.append(interim_perm.copy())
            return

        # if i >= len(nums):
        #     return

        for insertion_index in range(len(interim_perm) + 1):
            growing_perm = interim_perm.copy()
            growing_perm.insert(insertion_index, nums[i])
            dfs(i + 1, growing_perm.copy())

    dfs(0, [])
    return results

from collections import Counter

def find_perms_deduped_recursive(nums):
    store = Counter(nums)
    results = []

    def dfs(interim_perm):
        # base case 1: interim permutation is finally the correct length
        if len(interim_perm) == len(nums):
            results.append(interim_perm.copy())
            return

        # if i >= len(nums):
        #     return

        for num_to_insert, count in store.items():
            if count > 0:
                interim_perm.append(num_to_insert)

                # decrement count temporarily so not reusing the element
                store[num_to_insert] -= 1
                dfs(interim_perm.copy())

                # cleanup
                #   add the count back for the next loop
                #   remove the last element from interim permutations
                store[num_to_insert] += 1
                interim_perm.pop()

    dfs([])
    return results

# TESTS============================
subset_tests = [
[[10], [[], [10]]],
[[1, 2], [[], [1], [2], [1, 2]]],
[[1,2,3], [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]],
]

for test in subset_tests:
    iterative = find_subsets_iterative(test[0])
    iterative.sort()

    test[1].sort()
    sorted_expectation = test[1]
    assert iterative == sorted_expectation,f"actual: {iterative}; expected: {sorted_expectation}"

for test in subset_tests:
    recursive = find_subsets_recursive(test[0])
    recursive.sort()

    test[1].sort()
    sorted_expectation = test[1]
    assert recursive == sorted_expectation,f"actual: {recursive}; expected: {sorted_expectation}"


deduped_subsets = find_all_subsets_deduped_iterative([8, 5, 8])
deduped_subsets.sort()
expectation = [[], [5], [5, 8], [5, 8, 8], [8], [8, 8]]
expectation.sort()
assert deduped_subsets == expectation, f"actual: {deduped_subsets}; expected: {expectation}"

combo_tests = [ # nums, combo_size, expected_result
    [[5, 8, 3], 0, [[]]],
    [[1], 1, [[1]]],
    [[3, 2], 2, [[3, 2]]],
    [[4, 7, 2], 2, [[4, 7], [4, 2], [7, 2]]],
    [[4, 7, 2], 3, [[4, 7, 2]]],
    [[4, 1, 2, 3], 2, [[4, 1],[4, 2],[4, 3],[1, 2],[1, 3],[2, 3]]],
    [[9, 3, 1, 6, 5], 3, [[9, 3, 1], [9, 3, 6], [9, 3, 5], [9, 1, 6], [9, 1, 5], [9, 6, 5], [3, 1, 6], [3, 1, 5], [3, 6, 5], [1, 6, 5]]],
]

for test in combo_tests:
    i_result = find_combos_iterative(test[0], test[1])
    assert i_result == test[2],f"actual: {i_result}; expected: {test[2]}"

    r_result = find_combos_recursive(test[0], test[1])
    assert r_result == test[2],f"actual: {r_result}; expected: {test[2]}"

permutation_tests = [ # ! These tests do not cover arrays with duplicates.
    [[], [[]]],
    [[7], [[7]]],
    [[5, 6], [[5, 6], [6, 5]]],
    [[1, 2, 3], [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]],
]

for test in permutation_tests:
    test[1].sort()

    result_iterative = find_perms_iterative(test[0])
    result_iterative.sort()
    assert result_iterative == test[1],f"actual: {result_iterative}; expected: {test[1]}"

    result_recursive = find_perms_recursive(test[0])
    result_recursive.sort()
    assert result_recursive == test[1],f"actual: {result_recursive}; expected: {test[1]}"

permutation_tests_with_duplicates = [
    [[1,1,2], [[1,1,2], [1,2,1], [2,1,1]]],
    [[1,2,3], [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]],
]

for test in permutation_tests_with_duplicates:
    test[1].sort()

    result = find_perms_deduped_recursive(test[0])
    result.sort()
    assert result == test[1],f"actual: {result}; expected: {test[1]}"


# def find_subsets_iterative():
#     pass
#
#
# def find_subsets_recursive():
#     pass
#
#
# def find_all_subsets_deduped_iterative():
#     pass
#
#
# def find_combos_iterative():
#     pass
#
#
# def find_combos_recursive():
#     pass
#
#
# def find_perms_iterative():
#     pass
#
#
# def find_perms_recursive():
#     pass
#
#
# def find_perms_deduped_recursive():
#     pass