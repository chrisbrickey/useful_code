# TREE PRACTICE

import os, sys
path_to_trees = os.path.expanduser("insert file path here")
sys.path.insert(0, path_to_trees)

try:
    from binary_search_tree import BST
except ModuleNotFoundError as e:
    print(f"Error importing module: {e}")

class Node:
    def __init__(self, val):
        self.value = val
        self.left, self.right = None, None

from collections import deque

# Time: O(N) processes every node
# Space: O(N) q holds all leaves at the last layer; if tree is full then leaves are on the order of N (power series)
def count_layers(node):
    layers = 0

    if not node:
        return layers

    q = deque()
    q.append(node)
    while q:
        layers += 1

        for _ in range(len(q)):
            current = q.popleft()

            if current.left:
                q.append(current.left)
            if current.right:
                q.append(current.right)

    return layers

# Time: O(N) processes every node
# Space: O(N); 2N because results array holds on the order of N (all leaves) and stack holds height of tree which is N if tree is a line
def path_sums_iterative(node):
    sums = []
    stack = [(node, node.value)]
    while stack:
        current, current_sum = stack.pop()

        if not current.left and not current.right:
            sums.append(current_sum)

        if current.right:
            new_sum = current_sum + current.right.value
            stack.append((current.right, new_sum))
        if current.left:
            new_sum = current_sum + current.left.value
            stack.append((current.left, new_sum))

    return sums

def largest_path_iterative(node):
    largest = float('-inf')

    stack = [(node, node.value)]
    while stack:
        current, current_sum = stack.pop()

        if not current.left and not current.right:
            largest = max(largest, current_sum)

        if current.right:
            new_sum = current_sum + current.right.value
            stack.append((current.right, new_sum))
        if current.left:
            new_sum = current_sum + current.left.value
            stack.append((current.left, new_sum))

    return largest

# DFS Recursive
# Time: O(N) processes every node
# Space: O(N); actually 2N
#   stack frames: O(N) worst case if tree is a line; O(logN) if the tree is balanced
#   results array: O(N) number of leaves (sums) is on the order of N elements if the tree is full (power series)
def path_sums_recursive(node, current_sum):
    if not node:
        return []

    current_sum += node.value
    if not node.left and not node.right:
        return [current_sum]

    lefts, rights = [], []
    if node.left:
        lefts.extend(path_sums_recursive(node.left, current_sum))
    if node.right:
        rights.extend(path_sums_recursive(node.right, current_sum))

    return lefts + rights

def largest_path_recursive(node, current_sum, largest_sum):
    if not node:
        return largest_sum

    current_sum += node.value
    if not node.left and not node.right:
        if not largest_sum or current_sum > largest_sum:
            largest_sum = current_sum
        return largest_sum

    left_largest, right_largest = None, None
    if node.left:
        left_largest = largest_path_recursive(node.left, current_sum, largest_sum)
    if node.right:
        right_largest = largest_path_recursive(node.right, current_sum, largest_sum)

    if not left_largest:
        return right_largest
    elif not right_largest:
        return left_largest
    else: # both exist
        return max(left_largest, right_largest)

# DFS RECURSIVE (variation on the subset generation algorithm)
# Given an array of ints and a target, return subarrays of all possible combinations of numbers from the input that will sum to the target.
# Time: O(2^target) because two recursive calls on every iteration Time: (branching factor ^ height of tree)
# The height of tree is roughly target because we are adding individual elements (which could be as small as 2) to see what sums to target.
# Space: O(2^N) result might store all possible combinations
def combination_sum(nums, target):
    results = []

    def dfs(i, interim_sum, interim_sub):
        if interim_sum == target:
            results.append(interim_sub.copy())
            return

        if i >= len(nums) or interim_sum > target:
            return

        # exclude current element (at position i)
        dfs(i + 1, interim_sum, interim_sub.copy())

        # include current element; keep including until sum exceeds target
        interim_sum += nums[i]
        interim_sub.append(nums[i])
        dfs(i, interim_sum, interim_sub.copy())

    dfs(0, 0, []) # populates results
    return results

# Time: O(logN) if AVL tree b/c discarding half the nodes on every call; O(N) if tree is a line
# Space: O(logN) if AVL tree b/c descending height of tree; O(N) if tree is a line
def binary_search_nodes(node, target):
    if not node:
        return False

    if target < node.value:
        return binary_search_nodes(node.left, target)
    elif target > node.value:
        return binary_search_nodes(node.right, target)
    else:
        return True

# Time: O(N) processes every node; Space: O(N) actually 2N for result and stack arrays
# in-order (processing action in between the two recursive calls); lefts to node/parent to rights
# pre-order (processing action first... before all recursive calls); node/parent to lefts to rights
# post-order (processing action after all recursive calls); lefts to rights to node/parent
def inorder_iterative(node):
    results, stack = [], []
    while node or stack:

        # descend left side adding to stack
        while node:
            stack.append(node)
            node = node.left

        # node is None here
        current = stack.pop()
        results.append(current.value)

        # start descending right side
        node = current.right

    return results

# Time: O(N) process every node
# Space: O(N) N for results array; + logN if AVL tree OR + N if tree is a line
def inorder_recursive(node):
    results = []

    def dfs(current):
        if not current:
            return

        dfs(current.left)
        results.append(current.value)
        dfs(current.right)

    dfs(node)
    return results

# Time: O(logN) always halving the size of the input on every loop
# Space: O(1) using pointers without recursion
def binary_search_list(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if target < nums[mid]:
            right = mid - 1
        elif target > nums[mid]:
            left = mid + 1
        else:
            return True
    return False

def find_first_bad_version(highest):
    left, right = 0, highest
    last_good_version, last_bad_version = -1, highest + 1 # out of bounds

    while left <= right:
        guess = (left + right) // 2
        if is_good(guess):
            if guess == last_bad_version - 1:
                return last_bad_version
            last_good_version = guess
            left = guess + 1

        else: # is bad
            if guess == last_good_version + 1:
                return guess
            last_bad_version = guess
            right = guess - 1

def is_good(n):
    if n < FIRST_BAD_VERSION:
        return True
    return False

# Given an array of temperatures, return an array of the same length that indicates the number of days in the future we must wait until a day that is as warm or warmer than the current day.
# Monotonic Stack: For each element in the input array, return count of how many days we have to wait until a higher temperature
# Time: O(N); Space: O(N) actually 2N for stack and results
def daily_temperatures(temps):
    results = [0] * len(temps)
    stack = [] # indices; decreasing

    for i, num in enumerate(temps):
        while stack and num > temps[stack[-1]]:
            popped_index = stack.pop()
            day_diff = i - popped_index
            results[popped_index] = day_diff

        stack.append(i) # all cases
    return results

# Time: O(rows*cols) may process every cell
# Space: O(rows*cols); actually 2*rows*cols because visited and qs may hold all cells
def count_islands_bfs_or_dfs(grid, method='dfs'):
    count, visited = 0, set()
    ROWS, COLS = len(grid), len(grid[0])

    def explore_island(rr, cc):

        # !! You have to add to visited at same time as adding to stack
        # If you do this in different parts of the code then you may double count because the tuple may be in the stack but not yet in visited set.
        qs = deque()
        qs.append((rr, cc))
        visited.add((rr, cc))

        while qs:
            match method:
                case 'dfs':
                    cur_r, cur_c = qs.pop()
                case 'bfs':
                    cur_r, cur_c = qs.popleft()
                case _:
                    raise "Method not recognized"



            directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
            for pair in directions:
                new_r, new_c = cur_r + pair[0], cur_c + pair[1]
                in_bounds = new_r in range(ROWS) and new_c in range(COLS)
                if in_bounds and grid[new_r][new_c] == 1 and (new_r, new_c) not in visited:
                    # !! You have to add to visited at same time as adding to stack
                    # If you do this in different parts of the code then you may double count because the tuple may be in the stack but not yet in visited set.
                    qs.append((new_r, new_c))
                    visited.add((rr, cc))


    for r in range(ROWS):
       for c in range(COLS):
           if grid[r][c] == 1 and (r, c) not in visited:
               count += 1
               explore_island(r, c)

    return count

# NB: We don't need to keep track of a visited set at all if we are only allowed to move down and right.
# The visited set is only necessary when you can move in any direction (prevents cycles).
# Then... if you use a visited set AND you need paths to overlap, then you have to remove from the visited set after the recursive calls (bottom of DFS)
def count_path_on_graph_recursive(grid):
    ROWS, COLS = len(grid), len(grid[0])

    def dfs(r, c, visited):
        if (r, c) == (ROWS - 1, COLS - 1) and grid[r][c] == 0:
            return 1

        # variables for this path only
        path_count = 0
        visited.add((r, c))

        directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        for pair in directions:
            new_r, new_c = r + pair[0], c + pair[1]
            in_bounds = new_r in range(ROWS) and new_c in range(COLS)
            if in_bounds and grid[new_r][new_c] == 0 and (new_r, new_c) not in visited:
                path_count += dfs(new_r, new_c, visited.copy())

        # Remove current from path_visited so it can be reused on other paths
        visited.remove((r, c))

        return path_count

    return dfs(0, 0, set())

def count_path_on_graph_iterative(grid):
    ROWS, COLS = len(grid), len(grid[0])
    count = 0

    stack = [(0, 0, set())]
    while stack:
        r, c, visited = stack.pop()

        if (r, c) == (ROWS - 1, COLS - 1) and grid[r][c] == 0:
            count += 1

        # !! You have to add to visited at same time as adding to stack
        # If you do this in different parts of the code then you may double count because the tuple may be in the stack but not yet in visited set.
        visited.add((r, c))
        directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        for pair in directions:
            new_r, new_c = r + pair[0], c + pair[1]
            in_bounds = new_r in range(ROWS) and new_c in range(COLS)
            if in_bounds and grid[new_r][new_c] == 0 and (new_r, new_c) not in visited:
                # Each path downstream of current needs its own visited set to avoid cycles.
                # In the iterative solution, we never need to remove from visited because each path has it's own separate set.
                # We can't use a global visited set because paths overlap (we need to reuse cells)
                stack.append((new_r, new_c, visited.copy()))

    return count

# Time: O(N) loops N times
# Space: O(N) holding carries N elements; can be optimized by using pointers
def fib_iterative(n):
    if n < 3:
        return 1

    holding = [1, 1]
    for _ in range(n-2):
        new_sum = holding[-1] + holding[-2]
        holding.append(new_sum)

    return holding[-1]

# Time: O(2^N) two recursive calls on every loop; height is N because decrementing by only 1 on each call
# Space: O(N) N stack frames because decrementing by only 1 on each recursive call
def fib_recursive(n):
    if n < 3:
        return 1

    return fib_recursive(n-2) + fib_recursive(n-1)

# Time:O(N) memoization reduces recursive function to linear time; turns a decision tree into a decision line (height = N in this case because decrementing by 1 on each call)
# Space: O(N) N stack frames because decrementing by only 1
def fib_memoized(n, cache):
    if n < 3:
        return 1

    if n in cache:
        return cache[n]

    new_sum = fib_memoized(n-2, cache) + fib_memoized(n-1, cache)
    cache[n] = new_sum
    return cache[n]

# Time: O(N) looping N times
# Space: O(1) using pointers
def fib_tabulation(n):
    if n < 3:
        return 1

    prev_prev, prev = 1, 1
    for _ in range(n-2):
        new_sum = prev_prev + prev
        prev_prev = prev
        prev = new_sum

    return prev

def list_of_chars(phrase):
    punctuation = ''' ,;:!.?/'"\\-\n'''

    mutated = phrase
    for symbol in punctuation:
        mutated = mutated.replace(symbol, '')

    result = list(mutated)
    return result

def list_of_words(phrase):
    mutated = phrase
    punctuation = ''',;:!.?/'"\\-\n''' # excludes spaces
    for symbol in punctuation:
        mutated = mutated.replace(symbol, '')

    result = mutated.split()
    return result

def make_string(words):
    # Transform into string using spaces ' ' between the words
    mutated = " ".join(words)

    punctuation = ''',;:!.?/'"\\-\n'''
    for symbol in punctuation:
        mutated = mutated.replace(symbol, '')

    # Must remove spaces at the beginning or end, which we did not want to remove earlier
    result = mutated.strip()
    return result


# TESTS================================

#         1
#       /  \
#     8     -2
#    / \    / \
#   5   3  16   7
#  /        \
# -4         2
#
# [10,  12,   17,  6]

root = Node(1)
root.left = Node(8)
root.right = Node(-2)
root.left.left = Node(5)
root.left.left.left = Node(-4)
root.left.right = Node(3)
root.right.left = Node(16)
root.right.left.right = Node(2)
root.right.right = Node(7)

result_layers = count_layers(root)
expected_layers = 4
assert result_layers == expected_layers, f"actual: {result_layers}; expected: {expected_layers}"

result_sums = path_sums_iterative(root)
expected_sums = [10, 12, 17, 6]
assert result_sums == expected_sums, f"actual: {result_sums}; expected: {expected_sums}"
result_sums_recursive = path_sums_recursive(root, 0)
assert result_sums_recursive == expected_sums, f"actual: {result_sums_recursive}; expected: {expected_sums}"

result_largest = largest_path_iterative(root)
expected_largest = 17
assert result_largest == expected_largest, f"actual: {result_largest}; expected: {expected_largest}"
result_largest_recursive = largest_path_recursive(root, 0, None)
assert result_largest_recursive == expected_largest, f"actual: {result_largest_recursive}; expected: {expected_largest}"

combination_sum_tests = [
    [[2], 5, []],
    [[3], 5, []],
    [[4], 5, []],
    [[5], 5, [[5]]],
    [[3, 4], 5, []],
    [[2,5,6,9], 9, [[9], [2,2,5]]],
    [[3,4,5], 16, [[4,4,4,4], [3,4,4,5],[3,3,5,5], [3,3,3,3,4]]],
]

for test in combination_sum_tests:
    result = combination_sum(test[0], test[1])
    assert result == test[2],f"actual: {result}; expected: {test[2]}"

# Binary Search Tree Tests
bst1 = BST()
values = [10, 20, 5, 30, 40, 25]
for val in values:
    bst1.insert_value(val)

result1 = binary_search_nodes(bst1.root, 40)
expected1 = True
assert result1 == expected1,f"actual: {result1}, expected: {expected1}"

result2 = binary_search_nodes(bst1.root, 15)
expected2 = False
assert result2 == expected2,f"actual: {result2}, expected: {expected2}"

in_order_result = inorder_iterative(bst1.root)
in_order_result_recursive = inorder_recursive(bst1.root)
expected3 = [5, 10, 20, 25, 30, 40]
assert in_order_result == expected3,f"actual: {in_order_result}; expected: {expected3}"
assert in_order_result_recursive == expected3,f"actual: {in_order_result_recursive}; expected: {expected3}"

# Binary Search on List
nums = [-5, -2, 0, 1, 3, 100, 1000]
tests = [[1, True], [-5, True], [1000, True], [-8, False], [2, False], [10000, False]]
for test in tests:
    res = binary_search_list(nums, test[0])
    assert res == test[1],f"testing: {test[0]}; actual result: {res}"

bad_version_tests = [ # high end of range, first bad version, expected
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 2],
    [5, 4, 4],
]

for test in bad_version_tests:
    FIRST_BAD_VERSION = test[1]
    result = find_first_bad_version(test[0])
    assert result == test[2],f"actual: {result}; expected: {test[2]}"

# Monotonic stack problem
temperature_tests = [
    [[50], [0]],
    [[50, 50], [0, 0]],
    [[40, 50], [1, 0]],
    [[50, 40], [0, 0]],
    [[90, 60, 30], [0, 0, 0]],
    [[90, 60, 60, 30, 30], [0,0,0,0,0]],
    [[30, 60, 90], [1, 1, 0]],
    [[30,40,50,60], [1,1,1,0]],
    [[73,74,75,71,69,72,76,73],[1,1,4,2,1,1,0,0]],
]

for test in temperature_tests:
    result = daily_temperatures(test[0])
    assert result == test[1],f"actual: {result}, expected: {test[1]}"

# Matrix Tests
matrix = [
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1]]
expectation = 4

result_bfs = count_islands_bfs_or_dfs(matrix, "bfs")
assert result_bfs == expectation,f"actual: {result_bfs}, expected: 4"

result_dfs = count_islands_bfs_or_dfs(matrix, "dfs")
assert result_dfs == expectation,f"actual: {result_dfs}, expected: 4"


# No valid path
test0 = [[0, 1],
         [0, 1]]
result0 = count_path_on_graph_recursive(test0)
assert result0 == 0, f"actual: {result0}; expected: {0}"
result0 = count_path_on_graph_iterative(test0)
assert result0 == 0, f"actual: {result0}; expected: {0}"

# One valid path
test1 = [[0, 0, 0],
         [1, 1, 0]]
result1 = count_path_on_graph_recursive(test1)
assert result1 == 1, f"actual: {result1}; expected: {1}"
result1 = count_path_on_graph_iterative(test1)
assert result1 == 1, f"actual: {result1}; expected: {1}"

# Two valid paths
test2 = [[0, 0, 0, 0],
         [1, 0, 0, 1],
         [0, 1, 0, 1],
         [0, 1, 0, 0]]
result2 = count_path_on_graph_recursive(test2)
assert result2 == 2, f"actual: {result2}; expected: {2}"
result2 = count_path_on_graph_iterative(test2)
assert result2 == 2, f"actual: {result2}; expected: {2}"


# Fibonacci Tests
functions = [
    fib_iterative,
    fib_recursive,
    fib_memoized,
    fib_tabulation
]

fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
for f in functions:
    if f == fib_memoized:
        result = f(8, {})
    else:
        result = f(8)
    assert result == 21,f"actual: {result}; expected: 21"

result1 = list_of_chars('''asdf?   \n23!   \  ;''')
expected1 = ['a', 's', 'd', 'f', '2', '3']
assert result1 == expected1,f"actual: {result1}; exp: {expected1}"

result2 = list_of_words('  ;  as!df!    a   as?dfasdf   .  a  ?  ')
expected2 = ['asdf', 'a', 'asdfasdf', 'a']
assert result2 == expected2,f"actual: {result2}; exp: {expected2}"

result3 = make_string(['A-', 'cat', 'is', 'here!', ';'])
expected3 = 'A cat is here'
assert result3 == expected3,f"actual: {result3}; exp: {expected3}"

# def count_layers():
#     pass
#
# def path_sums_iterative():
#     pass
#
# def largest_path_iterative():
#     pass
#
# def path_sums_recursive():
#     pass
#
# def largest_path_recursive():
#     pass
#
# Given an array of ints and a target, return subarrays of all possible combinations of numbers from the input that will sum to the target.
# def combination_sum():
#     pass
#
# def binary_search_nodes():
#     pass
#
# def inorder_iterative():
#     pass
#
# def inorder_recursive():
#     pass
#
# def binary_search_list():
#     pass
#
# def find_first_bad_version():
#     pass
#
# Given an array of temperatures, return an array of the same length that indicates the number of days in the future we must wait until a day that is as warm or warmer than the current day.
# def daily_temperatures(temps):
#     pass
#
# def count_islands_bfs_or_dfs():
#    pass
#
# def count_path_on_graph_recursive():
#     pass
#
# def count_path_on_graph_iterative():
#     pass

# TODO: ADD MAKE LARGE ISLAND

# SPLIT HERE TO MAKE SEPARATE SHEET


# def fib_iterative():
#     pass
#
# def fib_recursive():
#     pass
#
# def fib_memoized():
#     pass
#
# def fib_tabulation():
#     pass
#
# def list_of_chars():
#     pass
#
# def list_of_words():
#     pass
#
# def make_string():
#     pass