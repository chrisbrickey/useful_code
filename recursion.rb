def factorial(n)
  return n if n == 1
  n * factorial(n - 1)
end

# p factorial(4)


def reverse(str)
  return str if str.length <= 1
  str[-1] + reverse(str[0...-1])
end

# p reverse("hello")


def nth_row_of_pascal(n)
  current_row = [1]
  return [1] if n == 0

  prev_row = nth_row_of_pascal(n - 1)
  (prev_row.length - 1).times do |index|
    current_row << prev_row[index] + prev_row[index + 1]
  end
  current_row << 1
end

# p nth_row_of_pascal(1)
# p nth_row_of_pascal(2)
# p nth_row_of_pascal(3)
# p nth_row_of_pascal(4)
# p nth_row_of_pascal(5)


def fibonacci(n)
  starter = [1, 1]
  return starter.take(n) if n <= 2
  prev_fibs = fibonacci(n - 1)
  next_num = prev_fibs[-1] + prev_fibs[-2]
  prev_fibs << next_num
end

# p fibonacci(1)
# p fibonacci(2)
# p fibonacci(8)

def range(min, max)
  return [] if max < min
  range(min, max - 1) << max
end

# p range(3, 10)


def sumto(arr)
  return arr[0] if arr.length <= 1
  sumto(arr[0...-1]) + arr[-1]
end

# p sumto([7, 2, 5])


def exp(base, exp)
  return 1 if exp == 0
  base * base**(exp - 1)
end

# p exp(4, 2)
# p exp(2, 5)
# p exp(100, 0)


def exp_next_level(base, exp)
  return 1 if exp == 0
  if exp.even?
    exp_next_level(base, exp - 1)**2
    # exp_next_level(base, exp / 2)**2
  else
    base * exp_next_level(base, (exp - 1) / 2)**2
  end

end
# p exp_next_level(4, 2)
# p exp_next_level(2, 5)
# p exp_next_level(100, 0)


def deep_dup(arr)
  return [arr[-1]] if arr.length <= 1
  deep_dup(arr[0...-1]) << arr[-1]
end

my_arr = [9, 10, 11, [0, [4, [88, 9]]], [0, 7, [9, [0, 1], 3]]]
duplicate = deep_dup(my_arr)
my_arr << "hello"
# p my_arr
# p duplicate


def deep_dup_inject(arr)
  return [arr[-1]] if arr.length <= 1

  arr.inject([]) do |accumulator, element|
    if element.is_a?(Array)
      deep_dup_inject(element)
    else
      accumulator << element
    end
  end

  accumulator
end

my_arr = [9, 10, 11, [0, [4, [88, 9]]], [0, 7, [9, [0, 1], 3]]]
duplicate = deep_dup(my_arr)
my_arr << "hello"
# p my_arr
# p duplicate


def subsets(arr)
  return [[]] if arr.empty?
  old = subsets(arr[0...-1])
  old + old.map { |subarray| subarray + [arr[-1]] }
end

# p subsets([3, 4, 5])


def permutations(arr)
  return [arr] if arr.length <= 1
  total_perms = []

  arr.each_index do |index|
    # pivot = arr[index]
    left = arr.take(index)
    right = arr.drop(index + 1)
    subarray = left + right

    total_perms += permutations(subarray).map do |element|
      element << arr[index]
    end
  end
  total_perms
end

p permutations([1, 2, 3])
