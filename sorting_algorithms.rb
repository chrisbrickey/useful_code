require 'byebug'

my_arr = [6, 9, 3, 67, 0, 12, 5, 8, 3]

def bubble_sort_short! (arr, &prc)
  length = arr.length
  return arr if length <= 1
  prc ||= Proc.new { |num1, num2| num1 <=> num2 }

  loop do
    swapped = false

    0.upto(length - 2) do |idx|
      x = idx
      y = idx + 1
      spaceship_result = prc.call(arr[x], arr[y])
      if spaceship_result == 1
        arr[x], arr[y] = arr[y], arr[x]
        swapped = true
      end
    end
    break if swapped == false
  end

  arr
end

def bubble_sort_long! (arr)
  length = arr.length
  return arr if length <= 1

  loop do
    swapped = false

    0.upto(length - 2) do |idx|
      x = idx
      y = idx + 1
      if arr[x] > arr[y]
        arr[x], arr[y] = arr[y], arr[x]
        swapped = true
      end
    end
    break if swapped == false
  end

  arr
end

# p bubble_sort_short!(my_arr) #sort ascending
# p bubble_sort_short!(my_arr) { |num1, num2| num2 <=> num1 } #sort descending

def merge_sort(arr, &prc)
  # debugger
  return arr if arr.length <= 1
  mid = arr.length / 2
  left = arr.take(mid)
  right = arr.drop(mid)
  sorted_left = merge_sort(left)
  sorted_right = merge_sort(right)
  compare_and_merge(sorted_left, sorted_right, prc)
end

def compare_and_merge(left, right, prc)
  merged_arr = []

  until left.empty? || right.empty?
    prc ||= Proc.new { |x, y| [x, y].min }
    element_that_should_come_first = prc.call(left[0], right[0])
    if element_that_should_come_first == left[0]
      merged_arr << left.shift
    else
      merged_arr << right.shift
    end
  end

  merged_arr + left + right
end

p merge_sort(my_arr)
