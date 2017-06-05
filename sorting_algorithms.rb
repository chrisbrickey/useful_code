require 'byebug'

my_arr = [6, 9, 3, 67, 0, 12, 5, 8, 3]


def bubble_sort! (arr)
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

# p bubble_sort!(my_arr) #sort ascending

def bubble_sort_with_proc! (arr, &prc)
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

# p bubble_sort_with_proc!(my_arr) #sort ascending
# p bubble_sort_with_proc!(my_arr) { |num1, num2| num2 <=> num1 } #sort descending

def merge_sort_with_spaceship(arr, &prc)
  prc ||= Proc.new { |x, y| x <=> y}
  return arr if arr.length <= 1
  mid = arr.length / 2
  left = arr.take(mid)
  right = arr.drop(mid)
  sorted_left = merge_sort_with_spaceship(left, &prc)
  sorted_right = merge_sort_with_spaceship(right, &prc)
  compare_and_merge(sorted_left, sorted_right, prc)
end

def compare_and_merge(left, right, prc)
  merged_arr = []

  until left.empty? || right.empty?
    spaceship_result = prc.call(left[0], right[0])
    if spaceship_result <= 0
      merged_arr << left.shift
    else
      merged_arr << right.shift
    end
  end

  merged_arr + left + right
end

# p merge_sort_with_spaceship(my_arr)
# p merge_sort_with_spaceship(my_arr) { |x, y| y <=> x }



def merge_sort_sans_spaceship(arr, &prc)
  prc ||= Proc.new { |x, y| [x, y].min }
  return arr if arr.length <= 1
  mid = arr.length / 2
  left = arr.take(mid)
  right = arr.drop(mid)
  sorted_left = merge_sort_sans_spaceship(left, &prc)
  sorted_right = merge_sort_sans_spaceship(right, &prc)
  compare_and_merge(sorted_left, sorted_right, prc)
end


def compare_and_merge(left, right, prc)
  merged_arr = []

  until left.empty? || right.empty?
    element_that_should_come_first = prc.call(left[0], right[0])
    if element_that_should_come_first == left[0]
      merged_arr << left.shift
    else
      merged_arr << right.shift
    end
  end

  merged_arr + left + right
end

# p merge_sort_sans_spaceship(my_arr)
# p merge_sort_sans_spaceship(my_arr) { |x, y| [x, y].max }


def quick_sort(arr)
  return [] if arr.length == 0
  pivot_point = arr[0]
  left = arr[1..-1].select { |el| el < pivot_point }
  right = arr[1..-1].select { |el| el >= pivot_point }
  quick_sort(left) + [pivot_point] + quick_sort(right)
end

# p quick_sort(my_arr)


#taking a proc doesn't make sense but included for fun
def binary_search(arr, target, &prc)
  return nil if arr.empty?
  prc ||= Proc.new {|x, y| x <=> y }
  probe = arr.length / 2

  result = prc.call(target, arr[probe])
  if result == -1
    binary_search(arr.take(probe), target, &prc)
  elsif result == 0
    probe
  else
    interim = binary_search(arr.drop(probe + 1), target, &prc)
    interim.nil? ? nil : probe + 1 + interim
  end
end

# p binary_search([1, 2, 3], 1) # => 0
# p binary_search([2, 3, 4, 5], 3) # => 1
# p binary_search([2, 4, 6, 8, 10], 6) # => 2
# p binary_search([1, 3, 4, 5, 9], 5) # => 3
# p binary_search([1, 2, 3, 4, 5, 6], 6) # => 5
# p binary_search([1, 2, 3, 4, 5, 6], 0) # => nil
# p binary_search([1, 2, 3, 4, 5, 7], 6) # => nil
# sorted_arr = my_arr.dup.sort
# p "sorted_arr: #{sorted_arr}"
# p binary_search(sorted_arr, 67) #3
