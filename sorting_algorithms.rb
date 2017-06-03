require 'byebug'

my_arr = [6, 9, 3, 67, 0, 12, 5, 8, 3]

# def bubble_sort_short!(arr, &prc)
#   # debugger
#   return arr if arr.length <= 1
#   prc ||= Proc.new { |x, y| y <=> x }
#
#
#   # arr.map! { |x, y| prc.call(x, y) }
# end
#
# p bubble_sort_short!(my_arr)
# p bubble_sort_short!(my_arr) { |x, y| x <=> y }





def bubble_sort_long! (arr)
  length = arr.length
  return arr if length <= 1



  loop do
    swapped = false

    0.upto(length - 2) do |idx|
      if arr[idx] > arr[idx + 1]
        arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
        swapped = true
      end
    end
    break if swapped == false
  end

  arr
end


p bubble_sort_long!(my_arr)
