def factorial(n)
  return n if n == 1
  n * factorial(n - 1)
end

p factorial(4)

def reverse(str)
  return str if str.length <= 1
  str[-1] + reverse(str[0...-1])
end

p reverse("hello")


def nth_row_of_pascal(n)
  current_row = [1]
  return [1] if n == 0
  prev_row = nth_row_of_pascal(n - 1)

  (prev_row.length - 1).times do |index|
    current_row << prev_row[index] + prev_row[index + 1]
  end
  current_row << 1
end

p nth_row_of_pascal(1)
p nth_row_of_pascal(2)
p nth_row_of_pascal(3)
p nth_row_of_pascal(4)
p nth_row_of_pascal(5)
