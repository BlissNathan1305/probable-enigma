def linear_equation(m, x, b):
    y = m * x + b
    return y

# Define the slope (m) and y-intercept (b)
m = 2
b = 3

# Define the x value
x = 4

# Calculate the y value
y = linear_equation(m, x, b)

print(f"The equation is: y = {m}x + {b}")
print(f"For x = {x}, y = {y}")
