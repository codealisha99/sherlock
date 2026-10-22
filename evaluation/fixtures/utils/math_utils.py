"""Mathematical helper functions."""


def average(numbers):
    """Return the arithmetic mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def clamp(value, min_value, max_value):
    """Clamp a value to the specified range."""
    return max(min_value, min(value, max_value))


def is_even(number):
    """Return True if the number is even."""
    return number % 2 == 0
