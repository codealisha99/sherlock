"""Calculator module with basic arithmetic operations."""


class Calculator:
    """A simple calculator supporting basic arithmetic."""

    def add(self, a, b):
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a, b):
        """Return the difference of a and b."""
        return a - b

    def multiply(self, a, b):
        """Return the product of a and b."""
        return a * b

    def divide(self, a, b):
        """Return the quotient of a and b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
