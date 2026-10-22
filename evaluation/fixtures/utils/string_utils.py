"""String manipulation utilities."""


def slugify(text):
    """Convert a string to a URL-friendly slug."""
    return text.lower().strip().replace(" ", "-")


def capitalize_words(text):
    """Capitalize the first letter of each word."""
    return " ".join(word.capitalize() for word in text.split())


def truncate(text, max_length):
    """Truncate a string to the specified maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
