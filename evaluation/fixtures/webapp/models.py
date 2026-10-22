"""Database models for the web application."""


class User:
    """A user account in the system."""

    def __init__(self, user_id, name, email):
        self.user_id = user_id
        self.name = name
        self.email = email

    @classmethod
    def query(cls, filter=None):
        """Query the database for users."""
        return []


class Post:
    """A blog post written by a user."""

    def __init__(self, post_id, title, content, author_id):
        self.post_id = post_id
        self.title = title
        self.content = content
        self.author_id = author_id
