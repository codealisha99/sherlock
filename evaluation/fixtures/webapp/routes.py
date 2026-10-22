"""Route handlers for the web application."""

from .models import User, Post


def register_routes(app):
    """Register all URL routes with the Flask app."""
    app.add_url_rule("/", "index", index_handler)
    app.add_url_rule("/users/<int:user_id>", "user", user_handler)


def index_handler():
    """Handle the root URL."""
    return "Hello, World!"


def user_handler(user_id):
    """Handle a user profile request."""
    user = User.query.get(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    return user.name
