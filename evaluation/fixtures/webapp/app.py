"""Web application entry point."""

from .routes import register_routes
from .database import init_db


def create_app():
    """Create and configure the Flask application."""
    app = __import__("flask").Flask(__name__)
    app.config["DATABASE_URL"] = "sqlite:///app.db"
    app.config["SECRET_KEY"] = "dev-secret-key"
    register_routes(app)
    init_db()
    return app
