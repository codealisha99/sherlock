"""Database connection and session management."""


import sqlite3


def init_db():
    """Initialize the database and create tables."""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, title TEXT, content TEXT, author_id INTEGER)"
    )
    conn.commit()
    conn.close()


def get_connection():
    """Return a new database connection."""
    return sqlite3.connect("app.db")
