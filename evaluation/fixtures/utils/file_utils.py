"""File operation utilities."""


import os


def read_file(path):
    """Read the contents of a file and return as a string."""
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)


def list_files(directory):
    """List all files in a directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def file_exists(path):
    """Check whether a file exists at the given path."""
    return os.path.exists(path)
