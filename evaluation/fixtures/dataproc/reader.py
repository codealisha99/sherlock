"""File readers for the data processing pipeline."""


import csv
import json


def read_csv(path):
    """Read a CSV file and return rows as dictionaries."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_json(path):
    """Read a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def read_text(path):
    """Read a text file and return its contents as a string."""
    with open(path, "r") as f:
        return f.read()
