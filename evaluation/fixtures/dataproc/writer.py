"""Data writers for the processing pipeline."""


import csv
import json


def write_csv(path, rows, fieldnames=None):
    """Write rows to a CSV file."""
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Write data to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_text(path, content):
    """Write text content to a file."""
    with open(path, "w") as f:
        f.write(content)
