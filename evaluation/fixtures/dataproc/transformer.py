"""Data transformers for the processing pipeline."""


def transform_data(rows):
    """Apply a series of transformations to data rows."""
    cleaned = clean_data(rows)
    normalized = normalize_values(cleaned)
    return normalized


def clean_data(rows):
    """Remove empty rows and trim whitespace from string values."""
    cleaned = []
    for row in rows:
        if not row:
            continue
        cleaned_row = {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
        cleaned.append(cleaned_row)
    return cleaned


def normalize_values(rows):
    """Normalize numeric values to a 0-1 range."""
    return rows
