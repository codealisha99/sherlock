"""Pipeline orchestration for data processing."""


from .reader import read_csv, read_json, read_text
from .transformer import transform_data
from .writer import write_csv, write_json, write_text


def run_pipeline(input_path, output_path, output_format="csv"):
    """Run the full data processing pipeline from input to output."""
    data = load_data(input_path)
    transformed = transform_data(data)
    save_data(transformed, output_path, output_format)
    return transformed


def load_data(path):
    """Load data from a file based on its extension."""
    if path.endswith(".csv"):
        return read_csv(path)
    if path.endswith(".json"):
        return read_json(path)
    return read_text(path)


def save_data(data, path, output_format="csv"):
    """Save transformed data to a file in the specified format."""
    if output_format == "csv":
        write_csv(path, data)
    elif output_format == "json":
        write_json(path, data)
    else:
        write_text(path, str(data))
