"""Generate the gold-standard benchmark dataset from synthetic fixtures.

Reads all .py files in evaluation/fixtures/, uses the Python AST to find
function and class definitions with exact line numbers, and generates a
JSON dataset of benchmark queries with independently authored relevance
labels.
"""

import ast
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
OUTPUT_PATH = Path(__file__).resolve().parent / "gold_standard.json"


def collect_fixture_files():
    """Return a sorted list of all .py files under evaluation/fixtures/."""
    files = sorted(FIXTURES_DIR.rglob("*.py"))
    return [f for f in files if f.name != "__init__.py"]


def extract_definitions(file_path):
    """Extract function and class definitions from a Python file.

    Returns a list of dicts with keys: name, kind, line, end_line.
    """
    source = file_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    definitions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(
                {
                    "name": node.name,
                    "kind": "function",
                    "line": node.lineno,
                    "end_line": node.end_lineno or node.lineno,
                }
            )
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.append(
                        {
                            "name": f"{node.name}.{item.name}",
                            "kind": "method",
                            "line": item.lineno,
                            "end_line": item.end_lineno or item.lineno,
                        }
                    )
            definitions.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "line": node.lineno,
                    "end_line": node.end_lineno or node.lineno,
                }
            )
    return definitions


def build_dataset():
    """Build the gold-standard dataset from fixture files.

    Each query is independently authored with gold-standard relevance labels
    based on the actual fixture source code.
    """
    fixture_files = collect_fixture_files()

    # Index all definitions by name for cross-referencing
    all_defs = []  # list of (repo, file_path, name, kind, line, end_line)
    for fpath in fixture_files:
        rel_path = fpath.relative_to(FIXTURES_DIR)
        repo = rel_path.parts[0]  # e.g., "calc", "webapp"
        file_rel = str(rel_path)
        defs = extract_definitions(fpath)
        for d in defs:
            all_defs.append(
                {
                    "repo": repo,
                    "file_path": file_rel,
                    "name": d["name"],
                    "kind": d["kind"],
                    "line": d["line"],
                    "end_line": d["end_line"],
                }
            )

    # Build lookup: name -> list of definitions
    name_index = {}
    for d in all_defs:
        name_index.setdefault(d["name"], []).append(d)

    # Also index by file_path
    file_index = {}
    for d in all_defs:
        file_index.setdefault(d["file_path"], []).append(d)

    queries = []

    def add_query(
        qid, query, description, category, repo, expected_files, expected_symbols,
        should_have_results, why_relevant
    ):
        queries.append({
            "query_id": qid,
            "query": query,
            "description": description,
            "category": category,
            "repo": repo,
            "expected_files": expected_files,
            "expected_symbols": expected_symbols,
            "should_have_results": should_have_results,
            "why_relevant": why_relevant,
        })

    # --- Exact identifier lookups ---
    add_query(
        "q001", "Calculator",
        "Exact class name lookup",
        "exact_identifier", "calc",
        ["calc/calculator.py"], ["Calculator"],
        True, "Exact class name match"
    )
    add_query(
        "q002", "add",
        "Exact function name lookup (appears in two files)",
        "exact_identifier", "calc",
        ["calc/calculator.py", "calc/operations.py"], ["add"],
        True, "Exact function name match in multiple files"
    )
    add_query(
        "q003", "read_csv",
        "Exact function name lookup",
        "exact_identifier", "dataproc",
        ["dataproc/reader.py"], ["read_csv"],
        True, "Exact function name match"
    )
    add_query(
        "q004", "slugify",
        "Exact function name lookup",
        "exact_identifier", "utils",
        ["utils/string_utils.py"], ["slugify"],
        True, "Exact function name match"
    )
    add_query(
        "q005", "write_json",
        "Exact function name lookup",
        "exact_identifier", "dataproc",
        ["dataproc/writer.py"], ["write_json"],
        True, "Exact function name match"
    )

    # --- Partial identifier lookups ---
    add_query(
        "q006", "calc",
        "Partial identifier match (substring of calc/calculator.py)",
        "partial_identifier", "calc",
        ["calc/calculator.py", "calc/operations.py"], [],
        True, "Partial identifier match in file path and content"
    )
    add_query(
        "q007", "transform",
        "Partial identifier match (substring of transform_data)",
        "partial_identifier", "dataproc",
        ["dataproc/transformer.py"], ["transform_data"],
        True, "Partial identifier match in function name"
    )
    add_query(
        "q008", "reader",
        "Partial identifier match (substring of read_csv/read_json/read_text)",
        "partial_identifier", "dataproc",
        ["dataproc/reader.py"], ["read_csv", "read_json", "read_text"],
        True, "Partial identifier match in multiple function names"
    )

    # --- Function/class discovery ---
    add_query(
        "q009", "def add",
        "Function definition keyword search",
        "function_discovery", "calc",
        ["calc/calculator.py", "calc/operations.py"], ["add"],
        True, "Keyword 'def' plus function name 'add'"
    )
    add_query(
        "q010", "class Calculator",
        "Class definition keyword search",
        "class_discovery", "calc",
        ["calc/calculator.py"], ["Calculator"],
        True, "Keyword 'class' plus class name 'Calculator'"
    )
    add_query(
        "q011", "create_app",
        "Function name lookup",
        "function_discovery", "webapp",
        ["webapp/app.py"], ["create_app"],
        True, "Exact function name match"
    )
    add_query(
        "q012", "run_pipeline",
        "Function name lookup",
        "function_discovery", "dataproc",
        ["dataproc/pipeline.py"], ["run_pipeline"],
        True, "Exact function name match"
    )

    # --- Import/dependency discovery ---
    add_query(
        "q013", "import csv",
        "Import statement search",
        "import_discovery", "dataproc",
        ["dataproc/reader.py"], [],
        True, "Import statement for csv module"
    )
    add_query(
        "q014", "from .reader",
        "Relative import search",
        "import_discovery", "dataproc",
        ["dataproc/pipeline.py"], [],
        True, "Relative import from .reader module"
    )
    add_query(
        "q015", "import sqlite3",
        "Import statement search",
        "import_discovery", "webapp",
        ["webapp/database.py"], [],
        True, "Import statement for sqlite3 module"
    )

    # --- Cross-file relationships ---
    add_query(
        "q016", "transform_data",
        "Function called across files (pipeline.py calls transformer.py)",
        "cross_file", "dataproc",
        ["dataproc/transformer.py", "dataproc/pipeline.py"], ["transform_data"],
        True, "Function defined in transformer.py, called from pipeline.py"
    )
    add_query(
        "q017", "read_json",
        "Function called across files (pipeline.py calls reader.py)",
        "cross_file", "dataproc",
        ["dataproc/reader.py", "dataproc/pipeline.py"], ["read_json"],
        True, "Function defined in reader.py, called from pipeline.py"
    )
    add_query(
        "q018", "register_routes",
        "Function called across files (app.py calls routes.py)",
        "cross_file", "webapp",
        ["webapp/routes.py", "webapp/app.py"], ["register_routes"],
        True, "Function defined in routes.py, called from app.py"
    )

    # --- Natural language descriptions ---
    add_query(
        "q019", "how to add two numbers",
        "Natural language query for addition function",
        "natural_language", "calc",
        ["calc/calculator.py", "calc/operations.py"], ["add"],
        True, "Natural language description of add function"
    )
    add_query(
        "q020", "how to read a CSV file",
        "Natural language query for CSV reading",
        "natural_language", "dataproc",
        ["dataproc/reader.py"], ["read_csv"],
        True, "Natural language description of read_csv function"
    )
    add_query(
        "q021", "how to write JSON data",
        "Natural language query for JSON writing",
        "natural_language", "dataproc",
        ["dataproc/writer.py"], ["write_json"],
        True, "Natural language description of write_json function"
    )
    add_query(
        "q022", "how to truncate a string",
        "Natural language query for string truncation",
        "natural_language", "utils",
        ["utils/string_utils.py"], ["truncate"],
        True, "Natural language description of truncate function"
    )

    # --- Synonym / vocabulary mismatch ---
    add_query(
        "q023", "compute sum",
        "Synonym for 'add' (vocabulary mismatch)",
        "synonym", "calc",
        ["calc/calculator.py", "calc/operations.py"], ["add"],
        True, "Synonym 'compute sum' refers to 'add' function"
    )
    add_query(
        "q024", "data transformation",
        "Synonym for 'transform_data' (vocabulary mismatch)",
        "synonym", "dataproc",
        ["dataproc/transformer.py"], ["transform_data"],
        True, "Synonym 'data transformation' refers to 'transform_data' function"
    )
    add_query(
        "q025", "file reader",
        "Synonym for 'read_csv' (vocabulary mismatch)",
        "synonym", "dataproc",
        ["dataproc/reader.py"], ["read_csv"],
        True, "Synonym 'file reader' refers to 'read_csv' function"
    )

    # --- Configuration discovery ---
    add_query(
        "q026", "DATABASE_URL",
        "Configuration constant discovery",
        "config_discovery", "webapp",
        ["webapp/app.py"], [],
        True, "Configuration constant in app.py"
    )
    add_query(
        "q027", "SECRET_KEY",
        "Configuration constant discovery",
        "config_discovery", "webapp",
        ["webapp/app.py"], [],
        True, "Configuration constant in app.py"
    )

    # --- Error handling discovery ---
    add_query(
        "q028", "raise ValueError",
        "Error handling pattern discovery",
        "error_handling", "calc",
        ["calc/calculator.py", "calc/operations.py"], [],
        True, "Error handling pattern in divide function"
    )
    add_query(
        "q029", "if user is None",
        "Error handling pattern discovery",
        "error_handling", "webapp",
        ["webapp/routes.py"], [],
        True, "Error handling pattern in user_handler"
    )

    # --- No-result queries ---
    add_query(
        "q030", "quantum computing",
        "Query with no matching results",
        "no_result", "calc",
        [], [],
        False, "No relevant code in fixture"
    )
    add_query(
        "q031", "nonexistent_function_xyz",
        "Query for nonexistent function",
        "no_result", "calc",
        [], [],
        False, "Function does not exist in any fixture"
    )
    add_query(
        "q032", "machine learning model",
        "Query with no matching results",
        "no_result", "webapp",
        [], [],
        False, "No ML-related code in fixture"
    )

    # --- Ambiguous queries ---
    add_query(
        "q033", "read",
        "Ambiguous query matching multiple read functions",
        "ambiguous", "dataproc",
        ["dataproc/reader.py"], ["read_csv", "read_json", "read_text"],
        True, "Ambiguous term matching multiple read functions"
    )
    add_query(
        "q034", "write",
        "Ambiguous query matching multiple write functions",
        "ambiguous", "dataproc",
        ["dataproc/writer.py"], ["write_csv", "write_json", "write_text"],
        True, "Ambiguous term matching multiple write functions"
    )
    add_query(
        "q035", "data",
        "Highly ambiguous query matching many symbols",
        "ambiguous", "dataproc",
        ["dataproc/transformer.py", "dataproc/writer.py", "dataproc/pipeline.py"], [],
        True, "Highly ambiguous term matching many functions"
    )

    return queries


if __name__ == "__main__":
    dataset = build_dataset()
    output = {
        "dataset_version": "0.1.0",
        "description": "Gold-standard benchmark dataset for Sherlock retrieval evaluation",
        "num_queries": len(dataset),
        "queries": dataset,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(dataset)} queries to {OUTPUT_PATH}")
