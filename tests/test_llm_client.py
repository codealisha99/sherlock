"""Tests for OllamaClient parsing, validation, and JSON repair."""

import unittest
from pydantic import BaseModel, Field
from app.llm.ollama_client import OllamaClient


class DummyModel(BaseModel):
    name: str
    count: int
    tags: list[str] = Field(default_factory=list)


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def test_clean_and_parse_clean_json(self):
        raw = '{"name": "test", "count": 42, "tags": ["a", "b"]}'
        data = self.client._clean_and_parse_json(raw)
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["count"], 42)

    def test_clean_and_parse_fenced_markdown(self):
        raw = '```json\n{"name": "fenced", "count": 10, "tags": []}\n```'
        data = self.client._clean_and_parse_json(raw)
        self.assertEqual(data["name"], "fenced")
        self.assertEqual(data["count"], 10)

    def test_clean_and_parse_with_surrounding_text(self):
        raw = 'Here is the JSON response you requested:\n\n{"name": "surrounded", "count": 7, "tags": ["ok"]}\n\nHope this helps!'
        data = self.client._clean_and_parse_json(raw)
        self.assertEqual(data["name"], "surrounded")
        self.assertEqual(data["count"], 7)

    def test_clean_and_parse_trailing_commas(self):
        raw = '{"name": "trailing", "count": 3, "tags": ["item",],}'
        data = self.client._clean_and_parse_json(raw)
        self.assertEqual(data["name"], "trailing")
        self.assertEqual(data["count"], 3)

    def test_pydantic_validation(self):
        raw = '{"name": "validated", "count": 99, "tags": ["x"]}'
        data = self.client._clean_and_parse_json(raw)
        model = DummyModel.model_validate(data)
        self.assertEqual(model.name, "validated")
        self.assertEqual(model.count, 99)


if __name__ == "__main__":
    unittest.main()
