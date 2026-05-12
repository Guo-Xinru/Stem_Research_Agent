import pytest

from stem_research.io_utils import load_json, load_jsonl, write_json, write_jsonl


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "nested" / "sample.json"
    data = {"name": "StemResearch", "items": [1, 2, 3]}

    write_json(path, data)

    assert load_json(path) == data


def test_load_json_fails_loudly_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    records = [{"id": "a"}, {"id": "b"}]

    write_jsonl(path, records)

    assert load_jsonl(path) == records
