import pytest

from stem_research.io_utils import load_json, write_json


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "nested" / "sample.json"
    data = {"name": "StemResearch", "items": [1, 2, 3]}

    write_json(path, data)

    assert load_json(path) == data


def test_load_json_fails_loudly_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")
