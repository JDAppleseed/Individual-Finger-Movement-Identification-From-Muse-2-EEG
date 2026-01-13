import pytest

jsonschema = pytest.importorskip("jsonschema")

from demo_backend import schemas


def test_schema_validation():
    bundle = schemas.schema_bundle()
    jsonschema.validate(bundle["sample_tick"], bundle["tick"])
    jsonschema.validate(bundle["sample_status"], bundle["status"])
