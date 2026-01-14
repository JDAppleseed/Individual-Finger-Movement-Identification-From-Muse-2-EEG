import pytest

jsonschema = pytest.importorskip("jsonschema")


def test_schema_validation():
    from demo_backend import schemas

    bundle = schemas.schema_bundle()
    jsonschema.validate(bundle["sample_tick"], bundle["tick"])
    jsonschema.validate(bundle["sample_status"], bundle["status"])
