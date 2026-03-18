import json
from pathlib import Path

from jsonschema import validate

from pgmpy.example_models.bnlearn.ecoli70 import Ecoli70


def test_lgbn_schema():
    """Validate data against LGBN schema with semantic checks."""

    schema_path = Path(__file__).parent.parent.parent.parent / "devtools" / "schema" / "lgbn_schema.json"

    ecoli_model = json.loads(Ecoli70()._get_raw_data())
    validate(instance=ecoli_model, schema=json.load(open(schema_path)))
