from nonconsumptive import Metadata, Corpus, Catalog
from nonconsumptive.metadata import *
import pytest
from pathlib import Path
import pandas as pd
import pyarrow as pa
from pyarrow import feather, json as pa_json
import json

@pytest.fixture(scope="session")
def corrected_dissertations(tmpdir_factory):
  # This cleans it up a bit in a way that isn't fair, but will let some tests
  # pass before we try the hard ones.
  fn = tmpdir_factory.mktemp("ndjson").join("catalog.ndjson")


class TestCatalog:
  def test_mixed_array(self, tmpdir_factory):
    dir = tmpdir_factory.mktemp("dir")
    a = Catalog("tests/catalogs/split_ndjson/a.ndjson", final_location = Path(dir), exclude_fields = ["labels"])