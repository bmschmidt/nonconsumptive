from nonconsumptive import Metadata, Corpus
from nonconsumptive.metadata import *
import pytest
from pathlib import Path
import pandas as pd
import pyarrow as pa
from pyarrow import feather, json as pa_json
import json

@pytest.fixture(scope="session")
def corrected_dissertations(tmpdir_factory):
  fn = tmpdir_factory.mktemp("ndjson").join("catalog.ndjson")
  with open(fn, "w") as fout:
    for line in open("tests/catalogs/dissertations.ndjson"):
      d = json.loads(line)
      for k in [*d.keys()]:
        if d[k] == "NA":
          del d[k]
      fout.write(json.dumps(d) + "\n")
  return Path(str(fn))

@pytest.fixture(scope="function")
def dissertation_corpus(corrected_dissertations, tmpdir):
  return Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = tmpdir,
            text_options = {"text_field" : "dissertation"})


@pytest.fixture(scope="function")
def non_metadata_corpus(tmpdir_factory):
  dir = Path(str(tmpdir_factory.mktemp("testing")))
  return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                metadata = None,
                dir = dir, text_options = {"format" : "txt"}, cache_set = {})  

class TestMetadata():
  def test_cat_alone_makes_metadata(self, dissertation_corpus):
    tb = dissertation_corpus.metadata.tb
    assert len(tb) == 12
  def test_text_alone_makes_metadata(self, non_metadata_corpus):
    tb = non_metadata_corpus.metadata.tb
    assert len(tb) == 3    
  def test_feather_files_written(self, tmpdir):
    pass
  def test_subsequent_loads_use_cache(self):
    pass
  def test_upstream_changes_invalidate_cache(self):
    pass
  def test_autogenerate_textids(self, dissertation_corpus, non_metadata_corpus):
    assert len(dissertation_corpus.metadata.text_ids) == 12
    assert len(non_metadata_corpus.metadata.text_ids) == 3

  def test_use_metadata_field(self, dissertation_corpus, tmpdir):
    pass

class TestCatalog():
  def test_ndjson_failure(self, tmpdir):
    with pytest.raises(pa.ArrowInvalid):
      cat = Catalog(Path("tests/catalogs/dissertations.ndjson"), Path(str(tmpdir)) / "test.feather")
      p = cat.nc_catalog

  def test_ndjson(self, corrected_dissertations):
    cat = Catalog(Path(corrected_dissertations), Path(str(tmpdir)) / "test.feather")
    nc = cat.nc_catalog      
    assert pa.types.is_list(nc['keywords'].type)
    assert pa.types.is_integer(nc['year'].type)
    assert pa.types.is_integer(nc['year'].type)    
  