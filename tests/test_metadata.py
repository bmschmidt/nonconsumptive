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
  # This cleans it up a bit in a way that isn't fair, but will let some tests
  # pass before we try the hard ones.
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
def tmp_feather(tmpdir_factory):
  # A temporary feather file path.
  fn = tmpdir_factory.mktemp("tmp").join("tmp.feather")
  path = Path(str(fn))
  if path.exists():
    path.unlink()
  return path

@pytest.fixture(scope="function")
def dissertation_corpus(corrected_dissertations, tmpdir):
  return Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = tmpdir,
            text_options = {"metadata_field" : "dissertation"})

@pytest.fixture(scope="function")
def non_metadata_corpus(tmpdir_factory):
  dir = Path(str(tmpdir_factory.mktemp("testing")))
  return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                metadata = None,                
                dir = dir, text_options = {"format" : "txt"}, cache_set = {})

class TestDatabasePrep():
  def test_flat_metadata(self, dissertation_corpus):
    dissertation_corpus.metadata.to_flat_catalog()



class TestMetadata():
  def test_cat_alone_makes_metadata(self, dissertation_corpus):
    tb = dissertation_corpus.metadata.tb
    assert len(tb) == 12

  def test_text_alone_makes_metadata(self, non_metadata_corpus):
    tb = non_metadata_corpus.metadata.tb
    assert len(tb) == 3

  def test_corpus_instantiation(self, corrected_dissertations, tmpdir):
    path = Path(str(tmpdir))
    first_pass = Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = path,
            text_options = {"metadata_field" : "dissertation"})
    return None
  
  def test_cache_creation(self, corrected_dissertations, tmpdir):
    path = Path(str(tmpdir))
    first_pass = Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = path,
            text_options = {"metadata_field" : "dissertation"})
    tb = first_pass.metadata.tb
    assert len(tb) == 12
#    first_pass.metadata.load_processed_catalog()
    assert (path / "metadata").exists()
    persisted_data = feather.read_table(path / "metadata/00000.feather")
    assert len(persisted_data) == 12

  def test_cache_use(self, corrected_dissertations, tmpdir):
    """
    Create an incorrect catalog at the correct location and make sure that 
    it is used rather than the whole thing being created anew. Since this 
    test is checking for failure, it would be OK to delete it later if a new
    test for cache use is added.
    """
    path = Path(str(tmpdir))
    schema = pa.schema({"@id": pa.string()},
        metadata = {"nonconsumptive": json.dumps({"schema_version": "0.1.0"})})
    tb = pa.table({"@id": pa.array(["a"])}, schema = schema)
    cat_dir =  path / "metadata"
    cat_dir.mkdir()
    feather.write_feather(tb, cat_dir / "00001.feather")
    should_use_cache = Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = path,
            text_options = {"metadata_field" : "@id"})
    tb = should_use_cache.metadata.tb
    assert len(tb) == 1

  def test_feather_ingest(self, corrected_dissertations, tmpdir):
    path = Path(str(tmpdir)) / "feather"
    path.mkdir()

    import pyarrow.json as json

    tb = json.read_json(corrected_dissertations)
    t2 = pa.table([tb['filename'], tb['dissertation'], tb['year']], names = ['alt_id_title', 'dissertation', 'year'])
    feather.write_feather(t2, path / "input.feather")
    corp = Corpus(texts = None,
            metadata = path / "input.feather",
            dir = tmpdir / "3",
            metadata_options = {"id_field": "alt_id_title"},
            text_options = {"metadata_field" : "dissertation"})

    c = corp.metadata.tb['@id']
    assert len(c) > 7

  def test_upstream_changes_invalidate_cache(self):
    pass

  def test_autogenerate_textids(self, dissertation_corpus, non_metadata_corpus):
    assert len(dissertation_corpus.metadata.ids) == 12
    assert len(non_metadata_corpus.metadata.ids) == 3

  def test_use_metadata_field(self, dissertation_corpus, tmpdir):
    pass


class TestCatalog():
  def test_ndjson_failure(self, tmpdir):
    with pytest.raises(pa.ArrowInvalid):
      cat = Catalog(Path("tests/catalogs/dissertations.ndjson"), Path(str(tmpdir)) / "test.feather")
      p = cat.nc_catalog

  def test_ndjson(self, corrected_dissertations, tmpdir):
    cat = Catalog(Path(corrected_dissertations), Path(str(tmpdir)) / "test.feather")
    nc = cat.nc_catalog      
    assert pa.types.is_list(nc['keywords'].type)
    assert pa.types.is_integer(nc['year'].type)
  
  def test_ia_catalog(self, tmp_feather):
    cat = Catalog("tests/catalogs/ia.ndjson.gz", tmp_feather)
    parsed = cat.nc_catalog
    return
