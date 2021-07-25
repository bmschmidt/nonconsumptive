from nonconsumptive.commander import parser, main
import pyarrow as pa
import pytest

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
  parse_argumets([])
  return Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = tmpdir,
            text_options = {"text_field" : "dissertation"})

class TestCommandLine():
  def test_parsing(self):
    pass