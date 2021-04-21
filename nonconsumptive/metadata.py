import json
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import json as pa_json
from pyarrow import csv as pa_csv
from pyarrow import parquet

class Metadata(object):
  def __init__(self, corpus, tb: pa.Table):
    self.corpus = corpus
    self.tb = self.parse_raw_arrow(tb)

  def __iter__(self):
    pass

  def clean(self):
    p = self.corpus.root / "metadata_derived.parquet"
    if p.exists():
      p.unlink()

  @classmethod
  def from_cache(cls, corpus):
      tb = parquet.read_table(corpus.root / "metadata_derived.parquet")
      return cls(corpus, tb)
      
  @classmethod
  def from_ndjson(cls, corpus, file):
    """
    Read metadata from an ndjson file.
    """
    tb = pa_json.read_json(file)
    return cls(corpus, tb)
    
  @classmethod
  def from_csv(cls, corpus, file):
    """
    Read metadata from an ndjson file.
    """
    tb = pa_csv.read_csv(file)
    return cls(corpus, tb)

  def parse_raw_arrow(self, tb, **kwargs):
    """
    Do some typechecking on the raw arrow. Ensure that the id field is 
    cast to a string, maybe do some date parsing, that sort of thing.

    kwargs: table metadata arguments.
    """
    id_field = guess_id_field(tb, **kwargs)
    self.id_field = id_field
    columns = {}
    for field in tb.schema:
      if field.name == id_field and field.type != pa.utf8():
        data = tb.column(field.name).cast(pa.utf8())
      else:
        data = tb.column(field.name)
      columns[field.name] = data
    tb = pa.table(
      columns
    )
    tb = tb.replace_schema_metadata(kwargs)
    return tb


  @classmethod
  def from_file(cls, corpus, file):
    if file is None:
      for name in ["metadata.ndjson", "metadata.csv", "metadata.parquet", "jsoncatalog.txt"]:
        try:
          return cls.from_file(corpus, file=name)
        except FileNotFoundError:
          continue
      raise FileNotFoundError("No file passed and no default file found.")
    file = corpus.root / file
    if file.suffix == ".ndjson":
      return cls.from_ndjson(corpus, file)
    if file.suffix == ".csv":
      return cls.from_csv(corpus, file)
    else:
      raise TypeError(f"{file} not loadable.")

  def get(self, id) -> dict:
    
    matching = pc.filter(self.tb, pc.equal(self.tb[self.id_field], id))
    try:
      assert(matching.shape[0] == 1)
    except:
      return {}
      raise IndexError(f"Couldn't find solitary match for {id}")
    return {k:v[0] for k, v in matching.to_pydict().items()}

def guess_id_field(tb: pa.Table, **kwargs) -> str:
  """
  tb: 
  """
  if "id_field" in kwargs:
    return kwargs['id_field']
  default_names = {"filename", "id"}
  if tb.schema[0].name in default_names:
    return tb.schema[0].name
  for field in tb.schema:
    if field.name in default_names:
       return field.name
  raise NameError("No columns named 'id' or 'filename' in data; please manually set an id for each document.")
