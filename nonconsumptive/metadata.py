from __future__ import annotations
import json
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import json as pa_json
from pyarrow import csv as pa_csv
from pyarrow import parquet, feather
from pathlib import Path
import numpy as np
from .inputs import FolderInput
import tempfile
from pyarrow import types
from collections import defaultdict

tmpdir = tempfile.TemporaryDirectory

class Metadata(object):
  def __init__(self, corpus, tb: pa.Table, materialize:bool = True):
    self.corpus = corpus
    self.tb = self.parse_raw_arrow(tb)
    self._do_not_materialize = not materialize
  def __iter__(self):
    pass

  def clean(self):
    p = self.dir / "metadata_derived.feather"
    if p.exists():
      p.unlink()

  @classmethod
  def from_cache(cls, corpus) -> cls:
    tb = feather.read_table(corpus.root / "metadata_derived.feather")
    return cls(corpus, tb)

  @classmethod
  def from_filenames(cls, corpus) -> cls:
    ids = []
    input = corpus.text_input_method(corpus)
    for i, (doc, text) in enumerate(input):
      ids.append(doc)
    # Not usually done, but necessary here to 
    # because filenames aren't cached.
    ids.sort()
    tb = pa.table({"id": pa.array(ids, pa.utf8())})
    return cls(corpus, tb, materialize = False)


  @classmethod
  def from_ndjson(cls, corpus, file) -> Metadata:
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

  @property
  def text_ids(self) -> pa.Table:
    path = Path(self.corpus.root / "metadata")
    path.mkdir(exist_ok=True)
    dest = path / "textids.feather"
    if dest.exists():
      return feather.read_table(dest)
    print(self.tb.schema)
    ints = np.arange(len(self.tb[self.id_field]), dtype = np.uint32)
    tb = pa.table([
      pa.array(ints),
      self.tb[self.id_field]
      ], [
        "bookid",
        self.id_field
      ])  
    pa.feather.write_feather(tb, dest)
    return tb

  @property
  def id_to_int_lookup(self):
    ids = self.text_ids
    dicto = dict(zip(ids[self.id_field].to_pylist(), ids['bookid'].to_pylist()))
    return dicto

  def parse_raw_arrow(self, tb):
    """
    Do some typechecking on the raw arrow. Ensure that the id field is 
    cast to a string, maybe do some date parsing, that sort of thing.

    kwargs: table metadata arguments.
    """
    self.id_field = id_field(tb)
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
    tb = tb.replace_schema_metadata()
    return tb


  @classmethod
  def from_file(cls, corpus, file):
    if file is None:
      for name in ["metadata.ndjson", "metadata.csv", "metadata.feather", "metadata.parquet", "jsoncatalog.txt"]:
        try:
          return cls.from_file(corpus, file = corpus.metadata_path)
        except FileNotFoundError:
          continue
      raise FileNotFoundError("No file passed and no default file found.")
    if file.suffix == ".ndjson":
      return cls.from_ndjson(corpus, file)
    if file.name == "jsoncatalog.txt":
      return cls.from_ndjson(corpus, file)
    if file.suffix == ".csv":
      return cls.from_csv(corpus, file)
    else:
      raise FileNotFoundError(f"{file} not loadable.")

  def get(self, id) -> dict:
    
    matching = pc.filter(self.tb, pc.equal(self.tb[self.id_field], pa.scalar(id)))
    try:
      assert(matching.shape[0] == 1)
    except:
      return {}
      raise IndexError(f"Couldn't find solitary match for {id}")
    return {k:v[0] for k, v in matching.to_pydict().items()}

def id_field(tb: pa.Table) -> str:
  """
  tb: a pyarrow table to find the id field from
  """
  default_names = {"@id", "filename", "id"}
  if tb.schema[0].name in default_names:
    return tb.schema[0].name
  for field in tb.schema:
    if field.name in default_names:
       return field.name
  raise NameError("No columns named '@id', 'id' or 'filename' in data; please manually set an id for each document.")

def ingest_json(file:Path):

    """
    JSON ingest includes some error handling for values with inconsistent encoding as 
    arrays or strings.
    """

    try:
        return pa.json.read_json(file)
    except pa.ArrowInvalid as err:
        match = re.search(r'Column\(/(.*)\) changed from (string|arry) to (array|string)', str(err))
        if match:
            # Wrap the column as a list
            bad_col = match.groups()[0]
            wrap_arrays_as_column(file, [str(bad_col)], "tmp.ndjson")
            return ingest_json("tmp.ndjson")
        
def wrap_arrays_as_column(ndjson_file, columns, new_dest):
    import json
    with tmpdir() as d:
        d = Path(d)
        with open(d / 'converted.json', 'w') as replacement:
            for line in open(ndjson_file):
                line = json.loads(line)
                for c in columns:
                    if c in line and not type(line[c]) == list:
                        line[c] = [line[c]]
                replacement.write(json.dumps(line) + "\n")
        if Path(new_dest).exists():
            Path(new_dest).unlink()
        (d / 'converted.json').replace(new_dest)
    return new_dest
class Catalog():
    def __init__(self, tb, relations = None, schema = {}, exclude_fields = set([])):
      """
      Initialized with a table indicating raw metadata.
      """
      self.tb = tb
      self.relations = relations
      self.key = self.primary_key

    def to_flat_catalog(self, metadata) -> None:
      """
      Writes flat parquet files suitable for duckdb ingest or other use
      in a traditional relational db setting, including integer IDs.
      """
      rich = self.feather_ld()
      outpath = metadata.corpus.root / "metadata" / "flat_catalog"
      outpath.mkdir(exist_ok = True)
      tables = defaultdict(dict)
      textids = metadata.text_ids
      tables['fastcat']['bookid'] = \
        textids['bookid'].take(pc.index_in(textids['filename'], value_set=rich['filename']))
      tables['catalog']['bookid'] = \
        tables['fastcat']['bookid']
      for name, col in zip(rich.column_names, rich.columns):
        print(name)
        if pa.types.is_string(col.type):
          tables['catalog'][name] = col
        elif pa.types.is_integer(col.type):
          tables['catalog'][name] = col
          tables['fastcat'][name] = col
        elif pa.types.is_dictionary(col.type):
          tables[name + "Lookup"][f'{name}__id'] = pa.array(np.arange(len(col.chunks[-1].dictionary)), col.type.index_type)
          tables[name + "Lookup"][f'{name}'] = col.chunks[-1].dictionary
          tables['fastcat'][f'{name}__id'] = pa.chunked_array([chunk.indices for chunk in rich[name].chunks])          
          tables['catalog'][name] = col # <- Only works b/c parquet has no dict type.
        elif pa.types.is_list(col.type):
          print("Skipping list ", name)
        else:
          print("WHAT IS ", name)
      for table_name in tables.keys():
        parquet.write_table(pa.table(tables[table_name]), outpath  / (table_name + ".parquet"))


    def columns(self):
      for col, name in zip(self.tb.columns, self.tb.schema.fields):
        yield Column(col, name)

    def primary_key(self):
      for f in self.schema.fields:
        if f.name in self.relations:
          return f.name

    def feather_ld(self):
      fields = []
      for name in self.tb.schema.names:
        col = Column(self.tb[name], name)
        fields.append(col.field())
      schemas, columns = zip(*fields)
      return pa.table(columns, schema = pa.schema(schemas))
    def feather_flat(self):
      pass


class Column():
    """
    A single column of metadata as represented in the original and imported into 
    arrow as an array through pa.read_csv, pa.read_json, pa.read_parquet, etc.    

    Provides support for casting into dictionary encoded forms with metadata 
    for further use in nonconsumptive data.
    """
    def __init__(self, data, name):
        self.c = data
        self.name = name
        self.meta = {}
        
    def to_pyarrow(self):
        pass
    
    def integer_form(self, dtype = pa.uint64()):
        itype = None
        for dtype in [pa.int64(), pa.uint64(), pa.int32(), pa.uint32(), pa.int16(), pa.uint16(), pa.int8(), pa.uint8()]:
            try:
                c = self.c.cast(dtype)
                itype = dtype
            except pa.ArrowInvalid:
                continue
            except pa.ArrowNotImplementedError:
                raise pa.ArrowInvalid("No integer reading possible")
        if itype is None:
            raise pa.ArrowInvalid("No integer reading possible")
        return self.c.cast(itype)
    
    def date_form(self):
        try:
            return self.c.cast(pa.date32())
        except pa.ArrowInvalid:
            return None
    
    def as_dictionary(self, thresh = .75):
        distinct, total = self.cardinality()
        if distinct/total < thresh:
            return self.freq_dict.encode()
        
    def freq_dict_encode(self, nt = pa.int32()):
        counts = pa.RecordBatch.from_struct_array(self.c.value_counts())
        # sort decreasing
        counts = counts.take(pc.sort_indices(pa.Table.from_batches([counts]), sort_keys = [("counts", "descending")]))
        ids = np.arange(len(counts))
        idLookup = pa.table({
            f"{self.name}__id": ids,
            f"{self.name}": counts['values']
            , f"_count": counts['counts']      
        })
        
        indices = pc.index_in(self.c, value_set=counts['values']).cast(nt)
        return pa.DictionaryArray.from_arrays(indices.combine_chunks(), idLookup[self.name].combine_chunks())

        return idLookup
        indices = pc.index_in(self.c, options=pc.SetLookupOptions(value_set=counts['values']))
        return indices, idLookup
        
    @property
    def is_list(self):
        return isinstance(self.c.type, pa.ListType)
    
    def cardinality(self):
      c = self.c
      if self.is_list:
        c = pc.list_flatten(c)
      return len(pc.unique(c)), len(c)

    def top_values(self):
      if isinstance(self.c.type, pa.DictionaryType):
        pass

    def quantiles(self, qs = [0, .005, .05, .5, .95, .995, 1]):
        try:
            quantiles = pc.quantile(self.c, q = qs)
        except pa.ArrowNotImplementedError:
            return None
        return list(zip(qs, quantiles.to_pylist()))

    @property
    def metadata(self):
        # quantiles
        quant = self.quantiles()
        if quant:
            self.meta['quantiles'] = json.dumps(quant)
        return self.meta
    
    def field(self):
        form = self.best_form
        return (pa.field(
            name = self.name,
            type = form.type,
            metadata= self.metadata
        ), form)

    @property
    def best_form(self):
        try:
            return self.integer_form()
        except pa.ArrowInvalid:
            pass
        counts, total = self.cardinality()
        if counts / total < .5:
            try:
                return self.dict_encode()
            except pa.ArrowInvalid:
                pass
        return self.c
    
    def extract_year(self, str):
        replaced = pc.replace_substring_regex(str, pattern = ".*([0-9]{4})?.*", replacement = r"\1")
        return replaced.cast(pa.int16())
    
    def dict_encode(self):
        if self.cardinality()[0] < 2**7:
            nt = pa.int8()
        elif self.cardinality()[0] < 2**15:
            nt = pa.int16()
        else:
            nt = pa.int32()
        if self.is_list:
            return pa.array(self.c.to_pylist(), pa.list_(pa.dictionary(nt, pa.string())))
        else:
            return self.freq_dict_encode(nt)