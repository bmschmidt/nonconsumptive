from __future__ import annotations
import json
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import json as pa_json
from pyarrow import csv as pa_csv
from pyarrow import parquet, feather, ipc
from pathlib import Path
import numpy as np
from .inputs import FolderInput
import tempfile
import gzip
from pyarrow import types
from collections import defaultdict
import polars as pl
from typing import DefaultDict, Iterator, Union, Optional, List
import nonconsumptive as nc

tmpdir = tempfile.TemporaryDirectory

import logging
logger = logging.getLogger("nonconsumptive")

nc_schema_version = 0.1

def cat_from_filenames(corpus) -> Metadata:
  ids = []
  for (doc, text) in corpus.texts:
    ids.append(doc)
  # Not usually done, but necessary here to 
  # ensure reproducibility across runs
  # because filenames aren't cached.
  ids.sort()
  tb = pa.table({"@id": pa.array(ids, pa.utf8())})
  return tb

class Metadata(object):
  def __init__(self, corpus: nc.Corpus, raw_file: Optional[Union[str, Path]]):
    self.corpus = corpus
    self._tb = None
    if self.nc_metadata_path.exists():
      if raw_file is not None:
        if self.nc_metadata_path.stat().st_mtime < Path(raw_file).stat().st_mtime:
          logger.info("Deleting out-of-date NC catalog.")
          self.nc_metadata_path.exists.unlink()
        else:
#          logger.info("Loading @id field from on-disk catalog.")
#          self.load_processed_catalog(["@id"])
          return
      else:
#        logger.info("Loading @id field from on-disk catalog.")
#        self.load_processed_catalog(["@id"])
        return
    if raw_file is None:
      # When no metadata is passed, create minimal metadata from the 
      # filenames on disk.
      logger.info("Creating catalog from filenames")
      basic_ids = cat_from_filenames(corpus)
      catalog = Catalog(None, self.nc_metadata_path, table = basic_ids)
    else:
      logger.info(f"Creating catalog from {raw_file}.")
      catalog = Catalog(raw_file, self.nc_metadata_path)
    self._ids = None
    logger.info(f"Saving metadata ({len(catalog.nc_catalog)} rows)")
    feather.write_feather(catalog.nc_catalog, self.nc_metadata_path)
#    self.load_processed_catalog()
  
  @property
  def ids(self) -> pa.Array:
    """
    An array of the string ids for each column in the dataset.
    """
    if self._ids is not None:
      return self._ids

    self._ids = self.tb['@id']
    return self._ids

  @property
  def tb(self):
    if self._tb is not None:
      return self._tb
    logger.warning("Expensively creating table")
    self.load_processed_catalog()
    return self._tb
  def __iter__(self):
    pass

  def load_processed_catalog(self, columns = ["@id"]):
    self._tb = feather.read_table(self.nc_metadata_path, columns = columns)
  
  @property
  def nc_metadata_path(self):
    return Path(self.corpus.root / "nonconsumptive_catalog.feather")

  """
  @property
  def text_ids(self) -> pa.Table:
    ""
    Create -- or save -- a folder of textid integer lookups.
    ""
    path = Path(self.corpus.root / "metadata")
    path.mkdir(exist_ok=True)
    dest = path / "textids.feather"
    if dest.exists():
      logger.debug("Reading textids from cache")
      return feather.read_table(dest)
    logger.debug("Creating textids")

    ints = np.arange(len(self.tb["@id"]), dtype = np.uint32)
    tb = pa.table([
      pa.array(ints),
      self.tb["@id"]
      ], [
        "bookid",
        "@id"
      ])  
    pa.feather.write_feather(tb, dest)
    return tb
  @property
  def id_to_int_lookup(self):
    ids = self.text_ids
    dicto = dict(zip(ids["@id"].to_pylist(), ids['bookid'].to_pylist()))
    return dicto
  """

  def get(self, id) -> dict:
    matching = pc.filter(self.tb, pc.equal(self.tb["@id"], pa.scalar(id)))
    try:
      assert(matching.shape[0] == 1)
    except:
      return {}
      raise IndexError(f"Couldn't find solitary match for {id}")
    return {k:v[0] for k, v in matching.to_pydict().items()}

  def to_flat_catalog(self) -> None:

    """
    Writes flat parquet files suitable for duckdb ingest or other use
    in a traditional relational db setting, including integer IDs.

    Just does it in one giant go; could do it by batches for performance/memory handling, but we don't.
    """
    logger.info("Writing flat catalog")
    outpath = self.corpus.root / "metadata" / "flat_catalog"
    outpath.mkdir(exist_ok = True, parents = True)

    logger.info("Reading full catalog")

    tab = ipc.open_file(self.nc_metadata_path)
    logger.debug(f"read file with schema {tab.schema.names}")

    writers = dict()
    written_meta = 0

    for i in range(tab.num_record_batches):
      batch = tab.get_batch(i)
      logging.debug(f"ingesting metadata batch {i}, {written_meta} items written total.")
      dict_tables = set()

      tables = defaultdict(dict)
      tables['fastcat']['bookid'] = pa.array(np.arange(written_meta, written_meta + len(batch)))
      tables['catalog']['bookid'] = tables['fastcat']['bookid']
      for name, col in zip(batch.schema.names, batch.columns):
        if pa.types.is_string(col.type):
          tables['catalog'][name] = col
        elif pa.types.is_integer(col.type) or pa.types.is_date(col.type):
          tables['catalog'][name] = col
          tables['fastcat'][name] = col
        elif pa.types.is_dictionary(col.type):
          dict_tables.add(name)
          tables['fastcat'][f'{name}__id'] = col.indices   
          tables['catalog'][name] = col.dictionary.take(col.indices)
        elif pa.types.is_list(col.type):
          logger.warning("Skipping list ", name)
        else:
          logger.error("WHAT IS ", name)
          raise
      for table_name in tables.keys():
        loc_batch = pa.table(tables[table_name])
        if not table_name in writers:
          writers[table_name] = parquet.ParquetWriter(outpath  / (table_name + ".parquet"), loc_batch.schema)
        writers[table_name].write_table(loc_batch)
      written_meta += len(batch)

    for name in dict_tables:
      # Use the last batch for dictionaries
      col = batch[name]
      table_name = name + "Lookup"
      tab = pa.table(
        [pa.array(np.arange(len(col.dictionary)), col.type.index_type),
        col.dictionary],
        names=[f"{name}__id", name]
      )
      parquet.write_table(tab, outpath  / (table_name + ".parquet"))

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
    """

    A catalog represents the elements of the metadata for a corpus that can be expressed 
    without reference to the full text of that corpus. The distinction here is that
    catalog parsing can happen independently of the rest of a corpus, while many calls in
    the metadata class rely on the corpus.

    In theory, the catalog elements here might even be useful in a project like Wax
    where records represented in a catalog are not even textual.

    """
    def __init__(self, file: Union[str, Path], final_location: Union[str, Path] = None, identifier = None, exclude_fields = set([]), table = None):
      """
      file: the file to load. This can be either a raw csv, json, etc; or a final nonconsumptive parquet/feather file.
      final_location: If 'file' is not a nonconsumptive catalog, the location at which one should be stored.

      identifier: the field containing a unique id for each item. If None, will use 
      {'@id', 'id', or 'filename'} if they are in the set. (In that order, except if
      one appears as the first column, in which case the first-column one will be 
      preferred.)

      schema: passed to nonconsumptive metadata

      table: an instantiated table from somewhere else. Can be used as an input.
      """

      if file is None and final_location is not None:
        file = final_location
      
      self.file = Path(file)
      if self.file.suffix == ".gz":
        self.compression = ".gz"
        self.format = self.file.with_suffix("").suffix
      else:
        self.compression = None
        self.format = self.file.suffix
      self.final_location = Path(final_location)
      self.tb = None
      self.identifier = None
      if self.format == ".feather":
        if not table:
          self.tb = feather.read_table(self.file)
        else:
          self.tb = table
        try:
          self.nc_schema = self.tb.schema.metadata.get(b"nonconsumptive")
          return
        except:
          self.raw_tb = self.tb
          self.tb = None
      self.load_existing_nc_metadata()

      if self.tb is None:
        self.load_preliminary_file(self.file)
        if self.identifier is None:
          self.identifier = id_field(self.raw_tb)
      else:
        self.identifier = "@id"

    def load_existing_nc_metadata(self):
      if not self.final_location.exists():
        return
      if self.final_location.stat().st_mtime < self.file.stat().st_mtime:
        # Remove the cache
        self.final_location.unlink()
        return None
      self.tb = feather.read_table(self.final_location)

    def load_preliminary_file(self, path):
      if self.format == ".ndjson" or path.name == "jsoncatalog.txt" or path.name == "jsoncatalog.txt.gz":
        self.load_ndjson(path)
      elif self.format == ".csv":
        self.load_csv(path)

    def load_ndjson(self, file):
      """
      Read metadata from an ndjson file.
      """
      if self.compression == ".gz":
        file = gzip.open(file)
      self.raw_tb = pa_json.read_json(file)

    def load_csv(self, file):
      if self.compression == ".gz":
        file = gzip.open(file)
      self.raw_tb = pa_csv.read_csv(file)

    @property
    def metadata(self):
      return {
        'version': nc.__version__
      }
    
    @property
    def nc_catalog(self):
      if self.tb:
        return self.tb
      fields = []
      for name in self.raw_tb.schema.names:
        role = None
        if name == self.identifier:
          role = "identifier"
        col = Column(self.raw_tb[name], name, role)
        fields.append(col.field())
      schemas, columns = zip(*fields)
      self.tb = pa.table(columns, schema = pa.schema(schemas, metadata = {'nonconsumptive': json.dumps(self.metadata)}))
      return self.tb

class Column():
    """
    A single column of metadata as represented in the original and imported into 
    arrow as an array through pa.read_csv, pa.read_json, pa.read_parquet, etc.    

    Provides support for casting into dictionary encoded forms with metadata 
    for further use in nonconsumptive data.
    """
    def __init__(self, data: pa.ChunkedArray, name, role = None):
      # Sometimes we gotta cast to polars.
      self._pl = None
      self.name = name
      self.role = role
      self.meta = {}
      self._best_form = None
      if isinstance(data.type, pa.ListType):
        self.parent_list_indices = pa.chunked_array([d.offsets for d in data.chunks])
        self.c = pa.chunked_array([d.flatten() for d in data.chunks])
      else:
        self.c = data
        self.parent_list_indices = None
        
    def to_pyarrow(self):
        pass
    
    @property
    def pl(self):
      if self._pl is not None:
        return self._pl
      else:
#        self._pl = pl.from_arrow(self.c.combine_chunks())
        # wrap in a table first b/c polars doesn't like types arrays.
        self._pl = pl.from_arrow(pa.table({self.name: self.c}))[self.name]
        return self._pl

    def integer_form(self, dtype = pa.uint64()):
        itype = None
        # Trying uint types as well to catch things that max out at--say--between 128 and 255.
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
    
    @property
    def date_form(self):
      if pa.types.is_timestamp(self.c.type):
        return self.c.cast(pa.date32())
      datelike_share = pc.mean(pc.match_substring_regex(self.c, "[0-9]{3,4}-[0-1]?[0-9]-[0-3]?[0-9]").cast(pa.int8()))
      if pc.greater(datelike_share, pa.scalar(.95)).as_py():
        return self.pl.str_parse_date(pl.datatypes.Date32, "%Y-%m-%d").to_arrow()
      else:
        raise ValueError(f"only {datelike_share} of values look like a date string.")

    def as_dictionary(self, thresh = .75):
      distinct, total = self.cardinality()
      if distinct/total < thresh:
          return self.freq_dict.encode()
        
    def freq_dict_encode(self, nt = pa.int32()):
      """
      Convert into a dictionary where keys are ordered by count.

      """
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
      return pa.DictionaryArray.from_arrays(indices.combine_chunks(), idLookup[self.name].combine_chunks(), ordered = True)

    
    def cardinality(self):
      c = self.c
      return len(pc.unique(c)), len(c)

    def quantiles(self, qs = [0, .005, .05, .25, .5, .75, .95, .995, 1]):
        try:
            quantiles = pc.quantile(self.best_form, q = qs)
        except pa.ArrowNotImplementedError:
          try:
            min = pc.min_max(self.best_form).to_py()
            qs = [0, 1]
            quantiles = [min, max]
          except pa.ArrowNotImplementedError:
            return None
        return list(zip(qs, quantiles.to_pylist()))

    @property
    def metadata(self):
      # quantiles
      quant = self.quantiles()
      if quant:
        self.meta['quantiles'] = json.dumps(quant)
      if pa.types.is_dictionary(self.best_form.type):
        self.meta['top_values'] = json.dumps(self.best_form.dictionary[:10].to_pylist())
      return self.meta

    def relist(self, form):
      # Convert back to a list form from an unnested one after type coercion.
      if self.parent_list_indices is None:
        return form
      return pa.ListArray.from_arrays(self.parent_list_indices.combine_chunks(), form.combine_chunks())
      
    def field(self):
      form = self.best_form
      form = self.relist(form)
      if self.role == "identifier":
        self.name = "@id"
      return (pa.field(
          name = self.name,
          type = form.type,
          metadata= self.metadata
      ), form)

    @property
    def best_form(self):
      if self._best_form:
        return self._best_form
      if self.role == "identifier":
        self._best_form = self.c.cast(pa.string())
        return self._best_form
      try:
          self._best_form = self.integer_form()
          return self._best_form
      except pa.ArrowInvalid:
          pass
      try:
          self._best_form = self.c.cast(pa.float32())
          return self._best_form
      except pa.ArrowInvalid:
          pass
      except pa.ArrowNotImplementedError:
        pass
      try:
        self._best_form = self.date_form
        return self._best_form
      except pa.ArrowInvalid:
        pass
      except ValueError:
        pass
      try:
        counts, total = self.cardinality()
        if counts / total < .5:
            try:
                self._best_form = self.dict_encode()
                return self._best_form
            except pa.ArrowInvalid:
                pass
      except pa.ArrowCapacityError:
        # Strings can overflow here. Probably a safer way to check this.
        pass
      return self.c
  
    def extract_year(self, arr):
      """
      Aggressively search for years in a field using regular expressions, and turn
      into integers. For people with
      datasets that include a lot of `c. 1875`, and so on.

      Will not work on years before 1000.
      """
      replaced = pc.replace_substring_regex(arr, pattern = ".*([0-9]{4})?.*", replacement = r"\1")
      return replaced.cast(pa.int16())
    
    def dict_encode(self):
        if self.cardinality()[0] < 2**7:
            nt = pa.int8()
        elif self.cardinality()[0] < 2**15:
            nt = pa.int16()
        else:
            nt = pa.int32()
        return self.freq_dict_encode(nt)