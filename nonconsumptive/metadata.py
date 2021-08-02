from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
  from nonconsumptive import Corpus

from typing import DefaultDict, Iterator, Union, Optional, List, Dict, Any

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
import nonconsumptive as nc
import regex as re
import logging
from .bookstack import Athenaeum

logger = logging.getLogger("nonconsumptive")

nc_schema_version = 0.1

def cat_from_filenames(input, path):
  tb = pa.table({"@id": pa.array(input.ids(), pa.utf8())})
  feather.write_feather(tb, path)

class Metadata(object):
  def __init__(self, corpus, raw_file: Path, id_field : Optional[str] = None):
    self.corpus = corpus
    self._tb = None
    self._ids = None
    self.id_field = id_field
    if self.path.exists():
      if raw_file is not None:
        if self.path.stat().st_mtime < Path(raw_file).stat().st_mtime:
          logger.info("Deleting out-of-date NC catalog.")
          self.path.unlink()
        else:
          return
      else:
        return
      
    logger.info(f"Creating catalog from {raw_file}.")
    if corpus.input_bookstacks:
      raw_file = corpus.input_bookstacks
    catalog = Catalog(raw_file, self.path, id_field = self.id_field)
    logger.info(f"Saving metadata ({len(catalog.nc_catalog)} rows)")
    if not 'bookstack_size' in corpus.slots:
      corpus.slots['bookstack_size'] = 2**14

    feather.write_feather(catalog.nc_catalog, self.path, chunksize = corpus.slots['bookstack_size'])
  
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
    logger.warning("Loading catalog from disk.")
    self.load_processed_catalog()
    return self._tb

  def __iter__(self):
    pass

  def load_processed_catalog(self, columns = ["@id"]):
    self._tb = feather.read_table(self.path, columns = columns)
  
  @property
  def path(self):
    return Path(self.corpus.root / "nonconsumptive_catalog.feather")

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

    tab = ipc.open_file(self.path)
    logger.debug(f"read file with schema {tab.schema.names}")

    writers = dict()
    written_meta = 0

    for i in range(tab.num_record_batches):
      batch = tab.get_batch(i)
      logging.debug(f"ingesting metadata batch {i}, {written_meta} items written total.")
      dict_tables = set()

      tables : Dict[str, dict] = defaultdict(dict)
      tables['fastcat']['_ncid'] = pa.array(np.arange(written_meta, written_meta + len(batch)))
      tables['catalog']['_ncid'] = tables['fastcat']['_ncid']
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
          tname = name
          parents = col.value_parent_indices()
          tables[tname]['_ncid'] = tables['fastcat']['_ncid'].take(parents)
          flat = col.flatten()
          if pa.types.is_dictionary(col.type):
            dict_tables.add(name)
            tables[tname][tname + "__id"] = flat.indices()
          else:
            tables[tname][tname] = flat
        else:
          logger.error("WHAT IS ", name)
          raise
      for table_name in tables.keys():
        loc_batch = pa.table(tables[table_name])
        if not table_name in writers:
          writers[table_name] = parquet.ParquetWriter(outpath / (table_name + ".parquet"), loc_batch.schema)
        writers[table_name].write_table(loc_batch)
      written_meta += len(batch)

    for name in dict_tables:
      # dictionaries can be written just once, using the final batch
      # from the iteration above.
      col = batch[name]
      table_name = name + "Lookup"
      tab = pa.table(
        [pa.array(np.arange(len(col.dictionary)), col.type.index_type),
        col.dictionary],
        names=[f"{name}__id", name]
      )
      parquet.write_table(tab, outpath  / (table_name + ".parquet"))

def infer_id_field(tb: pa.Table) -> str:
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

def ingest_json(file : Path, write_dir : Path) -> pa.Table:

    """
    JSON ingest includes some error handling for values with inconsistent encoding as 
    arrays or strings.
    
    Recurses every time it finds another array field (inefficient).
    """

    try:
      input : Union[Path, gzip.GzipFile] = file
      if file.suffix == ".gz":
        input = gzip.open(str(file), "rb") 
      # Needs large block size to avoid 'straddling object straddles two block boundaries' errors.
      return pa.json.read_json(input, read_options = pa.json.ReadOptions(block_size = 256_000_000))
    except pa.ArrowInvalid as err:
        match = re.search(r'Column\(/(.*)\) changed from (string|array) to (array|string)', str(err))
        if match:
            # Wrap the column as a list
            bad_col = match.groups()[0]
            logging.warning(f"Wrapping {bad_col} as an array in all json fields")
            newpath = write_dir / "tmp.ndjson.gz"
            wrap_arrays_as_column(file, [str(bad_col)], newpath)
            return ingest_json(newpath, newpath.parents[0])
        else:
          raise

def opener(path : Path):
  if path.suffix == ".gz":
    return gzip.open(path, "rt")
  else:
    return path.open()

def wrap_arrays_as_column(ndjson_file : Path, columns, new_dest : Path):
  import json
  dir = new_dest.parents[0]
  with gzip.open(dir / 'converted.json.gz', 'wt') as replacement:
    for line in opener(ndjson_file):
      line = json.loads(line)
      for c in columns:
        if c in line and not type(line[c]) == list:
          line[c] = [line[c]]
      replacement.write(json.dumps(line) + "\n")
  if Path(new_dest).exists():
    Path(new_dest).unlink()
  (dir / 'converted.json.gz').replace(new_dest)
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
    def __init__(self, 
      file: Optional[Union[str, Path, pa.Table]],
      final_location: Optional[Union[str, Path]] = None,
      id_field: Optional[str] = None,
      exclude_fields = set([])):
      """
      file: the file to load. This can be either a raw csv, json, etc; or a final nonconsumptive parquet/feather file.
      final_location: If 'file' is not a nonconsumptive catalog, the location at which one should be stored.

      id_field: the field containing a unique id for each item. If None, will use 
      {'@id', 'id', or 'filename'} if they are in the set. (In that order, except if
      one appears as the first column, in which case the first-column one will be 
      preferred.)

      schema: passed to nonconsumptive metadata

      table: an instantiated table from somewhere else. Can be used as an input.
      """
      self.final_location = Path(final_location)
      self.tb = None
      self.id_field = id_field
      if file is None and final_location is not None:
        file = final_location

      self.file = Path(file)

      if self.file.is_dir():
        self.raw_tb = file
        self.format = "bookstacks"
        self.id_field = id_field
      elif self.file.suffix == ".gz":
        self.compression = ".gz"
        self.format = self.file.with_suffix("").suffix
      else:
        self.compression = None
        self.format = self.file.suffix

      if self.format == ".feather":
        self.tb = feather.read_table(self.file)
        try:
          # Check if the feather file is 
          self.nc_schema = self.tb.schema.metadata.get(b"nonconsumptive")
          return
        except:
          self.raw_tb = self.tb
          self.tb = None
          
      self.load_existing_nc_metadata()

      if self.tb is None:
        self.load_preliminary_file(self.file)
        if self.id_field is None:
          self.id_field = infer_id_field(self.raw_tb)
        else:
          pass
      else:
        self.id_field = "this shouldn't matter"

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
      elif self.format == ".feather":
        self.load_feather(path)
      elif self.format == "bookstacks":
        self.load_bookstacks(path)
      else:
        raise NotImplementedError(f"Unable to load metadata information: strategy for format {self.format}")
    def load_bookstacks(self, path):
      self.raw_tb = Athenaeum(self.file).metadata()

    def load_feather(self, file):
      self.raw_tb = pa.feather.read_table(file)

    def load_ndjson(self, file : Path):
      """
      Read metadata from an ndjson file.
      """
      self.raw_tb = ingest_json(file, self.final_location.parents[0])

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
      if self.id_field is None:
        self.id_field = infer_id_field(self.raw_tb)
      all_ids = self.raw_tb[self.id_field]
      assert len(all_ids) == len(set(all_ids)), \
        f"Duplicate catalog values in {self.id_field} not allowed " + \
        f"but {len(all_ids) - len(set(all_ids))} present."
      for name in self.raw_tb.schema.names:
        role = None
        if name == self.id_field:
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
      self.meta : Dict[str, Any] = {}
      self._best_form = None
      if isinstance(data.type, pa.ListType):
        self.parent_list_indices = pa.chunked_array(
          # Only take the last offset from the last element.
          [d.offsets for d in data.chunks]
        )
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
      try:
        if pa.types.is_timestamp(self.c.type):
          # Getting weird intraday problems, so hacking my way out.
          #return pc.cast(self.date.c, pa.date64(), cast_options = pc.CastOptions(allow_time_truncate = True))
          return pc.multiply(1000, self.c.cast(pa.int64()).cast(pa.int64())).cast(pa.date64())          
        if pa.types.is_time(self.c.type):
          return self.c.cast(pa.date32())
        if pa.types.is_date(self.c.type):
          return self.c.cast(pa.date32())        
      except:
        raise TypeError("Unable to cast between times")
      datelike_share = pc.mean(pc.match_substring_regex(self.c, "[0-9]{3,4}-[0-1]?[0-9]-[0-3]?[0-9]").cast(pa.int8()))
      if pc.greater(datelike_share, pa.scalar(.95)).as_py():
        series = self.pl.str.strptime(pl.datatypes.Date32, "%Y-%m-%d")
        return series.to_arrow()
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

      # Rebuild into chunked array.
      ix = 0
      arrs = []

      dictionary = idLookup[self.name].combine_chunks()
      for chunks in self.c.chunks:
        indices = pc.index_in(self.c, value_set=dictionary).cast(nt)
        arr = pa.DictionaryArray.from_arrays(
          indices,
          dictionary,
          ordered = True
        )
        arrs.append(arr)
      return pa.chunked_array(arrs)
    
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
      chunks = []
      for i, chunk in enumerate(form.chunks):
        chunks.append(
          pa.ListArray.from_arrays(
            self.parent_list_indices.chunk(i),
            form.chunk(i)
          )
        )
      return pa.chunked_array(chunks)
      
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