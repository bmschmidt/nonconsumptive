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
from .bookstack import Bookstacks
from .catalog import Catalog
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
      if len([*self.path.glob("*.feather")]) == 0:
        # OK to just have an empty directory there.
        pass
      elif raw_file is not None:
        if self.path.stat().st_mtime < Path(raw_file).stat().st_mtime:
          logger.info("Deleting out-of-date NC catalog.")
          for file in self.path.glob("*"):
            file.unlink()
          self.path.rmdir()
        else:
          return
      else:
        return
      
    logger.info(f"Creating catalog from {raw_file}.")
    if corpus.input_bookstacks:
      raw_file = corpus.input_bookstacks
    catalog = Catalog(raw_file, self.path, id_field = self.id_field, batch_size = corpus.batch_size)
    logger.info(f"Saving metadata ({len(catalog.nc_catalog)} rows)")
    catalog.serialize_to_feather()
  
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
    tbs = []
    if columns is None:
      args = {}
    else:
      args = {"columns": columns}
    for path in self.path.glob("*.feather"):
      tbs.append(feather.read_table(path, **args))
    if len(tbs) == 0:
      return None
    self._tb = pa.concat_tables(tbs)

  @property
  def path(self) -> Path:
    return Path(self.corpus.root / "metadata")

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

    This corresponds to the first normal form on most fields.

    Just does it in one giant go; could do it by batches for
    performance/memory handling, but we don't.
    """
    logger.info("Writing flat catalog")
    outpath = self.corpus.root / "metadata" / "flat_catalog"
    outpath.mkdir(exist_ok = True, parents = True)

    logger.info("Reading full catalog")

    raise NotImplementedError("Must rewrite to use full catalog set.")
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
