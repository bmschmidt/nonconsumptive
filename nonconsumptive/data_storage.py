from pathlib import Path
from .document import Document
from pyarrow import parquet
from pyarrow import feather
import pyarrow as pa
import json
import yaml
import uuid
import numpy as np
from .prefs import prefs
import logging
from .metadata import Metadata
import gzip

from typing import Callable, Iterator, Union, Optional, List

class Reservoir(object):
  __doc__ = """

  A store for holding, writing, and accessing tables in parquet files.

  """

  name = None

  def __init__(self,
    corpus,
    cache = False,
    dir:Union[None, str, Path] = None,
    batch_size = 1024 * 1024 * 512
    ):
    """

    batch_size: number of bytes to write to each file before creating a new 
      one. Can be larger than memory.
    """
    self.corpus = corpus
    if dir is None:
      dir = corpus.root / self.name
    dir.mkdir(exist_ok = True)
    self.batch_size = batch_size
    self.path = dir
    self.cache = cache
    self._id_map = None

  def clean(self):
    for p in self.path.glob(f"*.{self.format}"):
      p.unlink()

  def empty(self):
    for file in self.path.glob(f"**/*.{self.format}"):
      return False
    return True

  def build_cache(self):
      raise NotImplementedError("Class does not allow caching.")

  def iter_cache(self):
      raise NotImplementedError("Class does not allow caching.")
  
  def __iter__(self):
    if self.empty():
      if self.name in self.corpus.cache_set:
        print(f"Building {self.name}")
        for b in self.build_cache():   
          assert b.schema.metadata
          yield b
      else:
        for b in self._from_upstream():
          assert b.schema.metadata

          yield b
    else:
      for batch in self.iter_cache():
        assert batch.schema.metadata
        yield batch

  @property
  def arrow_schema(self):
      raise NotImplementedError(f"Must define an arrow_schema for type {self.name}")

  @property
  def id_map(self):
    if self._id_map is not None:
      return self._id_map
    self._id_map = {}
    for p in self.path.glob("*.parquet"):
      ids = parquet.ParquetFile(p).schema_arrow.metadata[b'ids'].decode("utf-8").split("\t")
      for id in ids:
        self._id_map[id] = p
    return self._id_map


class OldParquetMethods():
  def __init__(self):
      pass
  def create(self):
    max_size = self.batch_size
    self.queue = []
    queue_size = 0

    for document in self.corpus.documents():
      if document.id in self.id_map:
        continue
      try:
        tokens = document.tokenize()
      except:
        print(document.id)
        continue
      # Could speed up.Tokenization
      ids = pa.array([document.id] * len(tokens), pa.utf8())
      batch = pa.RecordBatch.from_arrays(
        [
          ids,
          tokens
        ],
        schema = pa.schema(self.arrow_schema,
        metadata = {b"id": document.id.encode("utf-8")})
      )
      self.queue.append((document.id, batch))
      queue_size += batch.nbytes
      if queue_size > max_size:
        self.flush() # Intermedia write
        queue_size = 0
    self.flush() # final flush.
    self._id_map = None


  def flush_batch(self):
    if len(self.queue) == 0:
      return
    logging.warn(f"Flushing {len(self.queue)}")
    # Get it alphabetically by id
    self.queue.sort()
    ids = [q[0] for q in self.queue]
    schema = self.queue[0][1].schema
    revised = schema.with_metadata({"ids": "\t".join(ids)})
    tab = pa.Table.from_batches([q[1] for q in self.queue], schema = revised)
    fn = self.path / (str(uuid.uuid1()) + ".parquet")
    parquet.write_table(tab, fn)
    self.queue = []

  @property
  def id_map(self):
    if self._id_map is not None:
      return self._id_map
    self._id_map = {}
    for p in self.path.glob("*.parquet"):
      ids = parquet.ParquetFile(p).schema_arrow.metadata[b'ids'].decode("utf-8").split("\t")
      for id in ids:
        self._id_map[id] = p
    return self._id_map
  
  def fetch(self,
    value: str,
  #  filters: List[List[List[str, str, Union[str, int, float, set, list]]]],
    field: str):
    pass


class Livestream(Reservoir):
  format = None


class ArrowReservoir(Reservoir):
  format = "ipc"

  def open_writer(self):
    # Until arrow supports metadata on record batches for ipc, not just 
    # tables (which I'm pretty sure it doesn't; https://issues.apache.org/jira/browse/ARROW-6940)

    self.ids = []
    self.current_file_uuid = str(uuid.uuid1())
    self.bytes = 0
    self.batches = 0

    path = self.path / (self.current_file_uuid + ".ipc")
    fout = pa.ipc.new_file(path, schema = pa.schema(self.arrow_schema), 
      options=pa.ipc.IpcWriteOptions(compression = "zstd"))
    return fout

  def flush(self):
      pass

  def build_cache(self, max_size = 5e08, max_files = 1e06, iterator = True):
    """
    max_size: 
    max size of reservoir files

    max_files:
    max files in a single reservoir file.
    """
    sink = self.open_writer()    
    for batch in self._from_upstream():
        if iterator:
            yield batch
        assert(batch.schema.metadata)
        self.ids.append(batch.schema.metadata.get(b'id').decode('utf-8'))
        sink.write_batch(batch)
        self.batches += 1
        self.bytes += batch.nbytes
        if self.bytes > max_size or self.batches > max_files:
          sink.close()
          self.flush_ids()          
          sink = self.open_writer() 
    self.flush_ids()
    sink.close()               

  def flush_ids(self):
      path = self.path / (self.current_file_uuid + ".json")
      json.dump(self.ids, path.open("w"))
      self.ids = []

  def iter_cache(self) -> Iterator[pa.RecordBatch]:
    for file in self.path.glob("*.ipc"):
      fin = pa.ipc.open_file(file)
      ids = json.load(file.with_suffix(".json").open())
      for i in range(fin.num_record_batches):
        batch = fin.get_batch(i)
        batch = batch.replace_schema_metadata({"id": ids[i]})
        assert batch.schema.metadata
        yield batch

class ParquetReservoir(Reservoir):
  format = "parquet"
