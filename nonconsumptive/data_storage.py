from pathlib import Path
from .document import Document
from pyarrow import parquet, feather, ipc
from pyarrow import compute as pc
import pyarrow as pa
import json
import yaml
import uuid
import numpy as np
from numpy import packbits
import logging
logger = logging.getLogger("nonconsumptive")

from typing import Callable, Iterator, Union, Optional, List
from .prefs import prefs
from .metadata import Metadata
from .arrow_helpers import batch_id
import nonconsumptive as nc
SRP = None

class Reservoir(object):
  __doc__ = """
  A store for holding, writing, and accessing tables on disk.
  """
  name = None
  def __init__(self,
    corpus,
    cache:bool = False,
    dir:Union[None, str, Path] = None,
    batch_size:int = 1024 * 1024 * 512,
    uuid: Optional[str] = None,
    number:int = 0
    ):
    """

    batch_size: number of bytes to write to each file before creating a new 
      one. Can be larger than memory.

    uuid: One of three things.
    1. A defined uuid to handle multithreading, parallelism, filenames, etc.
    2. The special string "all" means the union of all IDs.
    3. None, which means 

    The file number within the register. New files will be created starting with this.

    """
    if uuid is None:
      self.uuid = str(uuid.uuid1())
    else:
      self.uuid = uuid
    self.number= number
    self.corpus = corpus
    if self.name is None:
      raise NameError("There must be a name defined on each reservoir subclass")
    if dir is None:
      dir = corpus.root / self.name
    dir.mkdir(exist_ok = True)
    self.batch_size = batch_size
    self.path = dir
    self.cache = cache
    self._id_map = None

  def empty(self):
    # Check if any items have been created.
    for _ in self.path.glob(f"**/*.{self.format}"):
      return False
    return True

  def full(self):
    # Check if every item in the metadata has been created.
    pass

  def build_cache(self):
    raise NotImplementedError("Class does not allow caching.")

  def iter_cache(self):
    raise NotImplementedError("Class does not allow caching.")
  
  def id_in_metadata(self):
    try:
      _ = self.arrow_schema.field('@id')
      return True
    except KeyError:
      return False

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

class Livestream(Reservoir):
  format = None

class ArrowReservoir(Reservoir):
  format = "ipc"

  def __init__(self, *args, **kwargs):
    self._writer = None
    super().__init__(*args, **kwargs)

  def open_writer(self):
    # Until arrow supports metadata on record batches for ipc, not just 
    # tables (which I'm pretty sure it doesn't; https://issues.apache.org/jira/browse/ARROW-6940)
    self.number += 1
    self.current_file_uuid = f"{self.uuid}_{self.number}"
    self.bytes = 0
    self.batches = 0
    path = self.path / (self.current_file_uuid + ".ipc")
    fout = pa.ipc.new_file(path, schema = pa.schema(self.schema), 
      options=pa.ipc.IpcWriteOptions(compression = "zstd"))
    return fout

  @property
  def writer(self):
    if self._writer is None:
      self._writer = self.open_writer()
    return self_writer

  @property
  def schema(self):
    return self.arrow_schema

  def build_cache(self, chunk_size = 1024 * 1024 * 128, max_size = 5e08, max_files = 1e06):
    """
    chunk_size: Max size of memory record batch, bytes. Default 128 MB.
    max_size:  Max size of reservoir files, in bytes.
    max_files: Max documents in a single reservoir file.
    """
    mem_size = 0
    sink = self.open_writer()
    cache = []
    for batch in self._from_upstream():
      cache.append(batch)
      mem_size += batch.nbytes
      self.batches += 1
      self.bytes += batch.nbytes
      if mem_size > chunk_size:
        all = pa.Table.from_batches(cache).combine_chunks()
        for megabatch in all.to_batches():
          sink.write_batch(megabatch)
          yield megabatch
          mem_size = 0
      if self.bytes > max_size or self.batches > max_files:
        sink.close()
        self.flush_ids()
        sink = self.open_writer()
        mem_size = 0
        self.batches = 0
    all = pa.Table.from_batches(cache).combine_chunks()
    for megabatch in all.to_batches():
      sink.write_batch(megabatch)
      yield megabatch
    sink.close()
    self.flush_ids()

class ArrowTableReservoir(ArrowReservoir):

  def __init__(self, *args, **kwargs):
    self._writer = None
    super().__init__(*args, **kwargs)

  def build_cache(self):
    """
    max_size:  Max size of reservoir files, in bytes.
    max_files: Max documents in a single reservoir file.
    """
    logger.info(f"Creating cache for {self.name} {self.uuid}")
    for batch in self._from_upstream():
      assert(batch.schema.metadata)
      yield batch
      self.ids.append(batch_id(batch))
      sink.write_batch(batch)
      self.batches += 1
      self.bytes += batch.nbytes
      if self.bytes > max_size or self.batches > max_files:
        sink.close()
        self.flush_ids()
        sink = self.open_writer()
        self.batches = 0
    sink.close()
    self.flush_ids()

  def flush_ids(self):
    path = self.path / (self.current_file_uuid + ".index.feather")
    tab = pa.table([
      pa.array(self.ids),
      pa.array(range(len(self.ids)))], names = ['@id', 'batch_no'])
    feather.write_feather(tab, path)
    self.ids = []

  @property
  def id_location_map(self):
    if self.empty():
      return None
    dest = Path(self.path / "id_location_lookup.parquet")
    if dest.exists():
      dest.unlink()
    files = []
    ids = []
    batch_numbers = []
    for file in self.path.glob("*.json"):
      ids_here = json.load(file.open())
      ids.extend(ids_here)
      batch_numbers.extend(range(len(ids_here)))
      fname = file.with_suffix("").name
      files.extend([fname] * len(ids_here))
    tab = pa.table(
      {
        'file': files,
        '@id': ids,
        'batch': batch_numbers
      })
    tab = tab.take(pc.sort_indices(tab['@id']))
    parquet.write_table(tab, dest)
    return dest

  def get_id(self, id) -> pa.RecordBatch: 
    if self.empty():
      for r in self.build_cache():
        pass
    rows = parquet.read_table(self.id_location_map, filters = [(('@id', '==', id))])
    dicto = rows.to_pydict()
    file = dicto['file'][0] + self.format
    batch = dicto['batch'][0]
    with ipc.open_file(self.path / file) as f:
      return f.get_batch(batch)

  def iter_cache(self) -> Iterator[pa.RecordBatch]:
    for file in self.path.glob("*.ipc"):
      fin = pa.ipc.open_file(file)
      # Must store metadata in json b/c not supported in IPC record batch writer
      # at the moment.
      ids = feather.read_table(file.with_suffix(".index.feather"), columns=["@id"])
      ids = ids['@id']
      for i in range(fin.num_record_batches):
        batch = fin.get_batch(i)
        batch = batch.replace_schema_metadata({"@id": ids[i].as_py()})
        assert batch.schema.metadata
        yield batch

class ArrowIDColReservoir(ArrowReservoir):

  def open_writer(self):
    return super().open_writer()

  def build_cache(self, max_size = 5e08, max_files = 1e06):
    """

    max_size:  Max size of reservoir files, in bytes.
    max_files: Max documents in a single reservoir file.
    """

    print("Building cache")
    sink = self.open_writer()
    for batch in self._from_upstream():
      assert(batch.schema.metadata)
      yield batch
      self.ids.append(batch_id(batch))
      sink.write_batch(batch)
      self.batches += 1
      self.bytes += batch.nbytes
      if self.bytes > max_size or self.batches > max_files:
        sink.close()
        self.flush_ids()
        sink = self.open_writer()
        self.batches = 0
    sink.close()
    self.flush_ids()

  def flush_ids(self):
    path = self.path / (self.current_file_uuid + ".index.feather")
    tab = pa.table([
      pa.array(self.ids),
      pa.array(range(len(self.ids)))], names = ['@id', 'batch_no'])
    feather.write_feather(tab, path)
    self.ids = []

  @property
  def id_location_map(self):
    if self.empty():
      return None
    dest = Path(self.path / "id_location_lookup.parquet")
    if dest.exists():
      dest.unlink()
    files = []
    ids = []
    batch_numbers = []
    for file in self.path.glob("*.json"):
      ids_here = json.load(file.open())
      ids.extend(ids_here)
      batch_numbers.extend(range(len(ids_here)))
      fname = file.with_suffix("").name
      files.extend([fname] * len(ids_here))
    tab = pa.table(
      {
        'file': files,
        '@id': ids,
        'batch': batch_numbers
      })
    tab = tab.take(pc.sort_indices(tab['@id']))
    parquet.write_table(tab, dest)
    return dest

  def get_id(self, id) -> pa.RecordBatch: 
    if self.empty():
      for r in self.build_cache():
        pass
    rows = parquet.read_table(self.id_location_map, filters = [(('@id', '==', id))])
    dicto = rows.to_pydict()
    file = dicto['file'][0] + self.format
    batch = dicto['batch'][0]
    with ipc.open_file(self.path / file) as f:
      return f.get_batch(batch)

  def iter_cache(self) -> Iterator[pa.RecordBatch]:
    for file in self.path.glob("*.ipc"):
      fin = pa.ipc.open_file(file)
      # Must store metadata in json b/c not supported in IPC record batch writer
      # at the moment.
      ids = feather.read_table(file.with_suffix(".index.feather"), columns=["@id"])
      ids = ids['@id']
      for i in range(fin.num_record_batches):
        batch = fin.get_batch(i)
        batch = batch.replace_schema_metadata({"@id": ids[i].as_py()})
        assert batch.schema.metadata
        yield batch


class SRP_set(ArrowReservoir):
  """ 
  Embed a set of tokencounts into a fixed high-dimensional
  space for comparisons. You've got to persist this to disk, because it 
  would be a gratuitous waste of energy not to and I'm not cool with that.
  """
  name = "SRP"
  schema = pa.schema({
    "@id": pa.string(),
    "SRP": pa.list_(pa.float32(), int(1280)),
    "SRP_bits": pa.binary(int(1280) // 8)
  })
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.corpus.cache_set.add("SRP")

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    # Yields 1-row batches. These can be assembled later into longer ones; the 
    # SRP step is expensive enough that it's NBD to wait.
    global SRP
    if SRP is None:
      import SRP
    hasher = SRP.SRP(dim = int(dims), cache = True)
    for batch in self.corpus.tokenization:
      # At a first pass, 
      id = batch_id(batch)
      tokens = batch['token'].to_pylist()    
      counts = batch['count'].to_numpy()
      hash_rep = hasher.stable_transform(words = tokens, counts = counts)
      bit_rep = packbits(hash_rep > 0).tobytes()
      yield pa.record_batch([
          pa.array([id], self.arrow_schema[0].type),
          pa.array([hash_rep], self.arrow_schema[1].type),
          pa.array([bit_rep], self.arrow_schema[2].type)
      ])


  def build_cache(self, max_size = 5e08, max_files = 1e06):
    """

    max_size:  Max size of reservoir files, in bytes.
    max_files: Max documents in a single reservoir file.
    """

    print("Building cache")
    sink = self.open_writer()
    for batch in self._from_upstream():
      assert(batch.schema.metadata)
      yield batch
      if not '@id' in self.arrow_schema:
        self.ids.append(batch_id(batch))
        sink.write_batch(batch)
      self.batches += 1
      self.bytes += batch.nbytes
      if self.bytes > max_size or self.batches > max_files:
        sink.close()
        self.flush_ids()
        sink = self.open_writer()
        self.batches = 0
    sink.close()
    self.flush_ids()

