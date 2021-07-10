from __future__ import annotations

from pathlib import Path
from .document import Document
from pyarrow import parquet, feather, ipc
from pyarrow import compute as pc
import pyarrow as pa
import json
import yaml
import numpy as np
from numpy import packbits
import logging
import nonconsumptive as nc

logger = logging.getLogger("nonconsumptive")
import types
from typing import Callable, Iterator, Union, Optional, List, Tuple, Any
from .utils import ConsumptionError
import SRP

class BatchIDIndex(object):
  def __init__(self, corpus: nc.Corpus, batch_size: int = 1024 * 16):
    self.corpus: nc.Corpus = corpus
    self.dir : Path = corpus.root / "build/bookstacks"
    self.dir.mkdir(exist_ok = True, parents = True)
    self.fn = self.dir / f"{self.corpus.uuid}.feather"
    self.batch_size = batch_size
    self.cache : list = []
    self.ncid_lookup = None
    self._fout = None
    self.schema: pa.Schema = pa.schema({
          "@id": pa.string(),
          "_ncid": pa.uint32()})

  def exists(self):
    return self.fn.exists()
  
  @property
  def fout(self):
    if self._fout is not None:
      return self._fout    
    self._fout = ipc.new_file(self.fn, self.schema)
    return self._fout      
    
  def iter_ids(self, which: str = '@id') -> Iterator[str]:
    if not self.fn.exists():
      logger.warning("ID lookup requested, but none found: "
                      "materializing one inefficiently with a full pass through "
                      "the data")
      self.close()
      logger.info("ID Lookup has been created.")

    ids = feather.read_table(self.fn, columns = [which])[which]
    for id in ids:
      yield id.as_py()
  
  def push(self, id: str):
    self.cache.append(id)
    if len(self.cache) > self.batch_size:
      self.flush()

  def flush(self):
    if len(self.cache):
      ids = pa.array(self.cache, pa.string())
      ncids = pc.index_in(ids, value_set = self.corpus.metadata.ids).cast(pa.uint32())
      self.fout.write(pa.record_batch([ids, ncids], self.schema))
      self.cache = []

  def __enter__(self):
    return self

  def close(self):
    self.flush()
    if self._fout is not None:
      self.fout.close()
      self._fout = None

  def __exit__(self, type, value, tb):
    self.close()

class Node(object):
  __doc__ = """
  An abstract point in a text transformation pipeline.
  """
  def __init__(self, network):
    if network is None:
      return
    if hasattr(network, "is_bookstack"):
      self.bookstack = network
      self.corpus = network.corpus
    else:
      self.corpus = network
      self.bookstack = None
    self.uuid = network.uuid    

class Reservoir(Node):
  __doc__ = """
  A store for holding, writing, and accessing tables on disk.
  """
  name : Optional[str] = None
  def __init__(self,
    bookstack,
    dir:Union[None, str, Path] = None,
    batch_size:int = 1024 * 1024 * 512,
    number:int = 0
    ):
    """

    batch_size: number of bytes to write to each file before creating a new 
      one. Can be larger than memory.

    uuid: One of two things.
    1. A defined uuid to handle multithreading, parallelism, filenames, etc.
    2. None, which means to handle all upstream items

    The file number within the register. New files will be created starting with this.

    """
    self._arrow_schema = None
    self.number = number
    if self.name is None:
      raise NameError("There must be a name defined on each reservoir subclass")
    if dir is None:
      dir = bookstack.root / self.name
    self.bookstack = bookstack
    truedir : Path = Path(dir)
    truedir.mkdir(exist_ok = True)
    self.batch_size = batch_size
    self.path = truedir
    self._id_map = None
    super().__init__(bookstack)

  def empty(self):
    # Check if any items have been created.
    if self.bookstack.uuid:
      p = (self.path / self.bookstack.uuid).with_suffix("." + self.format)
      return not p.exists()
    for _ in self.path.glob(f"**/*.{self.format}"):
      return False
    return True

  def build_cache(self):
    raise NotImplementedError("Class does not allow caching.")

  def _iter_cache(self):
    raise NotImplementedError("Class does not allow caching.")
  
  def id_in_metadata(self):
    try:
      _ = self.arrow_schema.field('@id')
      return True
    except KeyError:
      return False

  def __iter__(self):
    if not self.name in self.corpus.cache_set:
      yield from self._from_upstream()
    elif self.empty():
      logger.info(f"Now building {self.name}")
      yield from self._iter_and_cache()
    else:
      yield from self._iter_cache()

  @property
  def arrow_schema(self):
    if hasattr(self, "_arrow_schema"):
      return self._arrow_schema
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

class ArrowReservoir(Reservoir):
  format : str = "feather"

  def __init__(self, *args, max_size: int = 2 ** 20 * 128, **kwargs):
    """
    max_size: max size of objects to cache in memory.
    """
    self.max_size = max_size
    self._writer = None
    self.cache : list = []
    self._schema = None
    self.bytes = 0
    self._table = None
    super().__init__(*args, **kwargs)

  @property
  def writer(self):
    if self._writer is None:
      path = self.path / (self.uuid + ".feather")
      fout = pa.ipc.new_file(path, schema = pa.schema(self.arrow_schema), 
      options=pa.ipc.IpcWriteOptions(compression = "zstd"))
      self._writer = fout
    return self._writer

  def __enter__(self):
    return self

  def __exit__(self, type, value, tb):
    self.close()

  def close(self):
    if len(self.cache) > 0:
      self.flush_cache()
    if self._writer is not None:
      self.writer.close()
      self._writer = None
    
  def flush_cache(self):
    for batch in self.cache:
      self.writer.write(batch)
    self.cache = []
    self.bytes = 0

  def build_cache(self):
    for _ in self._iter_and_cache():
      pass

  def _iter_and_cache(self, mem_size = 1024 * 1024 * 128) -> Iterator[pa.RecordBatch]:
    """
    mem_size: Max size of memory record batch, bytes. Default 128 MB.
    """
    
    for batch in self._from_upstream():
      self.cache.append(batch)
      self.bytes += batch.nbytes
      if self.bytes > mem_size:
        self.flush_cache()
      yield batch
    self.close()
  
  def _from_upstream(self):
    raise NotImplementedError()

  @property
  def table(self) -> pa.Table:
    if self._table is not None:
      return self._table
    fin = self.path / (self.uuid + ".feather")
    logger.debug(f"Loading upstream feather table from {fin}")
    try:
      self._table = feather.read_table(fin)
    except pa.ArrowInvalid:
      fin.unlink()
      self.build_cache()
      self._table = feather.read_table(fin)
    except FileNotFoundError:
      self.build_cache()
      self._table = feather.read_table(fin)
    return self._table

  def _iter_cache(self) -> Iterator[pa.RecordBatch]:
    i = 0
    for batch in self.table.to_batches():
      i += 1
      yield batch

class ArrowLineChunkedReservoir(ArrowReservoir):
  def iter_with_ids(self, id_type = "@id"):
    ids = self.bookstack.ids[id_type]
    ix = 0
    for i_, batch in enumerate(super().__iter__()):
      n_here = len(batch)
      ids_here = ids[ix:ix + n_here]
      ix += n_here
      cols = [ids_here]
      for i, name in enumerate(batch.schema.names):
        cols.append(batch.column(i))
      try:
        tab = pa.table(cols, [id_type, *batch.schema.names])
        tab = tab.combine_chunks()
      except:
        logger.error(f"Error on batch {i_} {self.bookstack.uuid} length {n_here} with ids length {len(ids)} {ids[:10]}")
        logger.error(batch.to_pandas())
        raise
      yield from tab.to_batches()
      
  def iter_docs(self):
    for batch in self:
      names = [f.name for f in batch.schema]
      for i in range(len(batch)):
        yield {name: batch[name][i] for name in names}
#  def __iter__(self):
#    yield from self.iter_with_ids()

class ArrowIdChunkedReservoir(ArrowLineChunkedReservoir):
  
  """

  One document per row, with some additional guarantees:

  1. There is a single column, identified by self.name
  2. That column may be a struct.
  3. There is an upstream_arrays function that yields
     individual arrays.
  4. There is a process_batch function of type array -> array that
     generates individual column elements 
  
  """

  def __init__(self, origin, *args, **kwargs):
    assert hasattr(origin, "is_bookstack")
    self._upstream : Optional[Iterator[Any]] = None
    # May fill _upstream.
    super().__init__(origin, *args, **kwargs)

  @property
  def arrow_schema(self):
    # The schema is a list of whatever the base_type here is.
    return pa.schema({
      self.name: pa.list_(self.base_type)
    })

  def process_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
    raise NotImplementedError("Must define a batch -> batch method")

  def upstream(self):
    if self._upstream:
      return self._upstream
    else:
      raise

  def upstream_arrays(self) -> Iterator[pa.Array]:
    # Yields one array per document.
    for batch in self.upstream():
      col_1 = batch.columns[0]
      for row in col_1:
        # Coerce to an array type.
        yield row.values


  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    rows : List[List[pa.Array]] = []
    row_offsets = [0]
    cache_size = 0
    for array in self.upstream_arrays():
      value = self.process_batch(array)
      rows.append(value)
      row_offsets.append(row_offsets[-1] + len(value))      
      cache_size += value.nbytes
      if cache_size > self.bookstack.TARGET_BATCH_SIZE:
        values = pa.chunked_array(rows).combine_chunks()
        offsets = pa.array(row_offsets, pa.int32())
        batch = pa.RecordBatch.from_arrays(
          [pa.ListArray.from_arrays(offsets, values)],
          [self.name]
        )
        rows = []
        cache_size = 0
        yield batch
    values = pa.chunked_array(rows).combine_chunks()
    offsets = pa.array(row_offsets, pa.int32())
    yield pa.RecordBatch.from_arrays(
      [pa.ListArray.from_arrays(offsets, values)],
      [self.name]
    )

  def iter_docs(self):
    for batch in self:
      for row in batch[self.name]:
        yield pa.record_batch(row.values.flatten(), [m.name for m in self.base_type])

  def iter_with_ids(self, id = "_ncid"):
    # Slap an ID in front of the list.
    ids = self.bookstack.ids[id]
    offset = pa.scalar(0, pa.int32())

    for batch in self:
      indices = batch[self.name].value_parent_indices()      
      ids = pc.take(ids, pc.add(indices, offset))
      offset = pc.add(offset, pa.scalar(len(batch)))
      batch = pa.RecordBatch.from_struct_array(batch[self.name].flatten())
      yield pa.record_batch([indices, *batch.columns],
             [id, *[f.name for f in batch.schema]])

class SRP_set(ArrowReservoir):
  """ 
  Embed a set of tokencounts into a fixed high-dimensional
  space for comparisons. You've got to persist this to disk, because it 
  would be a gratuitous waste of energy not to and I'm not cool with that.
  """
  name = "SRP"
  schema = pa.schema({
    "SRP": pa.list_(pa.float32(), int(1280)),
    "SRP_bits": pa.binary(int(1280) // 8)
  })

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if not "SRP" in self.corpus.cache_set:
      logger.warning("Adding SRP to cache set")
      self.corpus.cache_set.add("SRP")

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    # Yields 1-row batches. These can be assembled later into longer ones; the 
    # SRP step is expensive enough that it's NBD to wait.

    hasher = SRP.SRP(dim = int(1280), cache = True)
    for batch in self.bookstack.get_transform("token_counts"):
      # At a first pass, 
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

    logger.info("Building cache")
    sink = self.open_writer()
    for batch in self._from_upstream():
      assert(batch.schema.metadata)
      yield batch
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

