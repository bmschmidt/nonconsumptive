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
    truedir : Path = Path(dir)
    truedir.mkdir(exist_ok = True)
    self.batch_size = batch_size
    self.path = truedir
    self._id_map = None
    super().__init__(bookstack)

  def empty(self):
    # Check if any items have been created.
    if self.uuid:
      return not (self.path / f"uuid").with_suffix("." + self.format).exists()
    for _ in self.path.glob(f"**/*.{self.format}"):
      return False
    return True

  def full(self):
    # Check if every item in the metadata has been created.
    pass

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
      logger.info(f"Building {self.name}")
      yield from self._iter_and_cache()
    else:
      yield from self._iter_cache()

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
  format : str = "NOFORMAT"
      

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
    logger.info(f"Loading upstream feather table from {fin}")
    self._table = feather.read_table(fin)
    return self._table

  def _iter_cache(self) -> Iterator[pa.RecordBatch]:
    for batch in self.table.to_batches():
      yield batch

class ArrowLineChunkedReservoir(ArrowReservoir):
  def iter_with_ids(self, id_type = "@id"):
    ids = self.bookstack.ids[id_type]
    ix = 0
    for batch in super().__iter__():
      n_here = len(batch)
      ids_here = ids[ix:ix + n_here]
      ix += n_here
      cols = [ids_here]
      for i,name in enumerate(batch.schema.names):
        cols.append(batch.column(i))
      tab = pa.table(cols, [id_type, *batch.schema.names]).combine_chunks()
      yield from tab.to_batches()

  def __iter__(self):
    yield from self.iter_with_ids()

class ArrowIDChunkedReservoir(ArrowReservoir):
  """
  An arrow reservoir where each record batch corresponds to a single uuid
  in the order documented at {root}/build/bookstack_index/{uuuid}.feather. Allows for fast iteration without the
  overhead of actually passing the ids through the pipeline.
  """

  def __init__(self, origin, *args, **kwargs):
    assert hasattr(origin, "is_bookstack")
    self._upstream : Optional[Iterator[Any]] = None
    super().__init__(origin, *args, **kwargs)

  def process_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
    raise NotImplementedError("Must define a batch -> batch method")

  def upstream(self):
    if self._upstream:
      return self._upstream
    else:
      raise

  def iter_alongside_ids(self, id_type = "@id") -> Iterator[Tuple[str, pa.RecordBatch]]:
    # Iterate
    for id, batch in zip(self.bookstack.ids[id_type], self):
      yield id, batch

  def iter_with_ids(self, id_type = "@id"):
    if id_type == "_ncid":
      dtype = pa.uint32()
    else:
      dtype = pa.string()
    new_schema = self.arrow_schema.insert(0, pa.field(id_type, dtype))
    # TODO: zip(strict=True) when only py 3.10 or later is supported?
    for id, batch in zip(self.bookstack.ids[id_type], self):
      id_list = np.full(len(batch), id.as_py())
      id = pa.array(id_list, dtype)
      yield pa.RecordBatch.from_arrays(
        [id, *batch.columns],
        schema = new_schema
        )

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    for batch in self.upstream():
      yield self.process_batch(batch)

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

