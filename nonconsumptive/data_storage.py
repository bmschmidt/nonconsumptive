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

logger = logging.getLogger("nonconsumptive")

from typing import Callable, Iterator, Union, Optional, List
from .prefs import prefs
from .utils import ConsumptionError

SRP = None

class BatchIDIndex(object):
  def __init__(self, corpus, uuid, batch_size = 1024 * 16):
    self.uuid = uuid
    self.corpus = corpus
    self.dir = corpus.root / "build" / "batch_indices"
    self.dir.mkdir(exist_ok = True, parents = True)
    self.fn = self.dir / f"{uuid}.feather"
    self.batch_size = batch_size
    self.cache = []
    self.ncid_lookup = None
    self._fout = None
    self.schema = pa.schema({
          "@id": pa.string(),
          "_ncid": pa.uint32()})
  def exists(self):
    return self.fn.exists()
  
  @property
  def fout(self):
    if self._fout is not None:
      return self._fout
    logger.warning("Here we are now!")

    self._fout = ipc.new_file(self.fn, self.schema)

    return self._fout

  def fill(self):
    """
    Create a child index and fill it up at once.
    """
    logger.warning("Here we go!")
    with BatchIDIndex(self.corpus, self.uuid, self.batch_size) as holder:
      logger.warning("Here we are!")
      for id, _ in self.corpus.texts:
        holder.push(id)
      
    
  def iter_ids(self, which = '@id'):
    if not self.fn.exists():
      logger.warning("ID lookup requested, but none found: "
                      "materializing one inefficiently with a full pass through "
                      "the data")
      self.fill()
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
  def __init__(self, corpus: "Corpus"):
    self.uuid = corpus.uuid
    self.corpus = corpus

class Reservoir(Node):
  __doc__ = """
  A store for holding, writing, and accessing tables on disk.
  """
  name = None
  def __init__(self,
    corpus,
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
      dir = corpus.root / self.name
    dir.mkdir(exist_ok = True)
    self.batch_size = batch_size
    self.path = dir
    self._id_map = None
    super().__init__(corpus)

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
          yield b
      else:
        for b in self._from_upstream():
          yield b
    else:
      for batch in self.iter_cache():
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

  def __init__(self, *args, max_size: int = 2 ** 20 * 128, **kwargs):
    """
    max_size: max size of objects to cache in memory.
    """
    self.max_size = max_size
    self._writer = None
    self.cache = []
    self._schema = None
    self.bytes = 0
    super().__init__(*args, **kwargs)

  @property
  def writer(self):
    if self._writer is None:
      path = self.path / (self.uuid + ".ipc")
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
    
  def flush_cache(self):
    for batch in self.cache:
      self.writer.write(batch)
      self.bytes = 0

  def build_cache(self, mem_size = 1024 * 1024 * 128) -> Iterator[pa.RecordBatch]:
    """
    mem_size: Max size of memory record batch, bytes. Default 128 MB.
    """
    
    for batch in self._from_upstream():
      self.cache.append(batch)
      self.bytes += batch.nbytes
      if self.bytes > mem_size:
        self.flush_cache()
      yield batch
    self.flush_cache()
    self.writer.close()
  
  def iter_cache(self) -> Iterator[pa.RecordBatch]:
    if self.uuid is None:
      fs = [*self.path.glob("*.ipc")]
    else:
      # Just the one file
      fs = [self.path / (self.uuid + ".ipc")]
    for file in fs:
      logging.warning(file)
      fin = pa.ipc.open_file(file)
      for i in range(fin.num_record_batches):
        batch = fin.get_batch(i)
        yield batch

class ArrowIDChunkedReservoir(ArrowReservoir):
  """
  An arrow reservoir where each record batch corresponds to a single uuid
  in the order documented at root / batch_indices / uuuid.feather
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def process_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
    raise NotImplementedError("Must define a batch -> batch method")

  def iter_with_ids(self, id_type = "@id"):
    if id_type == "_ncid":
      dtype = pa.uint32()
    else:
      dtype = pa.string()
    new_schema = self.arrow_schema.insert(0, pa.field(id_type, dtype))
    # TODO: zip(strict=True) when only py 3.10 or later is supported?

    for id, batch in zip(self.corpus.batch_ids(self.uuid, id_type), self):
      id = pa.array([id] * len(batch), dtype)
      yield pa.RecordBatch.from_arrays(
        [id, *batch.columns],
        schema = new_schema
        )

  def process_and_cache(self, batch):
    transformed = self.process_batch(batch)
    self.cache.push(transformed)
    self.bytes += transformed.nbytes
    if self.bytes > chunk_size:
      self.flush_cache()
    return transformed

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    for batch in self.upstream:
      yield self.process_batch(batch)

  @property
  def id_location_map(self):
    raise NotImplementedError("This functionality is temporarily removed")
    if self.empty():
      return None
    dest = Path(self.path / (self.uuid + "id_location_lookup.parquet"))
    if dest.exists():
      pass
      dest.unlink()
    tab = pa.table(
      {
        'batch': batch_numbers
      })
    tab = tab.take(pc.sort_indices(tab['_ncid']))
    parquet.write_table(tab, dest)
    return dest

  def get_id(self, id) -> pa.RecordBatch: 
    raise NotImplementedError("This functionality is temporarily removed")
    if self.empty():
      for r in self.build_cache():
        pass
    rows = parquet.read_table(self.id_location_map, filters = [(('@id', '==', id))])
    dicto = rows.to_pydict()
    file = dicto['file'][0] + self.format
    batch = dicto['batch'][0]
    with ipc.open_file(self.path / file) as f:
      return f.get_batch(batch)

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
    global SRP
    if SRP is None:
      import SRP    
    self.corpus.cache_set.add("SRP")

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    # Yields 1-row batches. These can be assembled later into longer ones; the 
    # SRP step is expensive enough that it's NBD to wait.

    hasher = SRP.SRP(dim = int(dims), cache = True)
    for batch in self.corpus.token_counts:
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

    print("Building cache")
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

