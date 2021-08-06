import pyarrow as pa
from pyarrow import parquet
import pandas as pd # For now!
from pathlib import Path

class ConsumptionError(Exception):
  pass

import numpy as np

# First text is sui generis, so call it dist 0.
def distances(all_texts):
  # This could be much, much more efficient by using bit operations or something,
  # probably.
  dists = np.zeros(len(all_texts), np.int32)
  for i in range(1, len(all_texts)):
    assigned = False
    for j_, (a, b) in enumerate(zip(all_texts[i], all_texts[i - 1])):
      if a != b:
        dists[i] = j_
        assigned = True
        break
    if not assigned:
      # If ID1 is distinguished by subsequent characters
      # after ID2 ends, or vice-versa.
      dists[i] = j_ + 1
  return dists

def breaks(array, min_length = 2**8, max_length = 2**14):
  if len(array) <= max_length:
      return [len(array)]
  min = np.argmin(array[min_length:-min_length])
  l = array[:(min + min_length)]
  r = array[(min + min_length):]
  return [*breaks(l, min_length, max_length), *breaks(r, min_length, max_length)]

def chunk_ids(ids, min_length = 2**8, max_length = 2**14):
  """
  Breaks list of ids into vaguely reasonable stacks by splitting wherever ids are very
  different from each other.

  ids: a python list of ids. (Should support pyarrow, eventually).


  returns: a pyarrow list with one element per stack.
  """
  ids = sorted(ids)
  dists = distances(ids)
  broken = breaks(dists, min_length, max_length)
  indices = np.cumsum(np.array(broken))
  indices = np.insert(indices, 0, 0)
  chunked = pa.ListArray.from_arrays(indices, ids)
  return chunked

class BookstackBuilder():
  def __init__(self, dir : Path, max_docs_in_file = float("Inf"), 
    max_docs_in_batch = 128, schema : pa.Schema = None):
    """
    max_docs: if a parquet file is more than this many 
    documents, create a new one.
    """
    self.dir = Path(dir)
    self._current_writer = None
    self.schema = schema
    self.max_docs_in_file = max_docs_in_file
    self.current_docs_in_file = 0
    self.file_num = 0
    self.max_docs_in_batch = max_docs_in_batch
    self.cache = []

  @property
  def current_writer(self):
    if self._current_writer is not None:
      return self._current_writer
    else:
      target = self.dir / f"{self.file_num:05d}.parquet"
      if target.exists():
        raise FileExistsError(f"{target} already exists")
      self._current_writer = parquet.ParquetWriter(target, schema = self.schema, compression = "zstd")
      return self._current_writer
  def add(self, metadata : dict, **kwargs):
    named_args = [k for k in kwargs.keys()]
    for key in named_args:
      try:
        assert key in ['text', 'tokenization']
      except:
        raise NameError("You can only pass 'text' or 'tokenization'")
      metadata[f'nc:{key}'] = kwargs[key]
    self.cache.append(metadata)
    if len(self.cache) > self.max_docs_in_batch:
      self.flush()
    self.current_docs_in_file += 1
    if self.current_docs_in_file > self.max_docs_in_file:
      self.advance_parquet_file()

  def flush(self):
    pandafied = pd.DataFrame(self.cache)
    for k in self.schema:
      if not k.name in pandafied.columns:
        print(k)
        pandafied[k.name] = np.NaN
    tb = pa.table(pandafied, schema = self.schema)
    self.current_writer.write_table(tb)
    self.cache = []
    
  def advance_parquet_file(self):
    self._current_writer = None
    self.file_num += 1
