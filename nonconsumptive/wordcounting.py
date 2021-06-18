import pyarrow as pa
from collections import Counter
import numpy as np

def chunked_wordcounts(tokens, chunk_size, chunk_name = "chunk") -> pa.RecordBatch:

  a_chunks = []
  a_tokens = []
  a_counts = []

  for i, chunklist in enumerate(chunk_tokens(tokens, chunk_size)):
    batch = wordcounts(chunklist)
    a_tokens.append(batch.column("token"))
    a_counts.append(batch.column("count"))
    a_chunks.append(pa.array(np.full(len(batch), i, dtype = "u4")))

  return pa.record_batch(
    [
      pa.concat_arrays(a_chunks),
      pa.concat_arrays(a_tokens),
      pa.concat_arrays(a_counts)
    ],
    schema = pa.schema(
      {
        chunk_name: pa.uint32(),
        "token": pa.utf8(), 
        "count": pa.uint32()
      }
    )
  )

def chunk_tokens(tokens, chunk_size):
  """ 
  yield tokens split up into groups of chunk_size as closely as possible.


  PERFORMANCE--uses python list, would be faster with zero-copy array.
  """

  overage = len(tokens) % chunk_size
  if (overage < chunk_size * 2 / 3):
    overage += chunk_size
  half_overage = overage // 2

  total = len(tokens)
  pos = 0
  if total > 0:
    yield tokens[:half_overage]
    pos += half_overage
  while total - pos > chunk_size + half_overage:
    # Middle chunks are the chunksize.
    yield tokens[pos:(pos+chunk_size)]
    pos += chunk_size 
  assert total - pos == half_overage + 1
  yield tokens[pos:]

def wordcounts(tokens) -> pa.RecordBatch:
  counts : Counter = Counter(tokens)
  keys = counts.keys()
  values = counts.values()
  return pa.record_batch([
    pa.array(keys),
    pa.array(values)
    ], 
    schema = pa.schema(
      {"token": pa.utf8(), "count": pa.uint32()}
    )
  )