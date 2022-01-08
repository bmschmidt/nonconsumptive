from .corpus import Corpus
from pathlib import Path
import pyarrow as pa
from pyarrow import ipc
from .arrow_helpers import batch_id
from numpy import packbits
import types
import SRP
from SRP import Vector_file
import logging

def embed_to_SRP(corpus: Corpus, filepath = None, dims = 1280, flush_every = 512, **kwargs):
  """ 
  Embed a set of tokencounts into a fixed high-dimensional
  space for comparisons. You've got to persist this to disk, because it 
  would be a gratuitous waste of energy not to and I'm not cool with that.
  """
  global SRP
  if filepath is None:
    filepath = corpus.root / "SRP.feather"
  filepath = Path(filepath)
  if filepath.exists():
    pass #return filepath
  if SRP is None:
    import SRP
  hasher = SRP.SRP(dim = int(dims), cache = True)
  schema = pa.schema({
    "id": pa.string(),
    "SRP": pa.list_(pa.float32(), int(dims)),
    "SRP_bits": pa.binary(int(dims) // 8)
  })
  ids = []
  bits = []
  hashed = []
  fout = ipc.new_file(filepath, schema = schema)
  for i, batch in enumerate(corpus.unigrams()):
    id = batch_id(batch)
    tokens = batch['token'].to_pylist()
    counts = batch['count'].to_numpy()
    hash_rep = hasher.stable_transform(words = tokens, counts = counts)
    bit_rep = packbits(hash_rep > 0).tobytes()
    ids.append(id)
    bits.append(bit_rep)
    hashed.append(hash_rep)
    if i > 0 and i % flush_every == 0:
      logging.debug("flushing", ids)
      fout.write_batch(pa.record_batch([
        pa.array(ids, schema[0].type),
        pa.array(hashed, schema[1].type),
        pa.array(bits, schema[2].type)
      ],
      schema = schema
      ))
      ids = []
      bits = []
      hashed = []
  if len(ids):
    fout.write_batch(pa.record_batch([
      pa.array(ids, schema[0].type),
      pa.array(hashed, schema[1].type),
      pa.array(bits, schema[2].type)
      ],
      schema = schema
    ))
  fout.close()
  return filepath

