from .corpus import Corpus
from pathlib import Path
import pyarrow as pa
from pyarrow import feather
SRP = None
Vector_file = None

def embed_to_SRP(corpus: Corpus, filepath, dims = 1280, **kwargs):
  """ 
  Embed a set of tokencounts into a fixed high-dimensional
  space for comparisons. You've got to persist this to disk, because it 
  would be a gratuitous waste of energy not to and I'm not cool with that.
  """
  global SRP
  filepath = Path(filepath)
  if SRP is None:
    import SRP
  hasher = SRP.SRP(dim = int(dims), cache = False)
  schema = pa.schema({
    "id": pa.string(),
    "SRP": pa.list_(pa.float32(), int(dims)),
    "SRP_bits": pa.binary(int(dims) // 8)
  })
  for metadata, element in corpus.feature_counts(**kwargs):
    id = metadata['id']
    tokens = element.column('token').to_pylist()    
    counts = element.column('count').to_numpy()
    hashed = hasher.stable_transform(words = tokens, counts = counts)
    bits = np.packbits(hashed > 0)
