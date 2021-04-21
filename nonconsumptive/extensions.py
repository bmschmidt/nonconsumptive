from .corpus import Corpus
from pathlib import Path
SRP = None
Vector_file = None

def embed_to_SRP(corpus: Corpus, filepath, dims = 1280, **kwargs):
  """ 
  Embed a set of tokencounts into a fixed high-dimensional
  space for comparisons.
  """
  global SRP
  global Vector_file
  filepath = Path(filepath)
  if SRP is None:
    from SRP import Vector_file
    import SRP
  hasher = SRP.SRP(dim = int(dims), cache = False)
  binary_vectors = Vector_file(filepath.with_suffix(".bits"), dims = dims, precision = "binary", mode = "w")
  with Vector_file(filepath, dims = dims, mode = "w") as vectors:
    for metadata, element in corpus.feature_counts(**kwargs):
      id = metadata['id']
      tokens = element.column('token').to_pylist()    
      counts = element.column('count').to_numpy()
      hashed = hasher.stable_transform(words = tokens, counts = counts)
      vectors.add_row(id, hashed)
      binary_vectors.add_row(id, hashed)