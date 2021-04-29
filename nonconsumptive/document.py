import pyarrow as pa
from collections import Counter
from typing import List, Set, Dict, Tuple, Optional
import re
import json
from pathlib import Path
import gzip
from .wordcounting import chunked_wordcounts, wordcounts

class BaseDocument(object):
  def __init__(self, path):
    self._path = path

  @property
  def path(self):
    if self._path is not None:
      return self._path
    return self.corpus.path_to(self.id)

  @property
  def id(self):
    if self._id:
      return self._id
    id = self._path.stem
    for compression in ["gz", "lz4", "bz2"]:
      if id.endswith(compression):
        id = id.rstrip("." + compression)
    for filetype in ["txt", "md", "rtf", "xml"]:
      if id.endswith(filetype):
        id = id.rstrip("." + filetype)
    return id

class Document(BaseDocument):

  def __init__(self, corpus, id = None, path = None):
    self.corpus = corpus
    self._id = id
    self._path = path    

  @property
  def full_text(self, mode = "r"):
    document = self.path
    if document.suffix == ".gz":
      fin = gzip.open(document, "rt")
    else:
      fin = document.open()
    return fin.read()

  @property
  def filename(self):
    try:
      return self.metadata['filename']
    except KeyError:
      return self.id

  @property
  def tokens(self) -> pa.StringArray:
    # Could get a lot fancier here.
    return self.corpus.tokenization.get_tokens(self.id)

  def tokenize(self) -> pa.StringArray:
    return tokenize(self.full_text)

  def ngrams(self, n) -> List[Tuple[str]]:
      vars = [self.tokens[n-i:i-n-1] for i in range(0, n)]
      return zip(*vars)

  def __repr__(self):
    return str(self)

  def __str__(self):
    return "<DOCUMENT> " +  json.dumps(self.metadata)

  @property
  def metadata(self):
    return self.corpus.metadata.get(self.id)

  def chunked_wordcounts(self, chunk_size) -> pa.RecordBatch:
    """
    As with the core elements 
    """
    return chunked_wordcounts(self.tokens, chunk_size)

  @property
  def wordcounts(self) -> pa.RecordBatch:
    c = pa.RecordBatch.from_struct_array(self.tokens.value_counts())
    return pa.record_batch(
      [c['values'],c['counts'].cast(pa.uint32())], 
    schema = pa.schema(
      {"token": pa.utf8(),
      "count": pa.uint32()},
      metadata={'nc_metadata': json.dumps(self.metadata)})
    )

def tokenize(string) -> pa.Array:
    return pa.array(re.findall(r"[\w^_]+|[^\w\s]+", string))

def token_counts(tokens: pa.Array, id: str) -> pa.RecordBatch:
  c = pa.RecordBatch.from_struct_array(tokens.value_counts())
  return pa.record_batch(
    [c['values'],c['counts'].cast(pa.uint32())], 
  schema = pa.schema(
    {
      "token": pa.utf8(),
      "count": pa.uint32()
    },
    metadata={'id': id})
    )
