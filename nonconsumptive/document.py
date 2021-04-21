import pyarrow as pa
from functools import cached_property
from collections import Counter
from typing import List, Set, Dict, Tuple, Optional
import re
import json
from pathlib import Path
import gzip
from .wordcounting import chunked_wordcounts

class Document(object):

  def __init__(self, corpus, id = None, path = None):
    self.corpus = corpus
    self._id = id
    self._path = path    

  @property
  def path(self):
    if self._path is not None:
      return self._path
    return self.corpus.path_to(self.id)

  @cached_property
  def id(self):
    if self._id:
      return self._id
    id = self._path.stem
    for compression in ["gz", "lz4", "bz2"]:
      id = id.rstrip("." + compression)
    for filetype in ["txt", "md", "rtf", "xml"]:
      id = id.rstrip("." + filetype)
    return id

  @cached_property
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
    return pa.array(re.findall("[\w^_]+|[^\w\s]+", self.full_text))

  def ngrams(self, n) -> List[Tuple[str]]:
      vars = [self.tokens[n-i:i-n-1] for i in range(0, n)]
      return zip(*vars)

  def __repr__(self):
    return str(self)

  def __str__(self):
    return "<DOCUMENT> " +  json.dumps(self.metadata)

  @cached_property
  def metadata(self):
    return self.corpus.metadata.get(self.id)

  def chunked_wordcounts(self, chunk_size) -> pa.RecordBatch:
    """
    As with the core elements 
    """
    return chunked_wordcounts(self.tokens, chunk_size)

  @cached_property
  def wordcounts(self) -> pa.RecordBatch:
    c = pa.RecordBatch.from_struct_array(self.tokens.value_counts())
    return pa.record_batch(
      [c['values'],c['counts'].cast(pa.uint32())], 
    schema = pa.schema(
      {"token": pa.utf8(),
      "count": pa.uint32()},
      metadata={'nc_metadata': json.dumps(self.metadata)})
    )