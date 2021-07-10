import pyarrow as pa
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
from pyarrow import parquet, feather, RecordBatch 
from pyarrow import compute as pc
from pyarrow import ipc
from .data_storage import ArrowIdChunkedReservoir, ArrowLineChunkedReservoir, ArrowReservoir
import logging
logger = logging.getLogger("nonconsumptive")
import polars as pl

try:
  import blingfire
  from blingfire import blingfire as bf # C bindings
  from ctypes import create_string_buffer, byref, c_int, c_char_p, cdll
except:
  blingfire = None
  logger.warning("Couldn't find blingfire, falling back to regex tokenization. `pip install blingfire` for faster tokenization.")

def tokenize(text):
  return blingfire.text_to_words(text).split(" ")

def tokenize_arrow(texts : pa.Array) -> pa.Array:
  # Convert 
  # Altered from the blingfire source code at https://github.com/microsoft/BlingFire/blob/master/dist-pypi/blingfire/__init__.py
  # to work directly on pyarrow arrays without casting to and from Python.

  # get the UTF-8 bytes
  as_bytes : List[bytes] = []
  for text in texts:
    s_bytes = text.as_buffer().to_pybytes()

    # allocate the output buffer
    o_bytes = create_string_buffer(len(s_bytes) * 3)
    o_bytes_count = len(o_bytes)
    # identify words
    o_len = bf.TextToWords(c_char_p(s_bytes), c_int(len(s_bytes)), byref(o_bytes), c_int(o_bytes_count))

    # check if no error has happened
    if -1 == o_len or o_len > o_bytes_count:
      as_bytes.append(None)
    else:
      as_bytes.append(o_bytes.value)

  return pc.split_pattern(
    pa.array(as_bytes, pa.string()),
  pattern = " ")

class Text(ArrowLineChunkedReservoir):
  """
  Represents text as an arrow string.
  """
  name = "text"
  arrow_schema = pa.schema({"text": pa.string()})

  def _from_upstream(self):
    ids = self.bookstack.ids['@id']
    current_batch = []
    current_size = 0
    for text in self.corpus.text_input.iter_texts_for_ids(ids):
      current_batch.append(text)
      current_size += len(text) # Use char length as proxy for byte length. Bad for Russian, worse for Chinese.
      if current_size >= self.bookstack.TARGET_BATCH_SIZE:
        yield pa.RecordBatch.from_arrays([pa.array(current_batch), pa.string()], ["text"])
        current_batch = []
        current_size = 0
    yield pa.RecordBatch.from_arrays([pa.array(current_batch, pa.string())], ["text"])

class Tokenization(ArrowLineChunkedReservoir):
  name = "tokenization"
  arrow_schema = pa.schema({"tokenization": pa.list_(pa.string())})

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def process_batch(self, text = str) -> RecordBatch:
    tokens = tokenize(text)
    return pa.record_batch([tokens], self.arrow_schema)

  def _from_upstream(self) -> Iterator[RecordBatch]:
    for batch in self.bookstack.get_transform("text"):
      array = tokenize_arrow(batch['text'])
      yield pa.RecordBatch.from_arrays([array], ['tokenization'])

class SRP_Transform(ArrowLineChunkedReservoir):
  name = "srp_transform"

class DocumentLengths(ArrowLineChunkedReservoir):
  name = "document_lengths"
  arrow_schema = pa.schema({"nwords": pa.uint32()})

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    counts = []
    nbytes = 0
    for batch in self.bookstack.get_transform("tokenization"):
      # Read the lengths directly from the list offsets.
      chunk_counts = pc.subtract(
        batch['tokenization'].offsets[1:], 
        batch['tokenization'].offsets[:-1]).cast(pa.uint32())
      counts.append(chunk_counts)
      nbytes += chunk_counts.nbytes
      if nbytes > self.bookstack.TARGET_BATCH_SIZE:
        yield pa.record_batch([pa.chunked_array(counts).combine_chunks()], ['nwords'])
        counts = []
        nbytes = 0
    if len(counts):
      yield pa.record_batch([pa.chunked_array(counts).combine_chunks()], ['nwords'])

class TokenCounts(ArrowIdChunkedReservoir):
  """
  A TokenCounts objects caches counts by unigram for 
  each document. 
  """

  name = "token_counts"
  base_type = pa.struct([
    pa.field("token", pa.string()), 
    pa.field("count", pa.uint32())])
  ngrams = 1

  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack, *args, **kwargs)
    self._upstream = self.bookstack.get_transform("tokenization")

  def process_batch(self, words: pa.Array) -> pa.Array:
    # Count the words, and cast to uint32.
    words, counts = pc.value_counts(words).flatten()
    counts = counts.cast(pa.uint32())
    return pa.StructArray.from_arrays([words, counts], ["token", "count"])

class Ngrams(ArrowIdChunkedReservoir):
  def __init__(self, bookstack, ngrams: int, end_chars: List[str] = [], beginning_chars: List[str] = [], **kwargs):
    """

    Creates an (optionally cached) iterator over record batches of ngram counts.
    Each batch is a single document, with columns ['token1', 'token2', ... , 'token{n}', count]

    ngrams: an integer. Size of the ngrams to construct.
    end_chars: a list of regular expression (re2 compatible) to treat as the *end* of an n-gram.
               For instance, [r"[\\.\\?\\!]"] would attach sentence-ending punctuators to the end of
               n-grams but not the beginning, and not allow ngrams from separate sentence.
    beginning_chars: a list of regular expression (re2 compatible) to treat as the *beginning* of an n-gram.
               For instance, ["“", "<"] would attach opening curly quotes and angle brackets to n-grams with the letters that
               follow. (This is provided experimentally, may be removed because it doesn't seem very useful.
               But perhaps there are languages where it helps.)

    *args, **kwargs: passed to ArrowReservoir.

    """
    self.ngrams = ngrams
    self.name = f"{ngrams}gram_counts"
    self.end_chars = end_chars
    self.beginning_chars = beginning_chars
    self._arrow_schema = None
    super().__init__(bookstack, **kwargs)

  def _from_local(self):
    pass

  @property
  def arrow_schema(self) -> pa.Schema:
    if self._arrow_schema:
      return self._arrow_schema
    labs = {}
    for n in range(self.ngrams):
      labs[f"token{n + 1}"] = pa.list_(pa.utf8())
      labs['count'] = pa.list_(pa.uint32())
    self._arrow_schema = pa.schema(labs)
    return self._arrow_schema
  
  def _from_upstream(self) -> Iterator[RecordBatch]:
    ngrams = self.ngrams
    for batch in self.bookstack.tokenization():
      zengrams = ngrams - 1 # Zero-indexed ngrams--slightly more convenient here.
      tokens = batch['token']
      cols = {}
      for n in range(ngrams):
          if n == 0:
              cols["token1"] = tokens[:-zengrams]
          elif n == ngrams - 1:
              cols[f"token{n+1}"] = tokens[n:]
          else:
              cols[f"token{n+1}"] = tokens[n:-(zengrams-n)]
      ngram_tab = pa.table(cols)
      if len(ngram_tab) < ngrams:
        empty : List = [[]]
        # Still register that we saw the id even if it's not long enough for ngrams
        yield pa.RecordBatch.from_arrays(empty * len(self.arrow_schema.names), schema=self.arrow_schema)
        continue
      # Temporarily pass through pandas because https://github.com/pola-rs/polars/issues/668
      t = pl.from_pandas(ngram_tab.to_pandas())
      counts = t.groupby([f"token{i+1}" for i in range(ngrams)])\
          .agg([pl.count("token1").alias("count")])

      other_cols = []
      for i in range(ngrams):
          other_cols.append(counts[f'token{i+1}'].to_arrow().cast(pa.string()))
      other_cols.append(counts['count'].to_arrow())
      yield pa.RecordBatch.from_arrays(other_cols, schema=self.arrow_schema)


class Bigrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  name = "bigrams"

  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=2, **kwargs)

class Trigrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=3, **kwargs)

class Quadgrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==4.
  """

  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=3, **kwargs)

class Quintgrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==5.
  """
  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=5, **kwargs)


class EncodedCounts(ArrowReservoir):

  def __init__(self, upstream: Union[TokenCounts, Ngrams], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._upstream = upstream
    self.ngrams = upstream.ngrams
    self._arrow_schema : pa.Schema = self.build_schema()

  def build_schema(self) -> pa.Schema:
    schema = {
      "_ncid": pa.uint32()
    }
    if (self.ngrams == 1):
      schema['wordid'] = pa.uint32()
    else:
      for i in range(self.ngrams):
        schema[f"word{i + 1}"] = pa.uint32()
    schema['count'] = pa.uint32()
    return pa.schema(schema)

  @property
  def wordids(self) -> pa.Array:
    return self.corpus.total_wordcounts["token"]

  def process_batch(self, batch) -> pa.RecordBatch:

    derived = []

    for f in batch.schema:
      name = f.name
      if name == "token":
        derived.append(pc.index_in(batch[name], value_set = self.wordids).cast(pa.uint32()))
      elif name.startswith("word"):
        derived.append(pc.index_in(batch[name], value_set = self.wordids).cast(pa.uint32()))
      else:
        derived.append(batch[name])
    return pa.record_batch(derived, schema=self.arrow_schema)

  def iter_docs(self):
    yield from self

  def _from_upstream(self) -> pa.Table:
    """
    Rather than work a batch at a time, try to do at least 10MB at once,
    so the join can be a bit more efficient.
    """

    for batch in self.bookstack.get_transform("token_counts").iter_with_ids("_ncid"):
      yield self.process_batch(batch)

class EncodedUnigrams(EncodedCounts):
  name : str = "ncid_wordid"
  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack.get_transform("token_counts"), bookstack, *args, **kwargs)


transformations = {
  'text': Text,
  'document_lengths': DocumentLengths,
  'token_counts': TokenCounts,
  'tokenization': Tokenization,
  'quintgrams': Quintgrams,
  'bigrams': Bigrams,
  'trigrams': Trigrams,
  'quadgrams': Quadgrams,
  'encoded_wordcounts': EncodedUnigrams,
  'encoded_unigrams': EncodedUnigrams
#  'encoded_bigrams': EncodedBigrams
}
