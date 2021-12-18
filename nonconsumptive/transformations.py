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

# Placeholders.
SRP = None
np = None

try:
  import blingfire
  from blingfire import blingfire as bf # C bindings
  from ctypes import create_string_buffer, byref, c_int, c_char_p, cdll
except ImportError:
#  raise ImportError("Non-blingfire methods are not available: pip install blingfire")
  blingfire = None
  logger.warning("Couldn't find blingfire, falling back to regex tokenization. `pip install blingfire` for faster tokenization.")
except OSError:
  blingfire = None
  logger.warning("OS error on blingfire import--known problem with M1 Macintoshes for now.")
def tokenize(text):
  return blingfire.text_to_words(text).split(" ")

def tokenize_without_blingfire(texts : pa.Array):
  return pc.split_pattern_regex(texts, pattern = r"[^\p{L}]")

def tokenize_arrow(texts : pa.Array) -> pa.Array:
  # Convert 
  # Altered from the blingfire source code at https://github.com/microsoft/BlingFire/blob/master/dist-pypi/blingfire/__init__.py
  # to work directly on pyarrow arrays without casting to and from Python.

  # get the UTF-8 bytes
  if blingfire is None:
    return tokenize_without_blingfire(texts)

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

  def _from_stacks(self):
    stack = self.corpus.input_bookstacks / f"{self.uuid}.parquet"
    fin = parquet.ParquetFile(stack)
    # Low batch size because books are long. For shorter texts, a larger batch size
    # might be marginally better.
    for i, batch in enumerate(fin.iter_batches(columns = ['nc:text'], batch_size = 250)):
      logging.debug(f"Yielding batch {i} from {stack}")
      yield pa.record_batch([batch['nc:text']], self.arrow_schema)
      
  def _from_upstream(self):
    if self.corpus.input_bookstacks:
      yield from self._from_stacks()
      return
    ids = self.bookstack.ids['@id']
    current_batch = []
    current_size = 0
    for text in self.corpus.text_input.iter_texts_for_ids(ids):
      current_batch.append(text)
      current_size += len(text) # Use char length as proxy for byte length. Bad for Russian, worse for Chinese.
      if current_size >= self.bookstack.TARGET_BATCH_SIZE:
        yield pa.RecordBatch.from_arrays([pa.array(current_batch, pa.string())], ["text"])
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

class SRP_Transform(ArrowIdChunkedReservoir):
  name = "SRP"

  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack, *args, **kwargs)
    self._upstream = self.bookstack.get_transform("token_counts")
    global SRP
    global np
    if SRP is None:
      import SRP
    if np is None:
      import numpy as np      
    
  base_type = pa.list_(pa.float32(), 1280)

  @property
  def hasher(self):
    global SRP
    global np
    if SRP is None:
      import SRP
    if np is None:
      import numpy as np        
    if "SRP_hasher" in self.corpus.slots:
      return self.corpus.slots["SRP_hasher"]
    # Because a hasher contains a cache, it's bound to the 
    # full corpus rather than the bookstack to avoid 
    # unnecessarily re-learning embeddings.
    hasher = SRP.SRP(1280)
    self.corpus.slots['SRP_hasher'] = hasher
    return hasher

  def process_batch(self, counts: pa.Array) -> pa.Array:
    # Count the words, and cast to uint32.
    words, counts = counts.flatten()
    try:
      hashed = self.hasher.stable_transform(words.to_pylist(), counts.to_numpy())
      # bit_rep = np.packbits(hashed > 0).tobytes()

    except SRP.EmptyTextError:
      hashed = np.full(1280, np.sqrt(1280), np.float32)
    return pa.array(hashed)

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
  def __init__(self, bookstack, ngrams: int, *args, **kwargs):
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
#    self.end_chars = end_chars
#    self.beginning_chars = beginning_chars  
    super().__init__(bookstack, *args, **kwargs)
    self._upstream = self.bookstack.get_transform("tokenization")
    
  def _from_local(self):
    pass

  @property
  def base_type(self):
    return pa.struct([
      *[pa.field(f"token{i + 1}", pa.string()) for i in range(self.ngrams)],
      pa.field("count", pa.uint32())])

  def process_batch(self, tokens: pa.Array) -> pa.Array:
    ngrams = self.ngrams
    # Could also do the ids here as part of the grouping.

    # Converting to pylist because of bug I can't track down in polars.
    frame = pl.DataFrame({'token': tokens.to_pylist()})
    lazy = frame.lazy()
    for i in range(ngrams):
        lazy = lazy.with_column(pl.col("token").shift(-i).alias(f"word{i+1}"))
    lazy = lazy.filter(pl.col(f"word{ngrams}").is_not_null())
    lazy = lazy.groupby([f"word{i+1}" for i in range(ngrams)])\
       .agg([pl.count("word1").alias("count")])
    frame = lazy.collect()

    cols = {}
    for i in range(ngrams):
      ngram = f"word{i + 1}"
      cols[ngram] = frame[ngram].to_arrow().cast(pa.string())
    cols['count'] = frame['count'].to_arrow().cast(pa.uint32())
    instructed = pa.StructArray.from_arrays(
      [*cols.values()], [*cols.keys()])
    return instructed

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
      "nc:id": pa.uint32()
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
    derived = {}

    # The first flatten combines chunks, and the second
    # disentangles the struct columns ['word1', 'count'].
    array = batch[0].flatten().flatten()
    step = len(array[0])

    # Array them end to end so the lookup can happen in a single pass.
    words = pa.chunked_array(array[:-1])
    encoded = pc.index_in(words, value_set = self.wordids).cast(pa.uint32())


    if self.ngrams == 1: # Just a different name.
      derived = {'wordid': encoded.chunk(0)}
    else: 
      derived = {f"word{n + 1}": encoded[(n*step):(n+1)*step].combine_chunks()
        for n in range(self.ngrams)
      }
    """
    for ngrams, tokens in enumerate(array[:-1]):
        #lookup word ids.
        k = f'word{ngrams + 1}'
        if self.ngrams == 1:
          k = 'wordid'
        derived[k] = pc.index_in(tokens, value_set = self.wordids).cast(pa.uint32())
      # The counts comes last.
    """
    derived['count'] = array[-1].cast(pa.uint32())

    # Stupid--for back-compatibility, to allow one-grams to have a different name in their table.
    return pa.record_batch([derived[k] for k in derived.keys()], names = [*derived.keys()])

  def iter_docs(self):
    yield from self

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    """
    Rather than work a batch at a time, try to do at least 10MB at once,
    so the join can be a bit more efficient.
    """
    offset = 0
    ids = self.bookstack.ids['nc:id']
    for batch in self._upstream:
      encoded = self.process_batch(batch)
      indices = batch[0].value_parent_indices()
      indices = pc.add(indices, offset)
      offset += len(batch)
      id_col = pc.take(ids, indices).combine_chunks()

      cols = [id_col, *encoded]
      names = ["nc:id", *[f.name for f in encoded.schema]]
      yield pa.record_batch(cols, names)
    self.close()

class EncodedUnigrams(EncodedCounts):
  name : str = "encoded_unigrams"
  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack.get_transform("token_counts"), bookstack, *args, **kwargs)

class EncodedBigrams(EncodedCounts):
  name : str = "encoded_bigrams"
  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack.get_transform("bigrams"), bookstack, *args, **kwargs)

class EncodedTrigrams(EncodedCounts):
  name : str = "encoded_trigrams"
  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack.get_transform("trigrams"), bookstack, *args, **kwargs)

transformations = {
  'text': Text,
  'document_lengths': DocumentLengths,
  'token_counts': TokenCounts,
  'tokenization': Tokenization,
  'quintgrams': Quintgrams,
  'bigrams': Bigrams,
  'trigrams': Trigrams,
  'quadgrams': Quadgrams,
  'encoded_unigrams': EncodedUnigrams,
  'encoded_bigrams': EncodedBigrams,
  'srp': SRP_Transform,
  'encoded_trigrams': EncodedTrigrams
}

