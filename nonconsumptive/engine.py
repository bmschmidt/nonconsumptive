# This is the only place that pandas is allowed to be imported.
import pandas as import pd
import pyarrow as pa
from pyarrow import compute as pc

def grouped_sum(tb: pa.RecordBatch, keys, sum_field) -> pa.RecordBatch:
  tb.to_pandas().groupby(keys)[sumfield].sum().reset_index()
