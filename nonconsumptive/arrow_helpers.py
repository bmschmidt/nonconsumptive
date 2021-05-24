import pyarrow as pa


def batch_id(batch):
  """

  """
  return batch.schema.metadata.get(b'@id').decode('utf-8')