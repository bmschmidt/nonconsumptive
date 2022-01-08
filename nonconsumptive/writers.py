from pathlib import Path
from nonconsumptive import Corpus
from pyarrow import parquet, Table

class Writer:
  def __init__(self, corpus, destination: Path, features = ["text"]):
    self.corpus = corpus
    self.destination = destination
    self.validate_path()
    self.features = features

  @property
  def stacks(self):
    for stack in self.corpus.bookstacks:
      group = stack.metadata
      for feature in self.features:
        f = stack.get_transform(feature).table
        for name in f.schema.names:
          group = group.append_column("nc:" + name, f[name])
      yield stack.uuid, group
      
  def validate_path(self):
    assert self.destination.parent.exists(), "Destination directory does not exist"
  def write(self):
    raise NotImplementedError("Each writer needs a custom write function")

class ParquetWriter(Writer):
  def write(self):
    for stackname, group in self.stacks:
      parquet.write_table(group, self.destination / f"{stackname}.parquet")