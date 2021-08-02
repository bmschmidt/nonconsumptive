import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive.inputs import FolderInput, MetadataInput, SingleFileInput
import pyarrow as pa

@pytest.fixture(scope="module")
def three_files(tmpdir_factory):
    return FolderInput(Path("tests/corpora/test1/texts"))
    
class TestFolderInput:
  def test_iter(self, three_files):
    i = 0
    for t in three_files:
      i += 1
    assert i == 3
  def test_get(self, three_files):
    pride = three_files['a']
    assert "of a wife" in pride
    
  def test_ids(self, three_files):
    ids = [*three_files.ids()]
    assert len(set(ids)) == 3