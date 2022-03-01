import pyarrow as pa
import pyarrow.types as pt
import yaml
from pathlib import Path
from typing import Union

class NC_Schema:
  def __init__(self, path):
    self.path = Path(path)
    self.types = {}
    self.load()

  def pa_schema(self) -> pa.Schema:
    dicto = self.types
    output = {}
    for key, d in dicto.items():
      dtype = getattr(pa, d['arrow_type'])()
      if type(d['list']):
        output[key] = pa.list_(dtype)
      else:
        output[key] = dtype
    return pa.schema(output)

  def load(self) -> None:
    if not self.path.exists():
      self.types = {}
      return None
    with open(self.path, "r") as f:
      self.types = yaml.load(f)


def field_as_string(field : pa.DataType) -> str:
    """/
    represent a single field as a dictionarry representation.
    """
    if pt.is_string(field) or pt.is_large_string(field):
        return "string"
    if pt.is_int64(field):
        return "int64"
    if pt.is_int32(field):
        return "int32"
    raise NotImplementedError(f"Can't determine JSON code for {field}")
    return field

def field_as_dict(field: pa.DataType) -> dict:
    listlike = False
    dictionarylike = False
    if pt.is_large_list(field) or pt.is_list(field):
        listlike = True
        field = field.value_field.type
    if pt.is_dictionary(field):
        dictionarylike = True
        raise NotImplementedError("dictionaries not yet supported")
        field = field.value_field.type        
    dtype = field_as_string(field)
    return {
        "list": listlike,
        "arrow_type": dtype,
        "role": None,
        "dictionary": dictionarylike
    }
    
