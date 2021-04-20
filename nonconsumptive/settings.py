import yaml
from pathlib import Path
from typing import Callable, Iterator, Union, Optional, List

settings: dict = {}

def load_settings(dir: Union[str, Path]) -> dict:
  """
  Load a yaml settings file for a project.

  Returns settings for reference, but usually you're better off
  importing the object.

  """
  global settings
  p = Path(dir)
  for path in [p, *p.parents]:
    if path.exists():
      settings = yaml.load((path / ".nonconsumptive.yaml").open())
      return settings
  return settings

