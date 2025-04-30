"""
This is the class for CARRY8 cells
"""
import numpy as np
from collections import defaultdict, deque
from bisect import insort

from numpy._typing import _TD64Like_co

class Cell:
  
  def __init__(self, tdl, cell_num):
    self.tdl = tdl
    self.cell_num = cell_num
    self.trial = 0
    self.history = []
    self.deleted_history = []
    self.tapped_bins = ()
    self.disabled = False
    self.POR_carry = None


  def update_pattern(self, tapped_bins, verbose=False) -> None:
    """
    Updates tapped_bins for the current trial.
    Stores the tapped_bins history directly (no IDs).
    """
  
    self.tapped_bins = tuple(tapped_bins)
    self.history.append(self.tapped_bins)
    self.trial += 1

    if verbose:
        print(f"Cell {self.cell_num} updated: {self.tapped_bins}")

  def get_tapped_history(self) -> tuple[tuple]:
    """Returns the full tapped bin history for this cell."""
    return tuple(self.history)

  def reset(self, POR_carry: list) -> None:
    self.trial = 0
    self.deleted_history.extend(self.history)
    self.history = []
    self.tapped_bins = ()
    self.POR_carry = POR_carry
