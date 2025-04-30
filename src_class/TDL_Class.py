"""
This is the class for TDL (made of logical bins)
"""

# from cell import Cell  # Import Cell class, if not in Google Colab.
from collections import defaultdict, deque
import numpy as np
import os
import sys
import importlib
import inspect

# To import from other py files
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.append(current_dir)

import Cell_Class
import UniqueHistory_Class
import utils
import initialize

importlib.reload(Cell_Class)
importlib.reload(UniqueHistory_Class)
importlib.reload(utils)
importlib.reload(initialize)


class TDL:
  """
  Brief groupings of the functions:
  1) Feeding the TDL instance with raw data 
    - creates & updates Cell instance + UniqueHistory instance

  2) Utils
    - reset_TDL()
    - undo_last_trial

  3) Carrying out the 1st POR (Partial Order Reconstruction) 
    - Calls perform_POR() function for each unique pattern in UniqueHistory.entries[0] elements
    - These unique patterns are, in turn, defined to be a BaseCellHistory Class
    - These unique patterns go through DAG extraction etc to build a dictionary
    - Each cell will then find their POR-corrected sequences by looking up in the dictionary
    - Some proof measures in case the offset in the TDL changes, unexpected cells appear/disappear

  4) Constructing the error library of the POR
    - Goes through every possibility consistent with the POR
    - Returns a dictionary of each unique pattern to another dictionary of:
        'If this another possibility is true, what will the data look like in the next trial, after this POR guess?'
        (next_data_prediction): []

  5) Carrying out the next POR & constructing the error library iteratively:
    - Finds a group of 
  """

  def __init__(self):
    self.slice_index = 0
    self.cells = {} # Dict of CARRY Cell class instances {cell_num: Cell instance}

    self.physical_num = []
    self.logical_assignment = {}
    self.POR_result = []

    self.start_num = float('inf')
    self.end_num = 0
    self.logical_start = float('inf')

    self.trial = 0
    self.unique_history = UniqueHistory_Class.UniqueHistory(self) 

    self.TDL_start = 0

  def initialize_TDL(self, physical_start, physical_end, slice_index, verbose = True):
    self.slice_index = slice_index
    self.TDL_start = physical_start
    self.logical_assignment, _ = initialize.print_sliced(physical_start,physical_end,slice_index, verbose = verbose)


  ##############################################################################################################################
  """ 1. For feeding the data"""
  ##############################################################################################################################


  def trace_physical_num(self, physical_nums: np.ndarray, logical_assignment: dict[int,int]) -> np.ndarray:
    """Maps logical bin numbers to POR corrected sequences using the given logical assignment.
    'split_into_cells' will then recover the Carry8 structures."""
    return np.vectorize(logical_assignment.get)(physical_nums)


  def split_into_cells(self, physical_nums) :
    """ This functions splits tapped data for each CARRY cell"""

    # Define 'custom_mod' function to have the bin position in a single cell to be 1 to 8, and b compatible with either a number or a npdarray
    def custom_mod(values, divisor = 8):
      result = (values + 1) % divisor + divisor * ((values + 1) % divisor == 0)
      return result.astype(int) if isinstance(values, np.ndarray) else int(result)

    cell_number = 1 + np.asarray(physical_nums) // 8 - self.TDL_start // 8
    cell_index = custom_mod(physical_nums)
    start_num = min(cell_number)
    end_num = max(cell_number)

    results = {num: sorted(cell_index[cell_number == num]) for num in np.unique(cell_number)}

    # Warn for empty cells
    missing_cells = list(set(np.arange(start_num, end_num + 1,3)) - set(cell_number))
    if missing_cells:
      raise ValueError(f"There are missing cells???: {missing_cells}")

    return results, start_num, end_num


  # def find_cell_nums(self, physical_nums):
  #   """ Returns the cell numbers present in the data """
  #   cell_number = 1 + np.asarray(physical_nums) // 8
  #   return np.unique(cell_number).tolist()


  def feed_physical_nums(self, physical_nums, trial_num, logical_assignment = None, converted = False, verbose = True) -> None:
    """ 
    Convert the raw data into physical bin numbers, if not already converted.
    If this is the first trial, record the logical_start, start_num, end_num, and initialize Cell instances.
    Update the UniqueHistory instance of this TDL. 
    Handles the case of change in the tapped cells 
    """
    trial_num -= 1 # To match with the trial number tracking with other functions ...

    if self.trial != trial_num:
      print(f"Trial number = {self.trial} needs to be fed. The current feeding trial is ignored. \n")
      return

    # 'start_num' and 'end_num' check
    # Convert to physical nums, if not already
    if not converted:
      if logical_assignment is None:
        logical_assignment = self.logical_assignment
      physical_nums = self.trace_physical_num(physical_nums, logical_assignment)

    results, start_num, end_num = self.split_into_cells(physical_nums)
    logical_start = min(physical_nums)

    if trial_num != 0:
      # Find current trial's cell numbers
      current_cells = set(results.keys())

      known_cells = set(self.cells.keys())

      # Find disabled cells (not tapped this time)
      absent_cells = known_cells - current_cells

      # Find newly tapped cells (not seen before)
      new_cells = current_cells - known_cells

      for cell_num in new_cells:
        self.cells[cell_num] = Cell_Class.Cell(self, cell_num)
        
      # Disable absent_cells
      for cell_num in absent_cells:
        self.cells[cell_num].disabled = True

    # Need to recreate the Start cell, if the logical_start shifts
      if logical_start != self.logical_start:  
        self.cells[self.start_num].disabled = True

      # Set a flag or notify the user
      if absent_cells or new_cells:
        if verbose:
          print(f"⚠️ Trial {self.trial} anomaly:")
          if absent_cells:
              print(f"  - Absent cells: {sorted(absent_cells)}")
          if new_cells:
              print(f"  - New cells: {sorted(new_cells)}")


    # Update Cells on first trial
    else:
      self.logical_start = logical_start
      for cell_num in results.keys():
        self.cells[cell_num] = Cell_Class.Cell(self, cell_num)

    # Update TDL variables
    self.logical_start = logical_start
    self.start_num = start_num
    self.end_num = end_num

    for cell_num, taps in results.items():
      self.cells[cell_num].update_pattern(taps, verbose = False)

    self.unique_history.update_from_cells()
    self.physical_num = physical_nums

    if verbose:
      print(f"Trial {self.trial} data processed and added to unique history. \n")
    self.trial += 1

  ##############################################################################################################################
  """ 2. Utils """
  ##############################################################################################################################

  def reset_TDL(self, physical_start, physical_end, slice_index, verbose = False) -> None:
    self.slice_index = 0
    self.cells = {} # Dict of CARRY Cell class instances {cell_num: Cell instance}

    self.physical_num = []
    self.POR_result = []

    self.start_num = float('inf')
    self.end_num = 0
    self.logical_start = float('inf')

    self.initialize_TDL(physical_start, physical_end, slice_index, verbose = False)

    self.trial = 0
    self.unique_history = UniqueHistory_Class.UniqueHistory(self) 
    if verbose == True:
      print(f"All is reset.")


  def undo_last_trial(self, verbose = False) -> None:
    """Undo the latest trial and revert all relevant information"""
    if self.trial == 0:
      return print(f"Nothing to undo. Trial: {self.trial}")

    self.trial -= 1

    # Remove the last element in cell.history for each cell and roll-back the tapped_bins accordingly
    for cell in self.cells.values():
      if self.trial < len(cell.history):
        cell.history.pop()
        if cell.history:
          cell.tapped_bins = cell.history[-1]
        else:  
          cell.tapped_bins = ()

    # Remove the last entry from UniqueHistory
    if self.unique_history.entries:
      self.unique_history.entries.pop()
      self.unique_history.trials -= 1

    if verbose == True:
      print(f"Undo done. Trial number now: {self.trial}")

  ##############################################################################################################################
  """ 3. Performing POR (Partial Order Reconstruction) by calling it for each of unique patterns """
  ##############################################################################################################################

  def perform_POR(self, ansatz = [2,1,3,8,4,6,5,7]) -> None:
    if self.trial < 1:
      return "Feed the physical nums first"
    elif self.trial > 1:
      return (f"Call 'next_POR_n_error_lib' function. \n"
             f"This function is only for trial num = 0 (current trial num: {self.trial-1})")

    if self.POR_result == []:
      self.POR_result, _ =  self.unique_history.perform_POR(trial = 0, ansatz = ansatz)


  def get_logical_assignments(self, offset_space = True):
    """
    Call this function only after perform_POR
    Outputs 2 dictionaries: 
      1. logical bin assignment: physical bin number
      2. logical bin assignment: 
    """
    logical_assignment = self.logical_assignment.copy()
    
    print(f"\n--------------------------------------------------------------------------------------------------------\n"
          f"🟡 Running '{inspect.currentframe().f_code.co_name}' in '{__name__}' module ...\n")

    # re = (self.logical_start + 1) % 8 + 8 * ((self.logical_start + 1) % 8 == 0) # Carry 8 index of the logical_start

    for cell_num, cell in self.cells.items():
      # Try getting the POR result from the self.POR_result dictionary
      try:
        POR_result = self.POR_result[cell.get_tapped_history()] # This is the problem; need [0] for the cell.trial = 1; no for other than 1
      except KeyError:
        cell.disabled = True

      # Manually type in the sequence, for disabled cells
      if cell.disabled:
        print("\n")
        raw = input(
            f"🛑 Cell {cell_num} is disabled. Please manually enter POR result as digits (e.g. 21347658):\n"
            f"History: {cell.history}\n"
            f"Type anything invalid to omit this cell\n→ "
        ).strip()

        if not raw.isdigit() or (len(raw) != 8 and self.start_num < cell_num < self.end_num):
          print("\n ⏭️ This cell is skipped.\n")
          continue

        POR_result = list(map(int, raw))

      # Special handling for start_cell
      if cell_num == self.start_num:
        POR_result = np.array(POR_result)
        re = 9 - len(POR_result)  # Carry 8 index of the start
        # POR_result = POR_result[POR_result >= re]

        for i, result in enumerate(POR_result, start= re - 1):
          logical_assignment[8 * ((cell_num - 1)//3) + i] = 8 * (cell_num-1) + result - 1 + self.TDL_start

      else:
        for j, result in enumerate(POR_result):
          logical_assignment[8 * ((cell_num - 1)//3) + j] = 8 * (cell_num-1) + result - 1 + self.TDL_start

    print(f"\n✅ '{inspect.currentframe().f_code.co_name}' in '{__name__}' module finished.\n"
          f"--------------------------------------------------------------------------------------------------------\n")
    return logical_assignment


  ##############################################################################################################################
  """ 4. Building error libraries to correct POR guesses (after all, only partial DAG information is given from data) """
  ##############################################################################################################################


  def build_initial_error_lib(self, trial = None):
    """
    This function is to construct the POR error library in the 1st POR
    """
    if trial is None:
      trial = self.trial - 1
    
    if trial != 0:
      raise ValueError(f" {inspect.currentframe().f_code.co_name} is only valid for the initial trial (trial 0)")

    # Initialize result dictionary
    error_lib = {}

    for unique_pattern, POR_result in zip(self.unique_history.entries[0], self.POR_result.values()):
      if isinstance(unique_pattern, UniqueHistory_Class.StartCellHistory):
        result = self.build_POR_error_lib(unique_pattern, POR_result, offset = min(unique_pattern.tapped_bins))
      else:
        result = self.build_POR_error_lib(unique_pattern, POR_result)

      error_lib[(unique_pattern.tapped_bins,)] = result  # Store results for each cell
    return error_lib


        
  def build_POR_error_lib(self, unique_pattern, POR_result = None, PORed = True, trial = None,  offset = 1, ansatz = [2,1,3,8,4,6,5,7]):
    """Goes through every possible option consistent with the given / to-be-calculated DAG to build the error library."""

    # Two different options whether POR_result has to be calculated or not
    if PORed:
      if POR_result is None:
        raise ValueError("POR_result can not be None for PORed = True")

      DAG, in_degree, zero_in_degree = unique_pattern.get_DAG()

    else:
      POR_result, DAG, in_degree, zero_in_degree = unique_pattern.perform_POR(ansatz)

    # backtracking function defined
    def backtrack(path, DAG, in_degree, zero_in_degree, result, tapped_bins_func):
      if len(path) == len(in_degree):  # Base case: all nodes are processed
        tapped_bins = tapped_bins_func(path)
        result[tapped_bins].append(tuple(path))
        return

      for node in list(zero_in_degree):  # Work with a copy to avoid mutating
        zero_in_degree_copy = zero_in_degree[:]
        in_degree_copy = in_degree.copy()

        # Choose the node
        zero_in_degree_copy.remove(node)
        path.append(node)
        for neighbor in DAG[node]:
            in_degree_copy[neighbor] -= 1
            if in_degree_copy[neighbor] == 0:
                zero_in_degree_copy.append(neighbor)

        # Recurse
        backtrack(path, DAG, in_degree_copy, zero_in_degree_copy, result, tapped_bins_func)

        # Backtrack: Undo the changes
        path.pop()
        

    result = defaultdict(list)
    backtrack([], DAG, in_degree, zero_in_degree, result, lambda seq: utils.predict_tapped(POR_result, seq, offset = offset))

    return result if PORed == True else (POR_result, result)
    
  ##############################################################################################################################
  """5. Carrying out the PORs at next trials (looking up the error library dictionary and selecting one iteratively;
  and carrying out the initial POR for new cells)

  and buliding the next error libraries from other possible choices / candidates from the dictionary Look-up. """
  ##############################################################################################################################


  def next_POR_n_error_lib(self, error_lib, for_which_trial):
    # Check for the corect trial version
    if for_which_trial != self.trial:
      print(f"Function called for trial {for_which_trial}. \n"
            f"But the current trial {self.trial-1} is insufficient. \n"
            f"Needs to be at current trial: {for_which_trial - 1}")
      return

    print(f"\n --------------------------------------------------------------------------------------------------------\n"
          f"🟡 Running '{inspect.currentframe().f_code.co_name}' in '{__name__}' module ...\n")
    next_POR_dict = {}
    next_error_lib = {}

    for unique_history in self.unique_history.entries[self.trial - 1]:

      # If the initial POR needs to be carried out
      if isinstance(unique_history, UniqueHistory_Class.BaseCellHistory):
        next_POR, result = self.build_POR_error_lib(unique_history, PORed = False)
        next_POR_dict[unique_history.compare_tapped] = next_POR
        next_error_lib[unique_history.compare_tapped] = result
        continue

      else:
      # Try look-up in the error library

        *previous, latest = unique_history
        value = error_lib[tuple(previous)][latest]

        if value != []:
          next_POR = value[0]

      # If value is an empty list (from using a key that does not exist in a 'defaultdict')
      # In this case, no corresponding error library will exist; change of sequence messes up with the whole calibration
        else:
          raw = input(
              f"🛑 Change of bin sequence detected. Please manually enter POR result as digits (e.g. 21347658):\n"
              f"History: {unique_history}\n"
              f"Type anything shorter than 8 letters to skip; this will have to be repeated for individual cells:\n\n→ "
          ).strip()

          if not raw.isdigit() or len(raw) != 8:
            print("⏭️ This unique_pattern is skipped.\n")
            continue

          next_POR = list(map(int, raw))
          next_POR_dict[unique_history] = next_POR
          continue

      next_POR_dict[unique_history] = next_POR
      offset = min(next_POR)

      # Construct the next error lib with the candidates
      lib = defaultdict(list)
      for candidate in value:
        pattern = utils.predict_tapped(next_POR, candidate, offset = offset)
        lib[pattern].append(candidate)
      next_error_lib[unique_history] = lib

    print(f"\n✅ '{inspect.currentframe().f_code.co_name}' in '{__name__}' module finished.\n"
          f"--------------------------------------------------------------------------------------------------------\n")
    self.POR_result = next_POR_dict
    return next_error_lib
