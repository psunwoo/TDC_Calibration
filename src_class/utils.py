"""
Functions:
 - predict_tapped
"""
from bisect import insort
import numpy as np


def predict_tapped(perceived_sequence, true_sequence, offset = 1):
  """
  - Input for this func: perceived sequence + true sequence of taps

  - Output for this func: tapped bins based on the two input sequences
  """
  def continuous_check(index_list):
    if index_list[0] != 0:
      return

    for i in range(1, len(index_list)):
      if index_list[i] != index_list[i-1] + 1:
        return index_list[i-1] + offset
    return index_list[-1] + offset if len(index_list) else None

  sequence_index = []
  tapped = set()
  index_dict = {number: index for index, number in enumerate(perceived_sequence)} # Better to compute the index dictionary all at once; index function is costly

  for number in true_sequence:
    insort(sequence_index, index_dict[number]) # Comparing the value: O(log k), inserting: O(k); full sorting would be O(k log k), since it essentially does what insort does for every element, and when updating the sorting repeatedly is desired, do insort
    index = continuous_check(sequence_index)
    if index:
      tapped.add(index)
  return tuple(sorted(tapped))

def trace_physical_num(physical_nums: np.ndarray, logical_assignment: dict[int,int]) -> np.ndarray:
  """Maps logical bin numbers to POR corrected sequences using the given logical assignment.
  'split_into_cells' will then recover the Carry8 structures."""
  return np.vectorize(logical_assignment.get)(physical_nums)


def split_into_cells(physical_nums: np.ndarray, converted = False, logical_assignment = None, TDL_start = 0) -> tuple[dict[int:np.ndarray], int, int]:
  """ This functions splits tapped data for each CARRY cell"""

  if not converted:
    if logical_assignment is None:
      raise ValueError(f"'logical_assignment' can be None if converted == {converted}")

    physical_nums = trace_physical_num(physical_nums, logical_assignment)

  # Define 'custom_mod' function to have the bin position in a single cell to be 1 to 8, and b compatible with either a number or a npdarray
  def custom_mod(values, divisor = 8):
    result = (values + 1) % divisor + divisor * ((values + 1) % divisor == 0)
    return result

  cell_number = 1 + np.asarray(physical_nums) // 8 - TDL_start//8
  cell_index = custom_mod(np.asarray(physical_nums))
  start_num = min(cell_number)
  end_num = max(cell_number)

  results = {num: sorted(cell_index[cell_number == num]) for num in np.unique(cell_number)}

  # Warn for empty cells
  missing_cells = list(set(np.arange(start_num, end_num + 1,3)) - set(cell_number))
  if missing_cells:
    raise ValueError(f"There are missing cells???: {missing_cells}")

  return results, start_num, end_num




def get_time_bins(bins: np.ndarray, freqs: np.ndarray, logical_to_physical_dict: dict[int,int], clock_period = 4000, verbose = True) -> dict:
  """
  This function outputs the result of the Code Density Test (CDC)
  clock_period is in ps
  """

  freqs = clock_period * freqs / np.sum(freqs)    # This converts the frequencies into time bin widths
  cdc_result = {}
  cumulated_time = 0    # Cumulated time over the time bins

  for b,f in zip(bins[:-1], freqs):
    b = logical_to_physical_dict.get(b)
    cumulated_time += f
    cdc_result[b] = cumulated_time

  if verbose == True:
    print(f"Output dictionary constructed until the cumulated time: {cumulated_time} ps \n")

  return cdc_result