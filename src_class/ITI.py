"""
This is a py file for Iterative Time-bin Interleaf(ITI):
  - get_bin_widths
  -
"""
import numpy as np
from typing import Union


def get_bin_widths(bins: np.ndarray, freqs: np.ndarray, logical_to_physical_dict: dict[int,int], clock_period = 4000, verbose = True) -> dict:
  """
  This function outputs the result of the Code Density Test (CDC) (called cdc_width)
  clock_period is in ps
  """

  freqs = clock_period * freqs / np.sum(freqs)    # This converts the frequencies into time bin widths
  cdc_width = {}    # This one is for visualization / display purpose
  cdc_to_pass = {}  # This one will be passed on and be used to calculate the starting time of each bin

  # cumulated_time = 0    # Cumulated time over the time bins

  for i,(b,f) in enumerate(zip(bins[:-1], freqs)):
    # The starting bin is just the starting point to be ignored? 

    b = logical_to_physical_dict.get(b)
    cdc_width[b] = f

    if i == 0:
      carry = f
      continue

    cdc_to_pass[b] = carry
    carry = f

  return cdc_width, cdc_to_pass


def get_bin_times(cdc_to_pass: Union[dict[int, float], tuple[list[int], np.ndarray]],
                  list_mode: bool = False,
                  threshold: float = 0.5,
                  offset: float = 0.0) -> dict[int, float]:
  """
  This function converts the cdc_result into a dictionary that has bins: their starting times
  """

  cumulated_time = offset
  cdc_time = {}

  if not list_mode:

    for b,f in cdc_to_pass.items():
      cumulated_time += f

      if f < threshold:
        continue

      cdc_time[b] = cumulated_time

  else:

    bins, width = cdc_to_pass

    for b,f in zip(bins, width):
      cumulated_time += f

      if f < threshold:
        continue
      
      cdc_time[b] = cumulated_time

  return cdc_time




def perform_ITI(cdc_to_pass_list: list[dict[int, float]],
                offset_lists: Union[None, list[float]] = None,
                threshold: float = 0.1) -> dict[int, float]:
  """
  Perform ITI across multiple TDL segments with optional time offset and narrow bin suppression.
  """
  if offset_lists is None:
    offset_lists = np.zeros(len(cdc_to_pass_list))
  elif len(offset_lists) != len(cdc_to_pass_list):
    raise ValueError(f"'offset_lists' should match length of 'cdc_to_pass_list' ({len(cdc_to_pass_list)})")

  # Step 1: Merge all bins into a single time-aligned dictionary
  before_ITI = {}
  for cdc_to_pass, offset in zip(cdc_to_pass_list, offset_lists):
    result = get_bin_times(cdc_to_pass, list_mode=False, offset=offset)
    before_ITI.update(result)

  # Step 2: Sort bins based on cumulative time
  sorted_items = sorted(before_ITI.items(), key=lambda x: x[1])
  bins, times = zip(*sorted_items)  # Separate into bins and times
  bins = list(bins)
  times = np.asarray(times)

  # Step 3: Calculate time bin widths and re-filter
  bin_widths = np.diff(np.insert(times, 0, 0))
  ITI_result = get_bin_times((bins, bin_widths), list_mode=True, threshold=threshold)

  return ITI_result, bin_widths




def get_logical_assignments_from_ITI(ITI_result: dict[int, float], global_offset: list[int], verbose = True) -> dict[int,int]:
  """
  This prints out the logical assignment dictionary from the ITI result
  """


  logical_assignment = {}

  counter = 0
  for i in global_offset:
    logical_assignment[counter] = i
    counter += 1

  for i,j in enumerate(ITI_result, start = counter):
    logical_assignment[i] = j
    

  if verbose:
    print(f"Hand this in to 'vivado_print_bin_assignment' in 'read_and_print' with 'list_fill_in' = False ... \n")

  return logical_assignment



def set_global_offset(minimum:int, length:int, skips:int = 0,
                      divisor: int = 8, TDL_segment_size: int = 3200, 
                      slice_size:int = 3) -> list[int]:

  # For safety
  if not isinstance(skips, int)  or skips < 0:
    raise ValueError(f"The 'skips' should be a non-negative integer. ")

  a = ((minimum - TDL_segment_size * (minimum // TDL_segment_size)) // divisor) // slice_size 
  r = minimum % divisor

  if length + skips > a * 8 + r:
    raise ValueError(f"The length value + skips value {length} is too big...")

  
  # List Construction
  quotient_num = minimum // divisor
  remainder_num = minimum % divisor

  offset = []
  while len(offset) < length:
    remainder_num -= 1
    if remainder_num == -1:
      remainder_num = 7
      quotient_num -= 3

    if skips != 0:
      skips -= 1
      continue

    offset.append(quotient_num * 8 + remainder_num)


  return offset[::-1]






