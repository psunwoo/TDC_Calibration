"""
Functions:
  - slice_carrys
  - print_sliced

"""
import numpy as np


def slice_carrys(array_start, array_end, slice_index, slice_num = 3):
  """
  'split_index' refers to which slice (1st,2nd,3rd) the current CARRY8 cell group is
  """
  num_array = np.arange(array_start, array_end ,1)
  quotients = ((num_array - array_start) // 8) % slice_num 
  return num_array[quotients == slice_index - 1]


def print_sliced(array_start, array_end, slice_index, slice_num = 3, verbose = True):
  """
  Two types of POR results are needed for processing.
  1) bin_assignment: in the format to be directly feedable to vivado.
  i.e. in the format of {logical bin number: physical bin number}
  e.g.
  {0: 0, 1: 1, 2: 2, 3: 3, 4: 4 .. }


  2) POR_result: in the format that shows the deduced bin sequence inside each CARRY8 cell
  e.g.
  [[(4, 8)], [(7, 2), (7, 3), (7, 1), (7, 8), (7, 4), (7, 6), (7, 7), (7, 5)], ...]

  """
  sliced = slice_carrys(array_start, array_end, slice_index, slice_num = 3)
  if verbose == True:
    print(sliced)

  # Calculate the first output
  bin_assignment = {number: sliced[number] for number in range(0, len(sliced))}

  # Calculate the second output
  POR_result = []
  sublist = []
  for number in range(np.min(sliced)//8 + 1, np.max(sliced)//8 + 2):
    sublist = []
    for i in range(1,9):
      sublist.append((number, i))
    POR_result.append(sublist)

  return bin_assignment, POR_result

