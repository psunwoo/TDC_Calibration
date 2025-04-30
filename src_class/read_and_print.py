"""
Functions:
  - load_csv
  - identify_sets
  - print_taps
  - print_Carry_positions
  - vivado_print_bin_assignment
  
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# To read the csv file and return a numpy array
def load_csv(file_path):
  file = pd.read_csv(file_path, header = 0)
  data = file.to_numpy(dtype = int).ravel()
  return data, file_path # Bear in mind: these tap numbers will differ with the full CARRY8 numbers



# To get tapped & missing bins
def identify_sets(data):
  present_taps = np.unique(data)
  present_taps = np.delete(present_taps, present_taps == 0)  # Does not take the weird 0 bin into account
  full_range = np.arange(np.min(present_taps), np.max(present_taps)+1,1)
  missing_taps = np.setdiff1d(full_range, present_taps)
  return full_range, present_taps, missing_taps
  """ Note the DATA Types - all unique entries only
  full_range: np.array
  present_taps: np array
  missing_taps: np array
  """



# To draw the tapped & missing bins
def print_taps(data, present_taps, missing_taps):
  bins = np.arange(min(data), max(data)+2, 1)

  plt.figure(figsize = (11,6))
  plt.subplot(1,3,1)
  freq, bin, _ = plt.hist(data, bins = bins, color = 'blue', edgecolor = 'black')
  plt.title("Tapped Bins")
  plt.xlabel("Tap number")
  plt.ylabel("Cumulative Frequency")

  plt.subplot(1,3,2)
  plt.hist(missing_taps, bins = bins, color = 'red', edgecolor = 'black')
  plt.title("Untapped Bins (Normalized)")
  plt.xlabel("Tap number")

  plt.subplot(1,3,3)
  differences = np.diff(present_taps)
  bins = np.arange(np.min(differences), np.max(differences) +2 , 1)
  plt.hist(differences, bins = bins, color = 'blue', edgecolor = 'black')
  plt.title("Distance Between Tapped Bins")
  plt.show()

  return bin, freq



def print_Carry_positions(Carry_data):
  """The expected input is a dictionary (cell number: tapped indices)."""
  data = []
  for entry in Carry_data.values():
    data.extend(entry)
  bins = np.arange(1,9,1)
  plt.figure(figsize = (10,6))
  plt.hist(data, bins = bins, color = 'blue', edgecolor = 'black')
  plt.title("Carry8 Positions Histogram")
  plt.xlabel("Carry8 Positions")
  plt.ylabel("Frequency")
  plt.show()



def vivado_print_bin_assignment(bin_assignment, list_fill_in = True, logical_start = None):
  lines = []
  
  # for i in range(0,logical_start + 1):
  #   lines.append(
  #     f"assign o_taps[{i}] = "
  #     f"i_taps[{i}]; \n"
  #   )

  for logical, physical in bin_assignment.items():
    lines.append(
      f"assign o_taps[{logical}] = "
      f"i_taps[{physical}]; \n"
    )

  if list_fill_in == True:
    for number in range(max(bin_assignment.keys())+1, 12800):
      lines.append(
        f"assign o_taps[{number}] = "
        f"0; \n"
      )
  
  ID = input("Type the ID for this txt file (e.g. '25_02_14'): ")
  filename = f"Bin_sequence_{ID}.txt"

  with open(filename, "w") as file:
    file.writelines(lines)

  print(f"The text file has been created in this directory under the name: {filename}.")
  return



# # To get the distance between two consecutive tapped bins
# def print_tapped_difference(present_taps):
#     differences = np.diff(present_taps) /2
#     bins = np.arange(np.min(differences), np.max(differences) +2 , 1)
#     plt.hist(differences, bins = bins, color = 'blue', edgecolor = 'black')
#     plt.title("Distance Between Tapped Bins")
#     plt.show()


