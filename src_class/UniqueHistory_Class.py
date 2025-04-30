
from collections import defaultdict, deque
from bisect import insort

class UniqueHistory:
  def __init__(self, tdl):
    self.tdl = tdl
    self.entries = []  # Each entry is a list of unique patterns for that trial
    self.trial = 0
 

  def update_from_cells(self):
    """
    After all Cells are updated, collect their histories and store unique paths.
    Each entry is a list of unique history tuples (tuples of tapped bins per trial).
    In trial 0, classify into Start/Batch/EndCellHistory for POR. Later trials simply extend paths.
    """
    unique_full_paths = []
    seen = set()

    for cell_num, cell in self.tdl.cells.items():
      # Do not update from the diabled Cells
      if cell.disabled:
        continue

      full_path = cell.get_tapped_history()
      if full_path not in seen:
        seen.add(full_path)

        if cell.trial == 1:
          # First trial – store as CellHistory subclasses
          if cell_num == self.tdl.start_num:
            unique_full_paths.append(StartCellHistory(full_path))
          # elif cell_num == self.tdl.end_num:
          #   unique_full_paths.append(EndCellHistory(full_path))
          else:
            unique_full_paths.append(BatchCellHistory(full_path))
        else:
          # Later trials – just store the full paths as tuples
          unique_full_paths.append(full_path)

    self.entries.append(unique_full_paths)
    self.trial += 1

    
  def perform_POR(self, trial = None, ansatz = [2,1,3,8,4,6,5,7]):
    """
    Perform POR for each unique pattern in the specified trial.
    Only valid for trial 0 when entries are CellHistory instances
    """
    if trial is None:
      trial = self.trials - 1

    if trial != 0:
      raise ValueError("POR is only valid for the initial trial (trial 0)")

    POR_results = {}
    DAG_results = {}
    for entry in self.entries[0]:
      result, DAG, _, _ = entry.perform_POR(ansatz)
      POR_results[entry.compare_tapped] = result
      DAG_results[entry.compare_tapped] = DAG

    return POR_results, DAG_results


class BaseCellHistory:

  def __init__(self, tapped_bins):
      self.compare_tapped = tapped_bins
      self.tapped_bins = tuple(tapped_bins[0])

  def get_DAG(self):
    """ This function is defined individually for each of the SubClasses"""
    raise NotImplementedError

  def perform_POR(self, ansatz):
    """ Partial Order Reconstruction (POR) for a single Carry8 cell"""

    DAG, in_degree, zero_degrees = self.get_DAG()

    if DAG is None:
      return None # Early exit for empty cells

    POR_result_single = []

    # Store things to return
    in_degree_to_return = in_degree.copy()
    zero_degrees_to_return = sorted(zero_degrees.copy(), key = lambda x: ansatz.index(x) if x in ansatz else float('inf'))

    # Partial Order Reconstruction (2) - topo sort with the given ansatz
    while zero_degrees:
      zero_degrees = deque(sorted(zero_degrees, key = lambda x: ansatz.index(x) if x in ansatz else float('inf')))
      node = zero_degrees.popleft()
      POR_result_single.append(node)
      for neighbor in DAG[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
          zero_degrees.append(neighbor)

    return POR_result_single, DAG, in_degree_to_return, zero_degrees_to_return


class StartCellHistory(BaseCellHistory):
  """Represents the first tapped CARRY8 cell with special logic."""

  def get_DAG(self):
    # Early exit for empty cells
    if not self.tapped_bins:
        return None, None, None

    DAG = defaultdict(list)
    bridge = min(self.tapped_bins) # Note that the definition of the bridge is abused here.

    # Deduce sequence from the pruned Start Cell
    if bridge != 8:
      DAG[bridge].append(bridge + 1)
    bridge += 1 # bridge needs to be updated

    # Deduce the rest of the DAG
    for value in range(bridge + 1, 9):
      if value - 1 in self.tapped_bins:
          DAG[bridge].append(value)
          bridge = value
      else:
          DAG[value].append(bridge)

    # Construct 'in_degree' and 'zero_degrees'
    in_degree = {node: 0 for node in range(min(self.tapped_bins), 9)}
    for edge in DAG.values():
      in_degree[edge[0]] += 1 # 'edge[0] is due to the data type: defaultdict(list)
    zero_degrees = [node for node, degree in in_degree.items() if degree == 0]

    return DAG, in_degree, zero_degrees


class EndCellHistory(BaseCellHistory):
  """Represents the last CARRY8 cell with special logic."""

  def get_DAG(self):
    # Early exit for empty cells
    if not self.tapped_bins:
      return None, None, None

    DAG = defaultdict(list)
    bridge = min(self.tapped_bins) # Note that the definition of the bridge is abused here
    end = max(self.tapped_bins)

    # Deduced sequence from the first tapped bin in the carry cell.
    for number in range(2, bridge + 1):
      DAG[number].append(1)
    if bridge != 8 and len(self.tapped_bins) != 1:
      DAG[1].append(bridge + 1)
    bridge += 1 # bridge needs to be updated

    # Deduce the rest of the DAG
    for value in range(bridge + 1, end + 1):
      if value - 1 in self.tapped_bins:
        DAG[bridge].append(value)
        bridge = value
      else:
        DAG[value].append(bridge)

    # Construct 'in_degree' and 'zero_degrees'
    in_degree = {node: 0 for node in range(1, end + 1)}
    for edge in DAG.values():
      in_degree[edge[0]] += 1
    zero_degrees = [node for node, degree in in_degree.items() if degree == 0]

    return DAG, in_degree, zero_degrees


class BatchCellHistory(BaseCellHistory):
  """Represents intermediate CARRY8 cells. This defines the default behavior."""

  def get_DAG(self):
    # Early exit for empty cells
    if not self.tapped_bins:
      return None, None, None

    DAG = defaultdict(list)
    bridge = min(self.tapped_bins) # Note that the definition of the bridge is abused here

    # Deduced sequence from the first tapped bin in the carry cell.
    for number in range(2, bridge + 1):
      DAG[number].append(1)
    if bridge != 8:
      DAG[1].append(bridge + 1)
    bridge += 1 # bridge needs to be updated

    # Deduce the rest of the DAG
    for value in range(bridge + 1, 9):
      if value - 1 in self.tapped_bins:
        DAG[bridge].append(value)
        bridge = value
      else:
        DAG[value].append(bridge)

    # Construct 'in_degree' and 'zero_degrees'
    in_degree = {node: 0 for node in range(1, 9)}
    for edge in DAG.values():
      in_degree[edge[0]] += 1
    zero_degrees = [node for node, degree in in_degree.items() if degree == 0]

    return DAG, in_degree, zero_degrees
