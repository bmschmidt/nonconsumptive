import pyarrow as pa

class ConsumptionError(Exception):
  pass

import numpy as np

# First text is sui generis, so call it dist 0.
def distances(all_texts):
    # This could be much, much more efficient by using bit operations or something,
    # probably.
    dists = np.zeros(len(all_texts), np.int32)
    for i in range(1, len(all_texts)):
        assigned = False
        for j_, (a, b) in enumerate(zip(all_texts[i], all_texts[i - 1])):
            if a != b:
                dists[i] = j_
                assigned = True
                break
        if not assigned:
            # If ID1 is distinguished by subsequent characters
            # after ID2 ends, or vice-versa.
            dists[i] = j_ + 1
    return dists

def breaks(array, min_length = 2**8, max_length = 2**14):
    if len(array) <= max_length:
        return [len(array)]
    min = np.argmin(array[min_length:-min_length])
    l = array[:(min + min_length)]
    r = array[(min + min_length):]
    return [*breaks(l, min_length, max_length), *breaks(r, min_length, max_length)]

def chunk_ids(ids, min_length = 2**8, max_length = 2**14):
    """
    Breaks list of ids into vaguely reasonable stacks by splitting wherever ids are very
    different from each other.

    ids: a python list of ids. (Should support pyarrow, eventually).


    returns: a pyarrow list with one element per stack.
    """
    ids = sorted(ids)
    dists = distances(ids)
    broken = breaks(dists, min_length, max_length)
    indices = np.cumsum(np.array(broken))
    indices = np.insert(indices, 0, 0)
    chunked = pa.ListArray.from_arrays(indices, ids)
    return chunked
