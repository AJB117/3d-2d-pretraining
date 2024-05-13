# import numpy as np
# cimport numpy as np
# from libc.stdlib cimport malloc, free

# cpdef np.ndarray generate_pairs(np.ndarray num_nodes_per_batch, int max_samples):
#     cdef int i, offset
#     cdef int num_batches = num_nodes_per_batch.shape[0]
#     cdef list pairs = []
#     cdef np.ndarray combinations

#     for i in range(num_batches):
#         offset = np.sum(num_nodes_per_batch[:i])
#         if max_samples > 0:
#             combinations = np.random.randint(0, num_nodes_per_batch[i], (max_samples, 2), dtype=np.int32)
#         else:
#             full_range = np.arange(num_nodes_per_batch[i], dtype=np.int32)
#             combinations = np.array([(x, y) for x in full_range for y in full_range if x != y], dtype=np.int32)

#         # Adding reverse pairs and offset
#         combinations = np.vstack((combinations, combinations[:, [1, 0]]))
#         pairs.append(combinations + offset)

#     return np.concatenate(pairs).astype(np.int32)

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generate_pairs(np.ndarray[np.int64_t, ndim=1] num_nodes_per_batch, int max_samples):
    cdef int i, n, offset = 0
    cdef int total_pairs = 0
    cdef np.ndarray[int, ndim=2] combinations, all_combinations
    cdef np.ndarray[int, ndim=1] nodes_indices
    cdef int num_batches = num_nodes_per_batch.shape[0]
    cdef list pairs = []

    # Calculate total number of pairs to preallocate all_combinations array
    if max_samples > 0:
        total_pairs = 2 * max_samples * num_batches  # Each batch will contribute exactly 2*max_samples pairs
    else:
        for n in num_nodes_per_batch:
            total_pairs += 2 * n * (n - 1)  # full combinations multiplied by 2 for the reversed pairs

    all_combinations = np.empty((total_pairs, 2), dtype=np.int32)
    offset = 0
    total_pairs = 0  # Reuse to keep track of the fill index

    for i in range(num_batches):
        n = num_nodes_per_batch[i]
        if max_samples > 0:
            combinations = np.random.randint(0, n, (max_samples, 2), dtype=np.int32)
        else:
            nodes_indices = np.arange(n, dtype=np.int32)
            combinations = np.empty((n * (n - 1), 2), dtype=np.int32)
            idx = 0
            for x in range(n):
                for y in range(n):
                    if x != y:
                        combinations[idx, 0] = x
                        combinations[idx, 1] = y
                        idx += 1

        # Add reverse pairs
        reversed_combinations = combinations[:, [1, 0]]
        batch_combinations = np.vstack((combinations, reversed_combinations))

        # Adjust indices based on the offset
        batch_combinations += offset
        all_combinations[total_pairs:total_pairs + batch_combinations.shape[0], :] = batch_combinations
        total_pairs += batch_combinations.shape[0]
        offset += n

    return all_combinations
