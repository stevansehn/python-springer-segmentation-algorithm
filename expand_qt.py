import numpy as np

def expand_qt(original_qt, old_fs, new_fs, new_length):

    expanded_qt = np.zeros(new_length)

    indices_of_changes = np.argwhere(np.diff(original_qt)!=0)

    indices_of_changes = np.hstack((indices_of_changes.ravel(), len(original_qt)))

    start_index = 0

    for i in range (len(indices_of_changes)):

        end_index = indices_of_changes[i]

        mid_point = int (round((end_index - start_index)/2) + start_index)

        value_at_mid_point = original_qt[mid_point]

        expanded_start_index = int (round((start_index/old_fs)*new_fs))
        expanded_end_index = int (round((end_index/(old_fs))*new_fs))

        if(expanded_end_index > new_length):
            expanded_end_index = new_length

        expanded_qt[expanded_start_index:expanded_end_index] = value_at_mid_point

        start_index = end_index

    return expanded_qt