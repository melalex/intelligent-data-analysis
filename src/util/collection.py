from collections import defaultdict


def list_to_index_dict(lst):
    index_dict = defaultdict(list)
    for idx, value in enumerate(lst):
        index_dict[value].append(idx)
    return dict(index_dict)
