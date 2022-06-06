import numpy as np


def swap_axis(a, range_1, range_2):
    a = a.copy()  # shallow copy does not work for object arrays.
    a[range_1], a[range_2] = a[range_2], a[range_1]
    return a


def cat_dict(d_a, d_b):
    d_new = {}
    for k, v in d_a.items():
        d_new[k] = np.concatenate([v, d_b.get(k, d_a)], axis=0)
    return d_new


def relabel(batch):
    # the dimensions: 10 + 3 + 15 + 15
    obs_keys = ['ob', 'o2']
    goal_keys = ['ag', 'bg', 'ag2', 'future_ag']
    batch_aug = {}
    for k, v in batch.items():
        if k in obs_keys:
            batch_aug[k] = swap_axis(v, [slice(0, None), slice(-30, -15)],
                                     [slice(0, None), slice(-15, None)], )
        elif k in goal_keys:
            batch_aug[k] = swap_axis(v, [slice(0, None), slice(0, 3)],
                                     [slice(0, None), slice(3, None)], )
        else:
            batch_aug[k] = v

    return cat_dict(batch, batch_aug)
