"""
## To-dos

- [ ] flow model with RealNVP
- [ ] q_cut-off, to further select only those goals that are practical.
"""
import numpy as np
from params_proto.neo_proto import ParamsProto, Flag, Proto
from sklearn.neighbors import KernelDensity


class OMEGA(ParamsProto):
    # Exploration
    use_omega = Flag(default=False, to_value=True, help="flag for turning off mega")
    use_behavior_goal = Flag(default=False, to_value=True,
                             help="flag for using desired goal as opposed to achieved goal")
    no_q_cutoff = Flag(default=False, to_value=True, help="flag to turn off q cutoff")
    kde_bandwidth = Proto(0.1, help="the smoothing parameter in sklearn kernel density model")
    n = 10_000


kde = None


def simple_min_kernel_density_model(goal_pool, k, **rest):
    """
    sample achieved goals that are less traveled to

    Use a singleton pattern for simplicity
    """
    # sample goals that have been achieved
    # sample goals that have been achieved the least (according to a density model)
    # We need a kernel entropy model for estimating the achieved goals that are
    # minimum in density. (state marginal matching)
    global kde
    if kde is None:
        kde = KernelDensity(bandwidth=OMEGA.kde_bandwidth)

    kde.fit(goal_pool)
    scores = kde.score_samples(goal_pool)
    return goal_pool[np.argpartition(scores, k)[:k]]
