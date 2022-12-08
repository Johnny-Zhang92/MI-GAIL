from enum import Enum
from collections import namedtuple


class Mask(Enum):
    ABSORBING = -1.
    DONE = 0.
    NOT_DONE = 1.


TimeStep = namedtuple('TimeStep', ('state', 'action', 'next_state', 'reward', 'mask', 'done'))
TimeStep_keys = TimeStep._fields


def get_key_dims(s_dim: int, a_dim: int):
    return (s_dim, a_dim, s_dim, 1, 1, 1)
