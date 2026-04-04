"""
Rotation Augmentation: random SO(3) rotation matrices.

Applied to: raw PMT data -> Geometry lookup -> -> HEALPix mapping -> tokenization.

 All spatial vectors are rotated, HEALPix mapping changes, all derived
 quantities are recalculated.

Reference: doc/ROTATION_AUG.md
"""

import numpy as np
import healpy as hp


def random_rotation_matrix() -> np.ndarray:
    """Generate uniform random SO(3) rotation matrix via quaternion method."""
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]],
    ], dtype=np.float32)
    return R
