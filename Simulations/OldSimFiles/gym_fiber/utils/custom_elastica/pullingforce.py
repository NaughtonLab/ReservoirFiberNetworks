from elastica.external_forces import NoForces
from elastica.typing import SystemType, RodType
from numba import njit
import numpy as np

class PullingForce(NoForces):
    """
    This class applies constant forces on a specific node.

        Attributes
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        hold_time: float
            Applied forces are held for this length of time before being released. 

    """

    def __init__(self, node_idx, pulling_force, ramp_up_time):
        """

        Parameters
        ----------
        pulling_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        hold_time: float
            Applied forces are held for this length of time before being released. 

        """
        super(PullingForce, self).__init__()
        self.node_idx = node_idx
        self.pulling_force = pulling_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time
        # assert hold_time >= 0.0
        self.hold_time = 0.0

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_pulling_forces(
            system.external_forces,
            self.node_idx,
            self.pulling_force,
            time,
            self.ramp_up_time,
            self.hold_time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_pulling_forces(
        external_forces, node_idx, pulling_force, time, ramp_up_time, hold_time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        hold_time: float
            Applied forces are held for this length of time before being released. 

        Returns
        -------

        """

        # rampup factor
        factor = min(1.0, time / ramp_up_time)

        # factor2 = min(0.0, time - (ramp_up_time+hold_time))
        # if time >= ramp_up_time + hold_time:
        #     factor = 0.0

        external_forces[..., node_idx] += pulling_force * factor