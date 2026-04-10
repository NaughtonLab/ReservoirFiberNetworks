from elastica.external_forces import NoForces
from elastica.typing import SystemType, RodType
from numba import njit
import numpy as np

class PointForce(NoForces):
    """
    This class applies constant forces on a specific node.
        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to
            last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        hold_time: float
            Applied forces are held for this length of time before being released.

    """

    def __init__(self, node_idx, point_force, ramp_up_time, hold_time):
        """

        Parameters
        ----------
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        hold_time: float
            Applied forces are held for this length of time before being released. 

        """
        super(PointForce, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time
        assert hold_time >= 0.0
        self.hold_time = hold_time

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            time,
            self.ramp_up_time,
            self.hold_time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, time, ramp_up_time, hold_time
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
        if time >= ramp_up_time + hold_time:
             factor = 0.0

        external_forces[..., node_idx] += point_force * factor


class PointForceSinsusoidal(NoForces):
    """
    This class applies sinusoidal forces on a specific node.
        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to
            last node of the system.

    """

    def __init__(self, node_idx, point_force, time_period_1, time_period_2):
        """

        Parameters
        ----------
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        node_idx: int
            Index of the node where force is applied.
        time_period_1: float
            Time period of the first sine wave.
        time_period_2: float
            Time period of the second sine wave.

        """
        super(PointForceSinsusoidal, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        self.time_period_1 = time_period_1
        self.time_period_2 = time_period_2

    def apply_forces(self, system: SystemType, time=0.0):
        # self.point_force = 
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            time,
            self.time_period_1,
            self.time_period_2,
        )

    
    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, time, time_period_1, time_period_2
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
        time_period_1: float
            Time period of the first sine wave.
        time_period_2: float
            Time period of the second sine wave.

        Returns
        -------

        """

        external_forces[..., node_idx] += np.sin(time * (2 * np.pi)/time_period_1) * np.sin(time * (2 * np.pi)/time_period_2) * point_force


class PointForceSpline(NoForces):
    """
    This class applies forces on a specific node that vary according to a spline function.
        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to 
            last node of the system.
        spline: callable
            A spline function that defines how the force varies with time.

    """

    def __init__(self, node_idx, point_force, spline):
        """

        Parameters
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        spline: callable
            A spline function that defines how the force varies with time.

        """
        super(PointForceSpline, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        self.hold_time = 0.0
        self.s = spline

    def apply_forces(self, system: SystemType, time=0.0):
        self.ft = self.s(time)
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            self.ft,
        )

    
    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, ft
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        ft: float
            Scaling factor for the force.
        Returns
        -------

        """

        external_forces[..., node_idx] += ft * point_force

class PointForceSplineVaryingSinusoidal(NoForces):
    """
    This class applies sinusoidal forces with a time-varying frequency on a specific node.
    The frequency varies according to a predefined spline function.
        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to
            last node of the system.
        spline: callable
            A spline function that defines how the frequency varies with time.

    """

    def __init__(self, node_idx, point_force, spline):
        """

        Parameters
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        spline: callable
            A spline function that defines how the frequency varies with time.

        """
        super(PointForceSplineVaryingSinusoidal, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        self.s = spline

    def apply_forces(self, system: SystemType, time=0.0):
        self.ft = np.sin(2*np.pi*self.s(time)*time)
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            self.ft,
        )

    
    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, ft
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        point_force: float
            Amplitude of the sine wave.
        ft: float
            Scaling factor for the force.

        Returns
        -------

        """

        external_forces[..., node_idx] += ft * point_force

class PointForceHoldVaryingSinsusoidal(NoForces):
    """
    This class applies sinuosoidal forces with a time-varying frequency on a specific node.
    This frequency varies according to a predefined array of hold times and frequencies.

        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: float
            Amplitude of the sine wave.
        hold_time_freq_array: numpy.ndarray
            2D array where each row contains [frequency, hold_time_start, hold_time_end]

    """

    def __init__(self, node_idx, point_force, hold_time_freq_array):
        """

        Parameters
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: float
            Amplitude of the sine wave.
        hold_time_freq_array: numpy.ndarray
            2D array where each row contains [frequency, hold_time_start, hold_time_end]
        """
        super(PointForceHoldVaryingSinsusoidal, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        self.hold_time_freq_array = hold_time_freq_array

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            time,
            self.hold_time_freq_array,
        )

    
    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, time, hold_time_freq_array
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        point_force: float
            Amplitude of the sine wave.
        time: float
        hold_time_freq_array: numpy.ndarray
            2D array where each row contains [frequency, hold_time_start, hold_time_end]

        Returns
        -------

        """

        idx = np.where((hold_time_freq_array[:, 1]<= time) & (hold_time_freq_array[:, 2]> time))[0][0]
        freq = hold_time_freq_array[idx, 0]
        external_forces[..., node_idx] += np.sin(time * (2 * np.pi) * freq) * point_force

class PointForceImpulse(NoForces):
    """
    This class applies impulse force on a specific node.

        Attributes
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: float
            Magnitude of the force applied to the node.
        t0: float
            Time at which the impulse starts.
        duration: float
            Duration for which the impulse is applied.

    """

    def __init__(self, node_idx, point_force, t0, duration):
        """

        Parameters
        ----------
        node_idx: int
            Index of the node where force is applied.
        point_force: float
            Magnitude of the force applied to the node.
        t0: float
            Time at which the impulse starts.
        duration: float
            Duration for which the impulse is applied.

        """
        super(PointForceImpulse, self).__init__()
        self.node_idx = node_idx
        self.point_force = point_force
        self.t0 = t0
        self.duration = duration

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_end_point_forces(
            system.external_forces,
            self.node_idx,
            self.point_force,
            time,
            self.t0,
            self.duration,
        )

    
    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, node_idx, point_force, time, t0, duration
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        point_force: float
            Magnitude of the force applied to the node.
        time: float
        t0: float
            Time at which the impulse starts.
        duration: float
            Duration for which the impulse is applied.

        Returns
        -------

        """

        if t0 <= time < (t0+duration):
            impulse = 1
        else:
            impulse = 0

        external_forces[..., node_idx] += impulse * point_force