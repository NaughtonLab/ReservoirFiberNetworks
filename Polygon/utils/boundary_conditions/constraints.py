import numpy as np
from typing import Optional, Type, Union
from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase
from elastica.boundary_conditions import ConstraintBase
from elastica._linalg import _batch_matvec, _batch_matrix_transpose
from elastica._rotations import _get_rotation_matrix
from elastica.typing import SystemType, RodType
from numba import njit
from scipy.interpolate import CubicSpline


class DisplacementBC(ConstraintBase):
    """
    This boundary condition class allows the specified node/link to have a configurable constraint.
    Index can be passed to fix either or both the position or the director.
    Constraining position is equivalent to setting 0 translational DOF.
    Constraining director is equivalent to setting 0 rotational DOF.

    Examples
    --------
    How to fix all translational and rotational dof except allowing twisting around the z-axis in an inertial frame:

    >>> simulator.constrain(system).using(
    ...    DisplacementBC,
    ...    constrained_position_idx=(0,),
    ...    constrained_director_idx=(0,),
    ...    translational_constraint_selector=np.array([True, True, True]),
    ...    rotational_constraint_selector=np.array([True, True, False]),
    ... )

    How to allow the end of the rod to move in the XY plane and allow all rotational dof:

    >>> simulator.constrain(rod).using(
    ...    DisplacementBC,
    ...    constrained_position_idx=(-1,),
    ...    translational_constraint_selector=np.array([True, True, False]),
    ... )
    """

    def __init__(
        self,
        *fixed_data,
        translational_constraint_selector: Optional[np.ndarray] = None,
        rotational_constraint_selector: Optional[np.array] = None,
        duration: float = 10.0,
        sample_freq: int = 5,
        displacement_magnitude: float = 0.02,
        **kwargs,
    ):
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        translational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained. If None, we constrain all dofs.
        rotational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained.
        """
        super().__init__(**kwargs)
        pos, dir = [], []
        for data in fixed_data:
            if isinstance(data, np.ndarray) and data.shape == (3,):
                pos.append(data)
            elif isinstance(data, np.ndarray) and data.shape == (
                3,
                3,
            ):
                dir.append(data)
            else:
                # TODO: This part is prone to error.
                break

        if len(pos) > 0:
            # transpose from (blocksize, dim) to (dim, blocksize)
            self.fixed_positions = np.array(pos).transpose((1, 0))

        if len(dir) > 0:
            # transpose from (blocksize, dim, dim) to (dim, dim, blocksize)
            self.fixed_directors = np.array(dir).transpose((1, 2, 0))

        if translational_constraint_selector is None:
            translational_constraint_selector = np.array([True, True, True])
        if rotational_constraint_selector is None:
            rotational_constraint_selector = np.array([True, True, True])
        # properly validate the user-provided constraint selectors
        assert (
            type(translational_constraint_selector) == np.ndarray
            and translational_constraint_selector.dtype == bool
            and translational_constraint_selector.shape == (3,)
        ), "Translational constraint selector must be a 1D boolean array of length 3."
        assert (
            type(rotational_constraint_selector) == np.ndarray
            and rotational_constraint_selector.dtype == bool
            and rotational_constraint_selector.shape == (3,)
        ), "Rotational constraint selector must be a 1D boolean array of length 3."
        # cast booleans to int
        self.translational_constraint_selector = (
            translational_constraint_selector.astype(int)
        )
        self.rotational_constraint_selector = rotational_constraint_selector.astype(int)

        seed_value = 1234
        np.random.seed(seed_value)

        sample_time = np.ceil(duration).astype(int)
        x_sample = np.linspace(0, sample_time, sample_time*sample_freq + 1)

        y_sample = np.random.uniform(-1,1, size=sample_time*sample_freq+1)
        y_sample[0] = 0.0    
        self.spline = CubicSpline(x_sample, y_sample)
        self.spline_derivative = self.spline.derivative()
        self.displacement_magnitude = displacement_magnitude

    def constrain_values(self, system: SystemType, time: float) -> None:
        if self.constrained_position_idx.size:
            x_pos = self.displacement_magnitude * self.spline(time)
            self.nb_constrain_translational_values(
                system.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
                self.translational_constraint_selector,
                x_pos
            )

    def constrain_rates(self, system: SystemType, time: float) -> None:
        if self.constrained_position_idx.size:
            x_vel = self.displacement_magnitude * self.spline_derivative(time)
            self.nb_constrain_translational_rates(
                system.velocity_collection,
                self.constrained_position_idx,
                self.translational_constraint_selector,
                x_vel
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                system.director_collection,
                system.omega_collection,
                self.constrained_director_idx,
                self.rotational_constraint_selector,
            )

    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_values(
        position_collection, fixed_position_collection, indices, constraint_selector, x_pos
    ) -> None:
        """
        Computes constrain values in numba njit decorator

        Parameters
        ----------
        position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        fixed_position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which translational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If entry is 1, the concerning DoF will be constrained, otherwise it will be free for translation.
            Selector shall be specified in the inertial frame
        """
        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            # First term: add the old position values using the inverse constraint selector (e.g. DoF)
            # Second term: add the fixed position values using the constraint selector (e.g. constraint dimensions)
            position_collection[..., k] = (
                1 - constraint_selector
            ) * position_collection[
                ..., k
            ] + constraint_selector * fixed_position_collection[
                ..., i
            ]
            position_collection[0, k] += x_pos * constraint_selector[0]

    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_rates(
        velocity_collection, indices, constraint_selector, x_vel
    ) -> None:
        """
        Compute constrain rates in numba njit decorator

        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which translational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If entry is 1, the concerning DoF will be constrained, otherwise it will be free for translation.
            Selector shall be specified in the inertial frame
        """

        block_size = indices.size
        for i in range(block_size):
            k = indices[i]
            # set the dofs to 0 where the constraint_selector mask is active
            velocity_collection[..., k] = (
                1 - constraint_selector
            ) * velocity_collection[..., k]
            velocity_collection[0, k] += x_vel * constraint_selector[0]

    @staticmethod
    @njit(cache=True)
    def nb_constrain_rotational_rates(
        director_collection, omega_collection, indices, constraint_selector
    ) -> None:
        """
        Compute constrain rates in numba njit decorator

        Parameters
        ----------
        director_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which rotational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If an entry is 1, the rotation around the respective axis will be constrained,
            otherwise the system can freely rotate around the axis.
            The selector shall be specified in the lab frame
        """
        directors = director_collection[..., indices]

        # rotate angular velocities to lab frame
        omega_collection_lab_frame = _batch_matvec(
            _batch_matrix_transpose(directors), omega_collection[..., indices]
        )

        # apply constraint selector to angular velocities in lab frame
        omega_collection_not_constrained = (
            1 - np.expand_dims(constraint_selector, 1)
        ) * omega_collection_lab_frame

        # rotate angular velocities vector back to local frame and apply to omega_collection
        omega_collection[..., indices] = _batch_matvec(
            directors, omega_collection_not_constrained
        )

'''
class DisplacementBC_based_on_HelicalBuckling(ConstraintBase):
        """
        This is the boundary condition class for Helical
        Buckling case in Gazzola et. al. RSoS (2018).
        The applied boundary condition is twist and slack on to
        the first and last nodes and elements of the rod.

        `Example case (helical buckling) <https://github.com/GazzolaLab/PyElastica/blob/master/examples/HelicalBucklingCase/helicalbuckling.py>`_

            Attributes
            ----------
            twisting_time: float
                Time to complete twist.
            final_start_position: numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Position of first node of rod after twist completed.
            final_end_position: numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Position of last node of rod after twist completed.
            ang_vel: numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Angular velocity of rod during twisting time.
            shrink_vel: numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Shrink velocity of rod during twisting time.
            final_start_directors: numpy.ndarray
                3D (dim, dim, 1) array containing data with 'float' type.
                Directors of first element of rod after twist completed.
            final_end_directors: numpy.ndarray
                3D (dim, dim, 1) array containing data with 'float' type.
                Directors of last element of rod after twist completed.
        """

        def __init__(
            self,
            position_start: np.ndarray,
            position_end: np.ndarray,
            director_start: np.ndarray,
            director_end: np.ndarray,
            slack: float,
            number_of_rotations: float,
            **kwargs,
        ):
            """

            Displacement BC initializer

            Parameters
            ----------

            position_start : numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Initial position of first node.
            position_end : numpy.ndarray
                2D (dim, 1) array containing data with 'float' type.
                Initial position of last node.
            director_start : numpy.ndarray
                3D (dim, dim, blocksize) array containing data with 'float' type.
                Initial director of first element.
            director_end : numpy.ndarray
                3D (dim, dim, blocksize) array containing data with 'float' type.
                Initial director of last element.
            twisting_time : float
                Time to complete twist.
            slack : float
                Slack applied to rod.
            number_of_rotations : float
                Number of rotations applied to rod.
            """
            super().__init__(**kwargs)

            self.n_elem_rigid=0
            self.n_elem_soft=n_elem_soft

            ########################################
            self.compression_speed = compression_speed

            self.buckling_pertubation_time = 0.05/self.compression_speed
            self.buckling_return_time = self.buckling_pertubation_time 
            self.pertubation_speed=0.01/self.buckling_pertubation_time
            

            self.anneal_time = 15.0
            
            self.constrained_time=1/compression_speed
            #compressed_distance=1.8
            self.compression_time=compressed_distance/compression_speed
            #self.y_disp=0.175
            self.y_disp=0.185
            ########################################
            theta = number_of_rotations * np.pi
            self.final_start_directors = (
                _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
                @ director_start
            )  # rotation_matrix wants vectors 3,1
            self.final_end_directors = (
                _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
                @ director_end
            )  # rotation_matrix wants vectors 3,1


        def constrain_values(
            self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
        ) -> None:

            rod.director_collection[...,  self.n_elem_rigid] = self.final_start_directors
            rod.director_collection[..., self.n_elem_soft+self.n_elem_rigid-1] = self.final_end_directors

        def constrain_rates(
            self, rod: Union[Type[RodBase], Type[RigidBodyBase]], time: float
        ) -> None:

            #relaxing
            if time > self.compression_time: 
                # a=0

                rod.velocity_collection[..., :,  self.n_elem_rigid] = np.array([ 0.0,0.0,  0.0]) 
                rod.velocity_collection[..., :, self.n_elem_soft+self.n_elem_rigid] = np.array([ 0.0,0.0, 0.0]) 

            #constrained phase
            elif time > self.constrained_time: 

                rod.velocity_collection[..., :,  self.n_elem_rigid] = np.array([ 0.0,0.0,  self.compression_speed/2]) 
                rod.velocity_collection[..., :, self.n_elem_soft+self.n_elem_rigid] = np.array([ 0.0,0.0, -self.compression_speed/2]) 
            
            #let tip float
            
            elif time > self.buckling_pertubation_time+self.buckling_return_time: 
                phase_time=self.constrained_time-self.buckling_pertubation_time-self.buckling_return_time
            
                rod.velocity_collection[..., :,  self.n_elem_rigid] =  np.array([0.0,self.y_disp/phase_time, self.compression_speed/2])
                rod.velocity_collection[..., :, self.n_elem_soft+self.n_elem_rigid] = np.array([0.0,-self.y_disp/phase_time, -self.compression_speed/2])


            #return perturbation
            elif time >self.buckling_pertubation_time: 

                rod.velocity_collection[..., :,  self.n_elem_rigid] = np.array([0.0, self.pertubation_speed,self.compression_speed/2]) #start tip y velocities

                #rod.velocity_collection[..., 2,  self.n_elem_rigid]=0  #vz               
                rod.velocity_collection[..., :, self.n_elem_soft+self.n_elem_rigid] = np.array([0.0,-self.pertubation_speed, -self.compression_speed/2]) #end tip y and z velocities

            #apply perturbation to induce buckling 
            else: 

                rod.velocity_collection[..., :,  self.n_elem_rigid] = np.array([0.0, -self.pertubation_speed,self.compression_speed/2]) #start tip y velocities

                rod.velocity_collection[..., :, self.n_elem_soft+self.n_elem_rigid] = np.array([0.0,self.pertubation_speed, -self.compression_speed/2]) #end tip y and z velocities

            rod.omega_collection[..., self.n_elem_rigid] = 0.0
            rod.omega_collection[..., self.n_elem_soft+self.n_elem_rigid-1] = 0.0

'''