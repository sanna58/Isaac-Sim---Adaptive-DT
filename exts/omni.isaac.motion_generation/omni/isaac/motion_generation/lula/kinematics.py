# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import List, Optional, Tuple

import lula
import numpy as np
from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices
from omni.isaac.core.utils.stage import get_stage_units

from ...motion_generation.kinematics_interface import KinematicsSolver
from . import utils as lula_utils
from .interface_helper import LulaInterfaceHelper


class LulaKinematicsSolver(KinematicsSolver):
    """A Lula-based implementaion of the KinematicsSolver interface.  Lula uses a URDF file describing the robot and
    a custom yaml file that specifies the cspace of the robot and other parameters.

    This class provides functions beyond the KinematicsSolver interface for getting and setting solver parameters.
    Inverse kinematics is solved quickly by first approximating a solution with cyclic coordinate descent (CCD) and then
    refining the solution with a second-order method (bfgs).  As such, parameters for both solvers are available and changable
    as properties of this class.

    Args:
        robot_description_path (str): path to a robot description yaml file describing the cspace of the robot and other relevant parameters
        urdf_path (str): path to a URDF file describing the robot
        robot_description (Optional[lula.RobotDescription]):  An initialized lula.RobotDescription object.  Other Lula-based classes such as RmpFlow may use
            a lula.RobotDescription object that they have already created to initialize a LulaKinematicsSolver.  When specified, the provided file paths are unused.
            Defaults to None.
    """

    def __init__(
        self, robot_description_path: str, urdf_path: str, robot_description: Optional[lula.RobotDescription] = None
    ):
        # Other Lula classes may initialize a KinematicsSolver using a pre-existing lula robot_description

        if robot_description is None:
            self._robot_description = lula.load_robot(robot_description_path, urdf_path)
        else:
            self._robot_description = robot_description
        self._kinematics = self._robot_description.kinematics()
        self._ik_config = lula.CyclicCoordDescentIkConfig()

        LulaInterfaceHelper.__init__(self, self._robot_description)  # for tracking robot base

        self._meters_per_unit = get_stage_units()

        self._default_orientation_tolerance = self._lula_orientation_tol_to_rad_tol(
            self._ik_config.orientation_tolerance
        )
        self._default_position_tolerance = self._ik_config.position_tolerance

        self._default_bfgs_orientation_weight = self._ik_config.bfgs_orientation_weight
        self._default_ccd_orientation_weight = self._ik_config.ccd_orientation_weight

        self._default_cspace_seeds = []

    @property
    def bfgs_cspace_limit_biasing(self):
        return self._ik_config.bfgs_cspace_limit_biasing

    @bfgs_cspace_limit_biasing.setter
    def bfgs_cspace_limit_biasing(self, value):
        self._ik_config.bfgs_cspace_limit_biasing = value

    @property
    def bfgs_cspace_limit_biasing_weight(self):
        return self._ik_config.bfgs_cspace_limit_biasing_weight

    @bfgs_cspace_limit_biasing_weight.setter
    def bfgs_cspace_limit_biasing_weight(self, value):
        self._ik_config.bfgs_cspace_limit_biasing_weight = value

    @property
    def bfgs_cspace_limit_penalty_region(self):
        return self._ik_config.bfgs_cspace_limit_penalty_region

    @bfgs_cspace_limit_penalty_region.setter
    def bfgs_cspace_limit_penalty_region(self, value):
        self._ik_config.bfgs_cspace_limit_penalty_region = value

    @property
    def bfgs_gradient_norm_termination(self):
        return self._ik_config.bfgs_gradient_norm_termination

    @bfgs_gradient_norm_termination.setter
    def bfgs_gradient_norm_termination(self, value):
        self._ik_config.bfgs_gradient_norm_termination = value

    @property
    def bfgs_gradient_norm_termination_coarse_scale_factor(self):
        return self._ik_config.bfgs_gradient_norm_termination_coarse_scale_factor

    @bfgs_gradient_norm_termination_coarse_scale_factor.setter
    def bfgs_gradient_norm_termination_coarse_scale_factor(self, value):
        self._ik_config.bfgs_gradient_norm_termination_coarse_scale_factor = value

    @property
    def bfgs_max_iterations(self):
        return self._ik_config.bfgs_max_iterations

    @bfgs_max_iterations.setter
    def bfgs_max_iterations(self, value):
        self._ik_config.bfgs_max_iterations = value

    @property
    def bfgs_orientation_weight(self):
        return self._default_bfgs_orientation_weight

    @bfgs_orientation_weight.setter
    def bfgs_orientation_weight(self, value):
        self._default_bfgs_orientation_weight = value

    @property
    def bfgs_position_weight(self):
        return self._ik_config.bfgs_position_weight

    @bfgs_position_weight.setter
    def bfgs_position_weight(self, value):
        self._ik_config.bfgs_position_weight = value

    @property
    def ccd_bracket_search_num_uniform_samples(self):
        return self._ik_config.ccd_bracket_search_num_uniform_samples

    @ccd_bracket_search_num_uniform_samples.setter
    def ccd_bracket_search_num_uniform_samples(self, value):
        self._ik_config.ccd_bracket_search_num_uniform_samples = value

    @property
    def ccd_descent_termination_delta(self):
        return self._ik_config.ccd_descent_termination_delta

    @ccd_descent_termination_delta.setter
    def ccd_descent_termination_delta(self, value):
        self._ik_config.ccd_descent_termination_delta = value

    @property
    def ccd_max_iterations(self):
        return self._ik_config.ccd_max_iterations

    @ccd_max_iterations.setter
    def ccd_max_iterations(self, value):
        self._ik_config.ccd_max_iterations = value

    @property
    def ccd_orientation_weight(self):
        return self._default_ccd_orientation_weight

    @ccd_orientation_weight.setter
    def ccd_orientation_weight(self, value):
        self._default_ccd_orientation_weight = value

    @property
    def ccd_position_weight(self):
        return self._ik_config.ccd_position_weight

    @ccd_position_weight.setter
    def ccd_position_weight(self, value):
        self._ik_config.ccd_position_weight = value

    @property
    def irwin_hall_sampling_order(self):
        return self._ik_config.irwin_hall_sampling_order

    @irwin_hall_sampling_order.setter
    def irwin_hall_sampling_order(self, value):
        self._ik_config.irwin_hall_sampling_order = value

    @property
    def max_num_descents(self):
        return self._ik_config.max_num_descents

    @max_num_descents.setter
    def max_num_descents(self, value):
        self._ik_config.max_num_descents = value

    @property
    def sampling_seed(self):
        return self._ik_config.sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, value):
        self._ik_config.sampling_seed = value

    def set_robot_base_pose(self, robot_position: np.array, robot_orientation: np.array) -> None:
        LulaInterfaceHelper.set_robot_base_pose(self, robot_position, robot_orientation)

    def get_joint_names(self) -> List[str]:
        return LulaInterfaceHelper.get_active_joints(self)

    def get_all_frame_names(self) -> List[str]:
        return self._kinematics.frame_names()

    def compute_forward_kinematics(
        self, frame_name: str, joint_positions: np.array, position_only: Optional[bool] = False
    ) -> Tuple[np.array, np.array]:
        """Compute the position of a given frame in the robot relative to the USD stage global frame

        Args:
            frame_name (str): Name of robot frame on which to calculate forward kinematics
            joint_positions (np.array): Joint positions for the joints returned by get_joint_names()
            position_only (bool): Lula Kinematics ignore this flag and always computes both position and orientation

        Returns:
            Tuple[np.array,np.array]:
            frame_positions: (3x1) vector describing the translation of the frame relative to the USD stage origin

            frame_rotation: (3x3) rotation matrix describing the rotation of the frame relative to the USD stage global frame
        """

        return LulaInterfaceHelper.get_end_effector_pose(self, joint_positions, frame_name)

    def compute_inverse_kinematics(
        self,
        frame_name: str,
        target_position: np.array,
        target_orientation: np.array = None,
        warm_start: np.array = None,
        position_tolerance: float = None,
        orientation_tolerance: float = None,
    ) -> Tuple[np.array, bool]:
        """Compute joint positions such that the specified robot frame will reach the desired translations and rotations.
        Lula Kinematics interpret the orientation tolerance as being the maximum rotation separating any standard axes.
        e.g. For a tolerance of .1: The X axes, Y axes, and Z axes of the rotation matrices may independently be as far as .1 radians apart

        Default values for position and orientation tolerances may be seen and changed with setter and getter functions.

        Args:
            frame_name (str): name of the target frame for inverse kinematics
            target_position (np.array): target translation of the target frame (in stage units) relative to the USD stage origin
            target_orientation (np.array): target orientation of the target frame relative to the USD stage global frame. Defaults to None.
            warm_start (np.array): a starting position that will be used when solving the IK problem.  If default cspace seeds have been set,
                the warm start will be given priority, but the default seeds will still be used. Defaults to None.
            position_tolerance (float): l-2 norm of acceptable position error (in stage units) between the target and achieved translations. Defaults to None.
            orientation tolerance (float): magnitude of rotation (in radians) separating the target orientation from the achieved orienatation.
                orientation_tolerance is well defined for values between 0 and pi.  Defaults to None.

        Returns:
            Tuple[np.array,bool]:
            joint_positions: in the order specified by get_joint_names() which result in the target frame acheiving the desired position

            success: True if the solver converged to a solution within the given tolerances
        """

        if position_tolerance is None:
            self._ik_config.position_tolerance = self._default_position_tolerance
        else:
            self._ik_config.position_tolerance = position_tolerance * self._meters_per_unit

        if orientation_tolerance is None:
            self._ik_config.orientation_tolerance = self._rad_tol_to_lula_orientation_tol(
                self._default_orientation_tolerance
            )
        else:
            self._ik_config.orientation_tolerance = self._rad_tol_to_lula_orientation_tol(orientation_tolerance)

        if target_orientation is None:
            target_orientation = np.array([1, 0, 0, 0])
            self._ik_config.orientation_tolerance = 2.0
            self._ik_config.ccd_orientation_weight = 0.0
            self._ik_config.bfgs_orientation_weight = 0.0
        else:
            self._ik_config.ccd_orientation_weight = self._default_ccd_orientation_weight
            self._ik_config.bfgs_orientation_weight = self._default_bfgs_orientation_weight

        rot = quats_to_rot_matrices(target_orientation).astype(np.float64)
        pos = target_position.astype(np.float64) * self._meters_per_unit

        pos, rot = LulaInterfaceHelper._get_pose_rel_robot_base(self, pos, rot)

        target_pose = lula_utils.get_pose3(pos, rot)

        if warm_start is not None:
            seeds = [warm_start]
            seeds.extend(self._default_cspace_seeds)
            self._ik_config.cspace_seeds = seeds
        else:
            self._ik_config.cspace_seeds = self._default_cspace_seeds

        results = lula.compute_ik_ccd(self._kinematics, target_pose, frame_name, self._ik_config)

        return results.cspace_position, results.success

    def supports_collision_avoidance(self) -> bool:
        """Lula Inverse Kinematics do not support collision avoidance with USD obstacles

        Returns:
            bool: Always False
        """

        return False

    def set_default_orientation_tolerance(self, tolerance: float) -> None:
        """Default orientation tolerance to be used when calculating IK when none is specified

        Args:
            tolerance (float): magnitude of rotation (in radians) separating the target orientation from the achieved orienatation.
                orientation_tolerance is well defined for values between 0 and pi.
        """

        self._default_orientation_tolerance = tolerance

    def set_default_position_tolerance(self, tolerance: float) -> None:
        """Default position tolerance to be used when calculating IK when none is specified

        Args:
            tolerance (float): l-2 norm of acceptable position error (in stage units) between the target and achieved translations
        """
        self._default_position_tolerance = tolerance * self._meters_per_unit

    def set_default_cspace_seeds(self, seeds: np.array) -> None:
        """Set a list of cspace seeds that the solver may use as starting points for solutions

        Args:
            seeds (np.array): An N x num_dof list of cspace seeds
        """
        self._default_cspace_seeds = seeds

    def get_default_orientation_tolerance(self) -> float:
        """Get the default orientation tolerance to be used when calculating IK when none is specified

        Returns:
            float: magnitude of rotation (in radians) separating the target orientation from the achieved orienatation.
                orientation_tolerance is well defined for values between 0 and pi.
        """
        return self._default_orientation_tolerance

    def get_default_position_tolerance(self) -> float:
        """Get the default position tolerance to be used when calculating IK when none is specified

        Returns:
            float: l-2 norm of acceptable position error (in stage units) between the target and achieved translations
        """
        return self._default_position_tolerance / self._meters_per_unit

    def get_default_cspace_seeds(self) -> List[np.array]:
        """Get a list of cspace seeds that the solver may use as starting points for solutions

        Returns:
            List[np.array]: An N x num_dof list of cspace seeds
        """
        return self._default_cspace_seeds

    def get_cspace_position_limits(self) -> Tuple[np.array, np.array]:
        """Get the default upper and lower joint limits of the active joints.

        Returns:
            Tuple[np.array, np.array]:
            default_lower_joint_position_limits : Default lower position limits of active joints

            default_upper_joint_position_limits : Default upper position limits of active joints
        """
        num_coords = self._kinematics.num_c_space_coords()

        lower = []
        upper = []
        for i in range(num_coords):
            limits = self._kinematics.c_space_coord_limits(i)
            lower.append(limits.lower)
            upper.append(limits.upper)

        c_space_position_upper_limits = np.array(upper, dtype=np.float64)
        c_space_position_lower_limits = np.array(lower, dtype=np.float64)

        return c_space_position_lower_limits, c_space_position_upper_limits

    def get_cspace_velocity_limits(self) -> np.array:
        """Get the default velocity limits of the active joints

        Returns:
            np.array: Default velocity limits of the active joints
        """
        num_coords = self._kinematics.num_c_space_coords()

        c_space_velocity_limits = np.array(
            [self._kinematics.c_space_coord_velocity_limit(i) for i in range(num_coords)], dtype=np.float64
        )
        return c_space_velocity_limits

    def get_cspace_acceleration_limits(self) -> np.array:
        """Get the default acceleration limits of the active joints.
        Default acceleration limits are read from the robot_description YAML file.

        Returns:
            np.array: Default acceleration limits of the active joints
        """
        num_coords = self._kinematics.num_c_space_coords()

        if self._kinematics.has_c_space_acceleration_limits():
            c_space_acceleration_limits = np.array(
                [self._kinematics.c_space_coord_acceleration_limit(i) for i in range(num_coords)], dtype=np.float64
            )
        else:
            c_space_acceleration_limits = None

        return c_space_acceleration_limits

    def get_cspace_jerk_limits(self) -> np.array:
        """Get the default jerk limits of the active joints.
        Default jerk limits are read from the robot_description YAML file.

        Returns:
            np.array: Default jerk limits of the active joints.
        """
        num_coords = self._kinematics.num_c_space_coords()

        if self._kinematics.has_c_space_jerk_limits():
            c_space_jerk_limits = np.array(
                [self._kinematics.c_space_coord_jerk_limit(i) for i in range(num_coords)], dtype=np.float64
            )
        else:
            c_space_jerk_limits = None

        return c_space_jerk_limits

    def _lula_orientation_tol_to_rad_tol(self, tol):
        # convert from lula IK orientation tolerance to radian magnitude tolerance
        # This function is the inverse of _rad_tol_to_lula_orientation_tol

        return np.arccos(1 - tol**2 / 2)

    def _rad_tol_to_lula_orientation_tol(self, tol):
        # convert from radian magnitude tolerance to lula IK orientation tolerance

        # Orientation tolerance in Lula is defined as the maximum l2-norm between rotation matrix columns paired by index.
        # e.g. rotating pi rad about the z axis maps to a norm of 2.0 when comparing the x columns

        return np.linalg.norm(np.subtract([1, 0], [np.cos(tol), np.sin(tol)]))


LulaKinematicsSolver.bfgs_cspace_limit_biasing.__doc__ = (
    lula.CyclicCoordDescentIkConfig.bfgs_cspace_limit_biasing.__doc__
)

LulaKinematicsSolver.bfgs_cspace_limit_biasing_weight.__doc__ = (
    lula.CyclicCoordDescentIkConfig.bfgs_cspace_limit_biasing_weight.__doc__
)

LulaKinematicsSolver.bfgs_cspace_limit_penalty_region.__doc__ = (
    lula.CyclicCoordDescentIkConfig.bfgs_cspace_limit_penalty_region.__doc__
)

LulaKinematicsSolver.bfgs_gradient_norm_termination.__doc__ = (
    lula.CyclicCoordDescentIkConfig.bfgs_gradient_norm_termination.__doc__
)

LulaKinematicsSolver.bfgs_gradient_norm_termination_coarse_scale_factor.__doc__ = (
    lula.CyclicCoordDescentIkConfig.bfgs_gradient_norm_termination_coarse_scale_factor.__doc__
)

LulaKinematicsSolver.bfgs_max_iterations.__doc__ = lula.CyclicCoordDescentIkConfig.bfgs_max_iterations.__doc__

LulaKinematicsSolver.bfgs_orientation_weight.__doc__ = lula.CyclicCoordDescentIkConfig.bfgs_orientation_weight.__doc__

LulaKinematicsSolver.bfgs_position_weight.__doc__ = lula.CyclicCoordDescentIkConfig.bfgs_position_weight.__doc__

LulaKinematicsSolver.ccd_bracket_search_num_uniform_samples.__doc__ = (
    lula.CyclicCoordDescentIkConfig.ccd_bracket_search_num_uniform_samples.__doc__
)

LulaKinematicsSolver.ccd_descent_termination_delta.__doc__ = (
    lula.CyclicCoordDescentIkConfig.ccd_descent_termination_delta.__doc__
)

LulaKinematicsSolver.ccd_max_iterations.__doc__ = lula.CyclicCoordDescentIkConfig.ccd_max_iterations.__doc__

LulaKinematicsSolver.ccd_orientation_weight.__doc__ = lula.CyclicCoordDescentIkConfig.ccd_orientation_weight.__doc__

LulaKinematicsSolver.ccd_position_weight.__doc__ = lula.CyclicCoordDescentIkConfig.ccd_position_weight.__doc__

LulaKinematicsSolver.irwin_hall_sampling_order.__doc__ = (
    lula.CyclicCoordDescentIkConfig.irwin_hall_sampling_order.__doc__
)

LulaKinematicsSolver.max_num_descents.__doc__ = lula.CyclicCoordDescentIkConfig.max_num_descents.__doc__

LulaKinematicsSolver.sampling_seed.__doc__ = lula.CyclicCoordDescentIkConfig.sampling_seed.__doc__
