# Copyright (c) 2022, NVIDIA  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
from collections import OrderedDict
import copy
import math
import numpy as np
import random
import sys
import time

from omni.isaac.cortex.cortex_object import CortexObject

from omni.isaac.cortex.df import *
from omni.isaac.cortex.dfb import DfRobotApiContext, DfApproachGrasp, DfCloseGripper, DfOpenGripper, make_go_home
import omni.isaac.cortex.math_util as math_util
from omni.isaac.cortex.motion_commander import MotionCommand, PosePq
# from omni.isaac.examples.yumi_dbcortex.yumi_dbcortex_extension import YumidbCortexExtension

import omni
from pxr import Usd, UsdGeom
import sqlite3
# import ollama
# import re

def make_grasp_T(t, ay, offset=np.array([0.0,0.0,0.0])):
    az = math_util.normalized(-t)
    print("t :",-t,"(result of az", az)
    ax = np.cross(ay, az)

    T = np.eye(4)
    T[:3, 0] = ax
    T[:3, 1] = ay
    T[:3, 2] = az
    T[:3, 3] = t +offset

    return T


def make_block_grasp_Ts(block_pick_height, offset=np.array([0.0,0.0,0.0])):

    R = np.eye(3)

    Ts = []
    for i in range(3):
        t = block_pick_height * R[:, i]
        for j in range(2):
            ay = R[:, (i + j + 1) % 3]
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    Ts.append(make_grasp_T(s1 * t, s2 * ay,offset=offset))

    return Ts


def get_world_block_grasp_Ts(
    obj_T,
    obj_grasp_Ts,
    axis_x_filter=None,
    axis_x_filter_thresh=0.1,
    axis_y_filter=None,
    axis_y_filter_thresh=0.1,
    axis_z_filter=None,
    axis_z_filter_thresh=0.1,
):
    world_grasp_Ts = []
    for gT in obj_grasp_Ts:
        world_gT = obj_T.dot(gT)
        if axis_x_filter is not None and (
            1.0 - world_gT[:3, 0].dot(math_util.normalized(axis_x_filter)) > axis_x_filter_thresh
        ):
            continue
        if axis_y_filter is not None and (
            1.0 - world_gT[:3, 1].dot(math_util.normalized(axis_y_filter)) > axis_y_filter_thresh
        ):
            continue
        if axis_z_filter is not None and (
            1.0 - world_gT[:3, 2].dot(math_util.normalized(axis_z_filter)) > axis_z_filter_thresh
        ):
            continue

        world_grasp_Ts.append(world_gT)
        
    return world_grasp_Ts


def get_best_obj_grasp(obj_T, obj_grasp_Ts, eff_T, other_obj_Ts):
    """ Uses a manually defined score-based classifier for choosing which grasp to use on a given
    block.

    It chooses a grasp that's simultaneoulsy natural for the arm and avoids any nearby blocks.

    args:
        obj_T: The block object being grasped.
        obj_grasp_Ts: The grasp transforms in coordinates local to the block.
        eff_T: The current end-effector transform.
        other_obj_Ts: The transforms of all other surrounding blocks we want to consider.
    """
    Ts = get_world_block_grasp_Ts(obj_T, obj_grasp_Ts, axis_z_filter=np.array([0.0, 0.0, -1.0]))
    # Ts = get_world_block_grasp_Ts(obj_T, obj_grasp_Ts, axis_z_filter=np.array([1.0, 0.0, 0.0]))

    # This could happen if all the grasps are filtered out.
    if len(Ts) == 0:
        return None

    # Score each grasp based on how well the gripper's x-axis will correlate with the direction to
    # the robot (most natural configuration).
    obj_p = obj_T[:3, 3]
    v = math_util.normalized(-obj_p)
    block_y_axis = obj_T[:3, 1]
    # Score each of the candidate grasps based on how their world transform is oriented relative to
    # the robot and relative to nearby other blocks.
    scores = np.zeros(len(Ts))
    for i, grasp_T in enumerate(Ts):
        # The base score is a function of how the end-effector would be oriented relative to the
        # base of the robot.
        score = grasp_T[:3, 0].dot(v)
        grasp_x_axis = grasp_T[:3, 1]
        # score = np.dot(grasp_x_axis, block_y_axis)

        # For all surrounding objects, if the object is closer than 15cm away, add a proximity cost
        # (negative score) for the grasp based on whether the finger might clip it.
        for obj_T in other_obj_Ts:
            other_obj_p = obj_T[:3, 3]
            toward_other = other_obj_p - obj_p
            dist = np.linalg.norm(toward_other)
            if dist < 0.25:
                # Care about closer blocks more.
                w = np.exp(-0.5 * (dist / 0.15) ** 2)
                prox_score = -10.0 * w * (grasp_T[:3, 1].dot(math_util.normalized(toward_other))) ** 2
                score += prox_score

        scores[i] += score

    # Return the highest scoring transform.
    scored_Ts = zip(scores, Ts)
    T = max(scored_Ts, key=lambda v: v[0])[1]
    return T


def calc_grasp_for_block_T(context, block_T, desired_ax):
    ct = context
    candidate_Ts = get_world_block_grasp_Ts(block_T, ct.active_block.grasp_Ts, axis_z_filter=np.array([0.0, 0.0, -1.0]))
    if len(candidate_Ts) == 0:
        return None

    scored_candidate_Ts = [(np.dot(desired_ax, T[:3, 0]), T) for T in candidate_Ts]

    grasp_T = max(scored_candidate_Ts, key=lambda v: v[0])[1]
    return grasp_T


def calc_grasp_for_top_of_tower(context,axis):
    # TODO: use calc_grasp_for_block_T for this calculation
    ct = context
    block_target_T = ct.block_tower.next_block_placement_T
    # Original Script
    # candidate_Ts = get_world_block_grasp_Ts(
    #     block_target_T, ct.active_block.grasp_Ts, axis_z_filter=np.array([0.0, 0.0, -1.0])
    # ) 
    if axis=="y-axis":
        candidate_Ts = get_world_block_grasp_Ts(block_target_T, ct.active_block.grasp_Ts, axis_y_filter=np.array([0.0, 0.0, -1.0]))
    else:
        candidate_Ts = get_world_block_grasp_Ts(block_target_T, ct.active_block.grasp_Ts, axis_z_filter=np.array([0.0, 0.0, -1.0]))

    
    if len(candidate_Ts) == 0:
        return None

    # desired_ax = np.array([0.0, -1.0, 0.0])
    desired_ax = np.array([0.0, -1.0, 0.0])
    scored_candidate_Ts = [(np.dot(desired_ax, T[:3, 0]), T) for T in candidate_Ts]

    grasp_T = max(scored_candidate_Ts, key=lambda v: v[0])[1]
    
    return grasp_T


class BuildTowerContext(DfRobotApiContext):
    class Block:
        def __init__(self, i, obj, grasp_Ts):
            self.i = i
            self.obj = obj
            self.is_aligned = None
            self.grasp_Ts = grasp_Ts
            self.chosen_grasp = None
            self.collision_avoidance_enabled = True

        @property
        def has_chosen_grasp(self):
            return self.chosen_grasp is not None

        @property
        def name(self):
            return self.obj.name

        def get_world_grasp_Ts(
            self,
            axis_x_filter=None,
            axis_x_filter_thresh=0.1,
            axis_y_filter=None,
            axis_y_filter_thresh=0.1,
            axis_z_filter=None,
            axis_z_filter_thresh=0.1,
        ):
            return get_world_block_grasp_Ts(self.obj.get_transform(), self.grasp_Ts)

        def get_best_grasp(self, eff_T, other_obj_Ts):
            return get_best_obj_grasp(self.obj.get_transform(), self.grasp_Ts, eff_T, other_obj_Ts)

        def set_aligned(self):
            self.is_aligned = True

    class BlockTower:
        def __init__(self, tower_position, block_height, context):
            self.context = context

            order_preference = self.context.query_database(table_name="Sort_Order", condition_value="sequence_id", column_to_query="object_name", condition_column=None)
            self.desired_stack = order_preference
            # order_preference = ["Blue", "Yellow", "Green", "Red"]
            # self.desired_stack = [("%sCube" % c) for c in order_preference]
            self.tower_position = tower_position
            self.block_height = block_height
            self.stack = []
            self.prev_stack = None
            db_manager=DatabaseManager()

    



        
        @property
        def height(self):
            return len(self.stack)

        @property
        def top_block(self):
            if self.height == 0:
                return None
            return self.stack[-1]

        @property
        def current_stack_in_correct_order(self):
            """ Returns true if the current tower is in the correct order. False otherwise.
            """
            # print("in desired order before the loop",self.desired_stack)
            # print("in current order before the loop",self.stack)
            for pref_name, curr_block in zip(self.desired_stack, self.stack):
                # print("desired stack:", pref_name)
                # print("current stack:", curr_block.name)
                if curr_block.name != pref_name:
                    # print("in current order",curr_block.name)
                    # print("in desired order",pref_name)
                    return False

            return True

        @property
        def is_complete(self):
            # TODO: This doesn't account for desired vs actual ordering currently.
            if self.height != len(self.desired_stack):
                # print("First*********************")
                return False

            return self.current_stack_in_correct_order

        def stash_stack(self):
            self.prev_stack = self.stack
            self.stack = []

        def find_new_and_removed(self):
            if self.prev_stack is None:
                return [b for b in self.stack]

            i = 0
            while i < len(self.stack) and i < len(self.prev_stack):
                if self.stack[i] != self.prev_stack[i]:
                    break
                else:
                    i += 1

            new_blocks = self.stack[i:]
            removed_blocks = self.prev_stack[i:]
            return new_blocks, removed_blocks

        def set_top_block_to_aligned(self):
            
            print("self stack value",self.stack)
            if len(self.stack) > 0:
                self.stack[-1].is_aligned = True

        @property
        def next_block(self):
            """ Returns the first name in the desired stack that's not in the current stack. This
            models order preference, but not the strict requirement that the block stack be exactly
            in that order. Use current_stack_in_correct_order to additionally check that the current
            stack is in the correct order.
            """
            stack_names = [b.name for b in self.stack]
            for name in self.desired_stack:
                if name not in stack_names:
                    return self.context.blocks[name]

        @property
        def next_block_placement_T(self):
            
            # h = self.height
            # fractional_margin = 0.025
            # dz = (h + 0.5 + fractional_margin) * self.block_height
            # p = self.tower_position + np.array([0.0, 0.0, dz])
            db_manager=DatabaseManager()
            object_name = self.context.in_gripper.name

            Gripperaxis=db_manager.query_task_info(table_name="Travel_Op_Parmeters", condition_value=object_name, column_to_query="Gripper_Rotation", condition_column="object_id")
            if Gripperaxis=="y-axis":
                stage = omni.usd.get_context().get_stage()
                tree_node_path = "/World/Holder"
                tree_prim = stage.GetPrimAtPath(tree_node_path)
                if not tree_prim.IsValid():
                    print(f"Tree node at {tree_node_path} not found.")
                else:
                    # print(f"Tree node found at {tree_node_path}")

                    # Access properties or children of the tree node
                    # For example, get the tree's transform or children
                    xform = UsdGeom.Xform(tree_prim)
                    # transform_matrix = xform.GetLocalTransformation()
                    tray_xform = UsdGeom.Xformable(tree_prim)
                    transform= tray_xform.GetLocalTransformation()
                    getprim_pos = transform.ExtractTranslation()
                    # self.Drop_off_location = (POS_HOLD[0]+0.065+(0.075/2),POS_HOLD[1],POS_HOLD[2]+0.065+(0.075/2))
                    self.towerlocation=(getprim_pos[0]+0.012+(0.005/2) + 0.007,getprim_pos[1],getprim_pos[2]+0.065+(0.075/2))

                h = self.height
                db_manager=DatabaseManager()

                dx = h * 0.007 # Distance between one holder slot to the next one
                # p = self.tower_position + np.array([dx, 0.0, 0.0])
                object_name = self.context.in_gripper.name
                Drop_Height=db_manager.query_task_info(table_name="Drop_Op_Parmeters", condition_value=object_name, column_to_query="Drop_Height", condition_column="object_id")

                p = self.towerlocation + np.array([dx, 0.0, Drop_Height]) #<============== drop distance and offset direction
            
            elif Gripperaxis=="z-axis":
                h = self.height
                fractional_margin = 0.025
                dz = (h + 0.5 + fractional_margin) * self.block_height
                p = self.tower_position + np.array([0.0, 0.0, dz])
            
            
            # # Define rotation angles in radians
            # theta_z = np.radians(180)  # 90 degrees in radians
            # theta_x = np.radians(90)  # 90 degrees in radians
            # theta_y = np.radians(90)  # 90 degrees in radians
            # # Create rotation matrices for Z and X rotations
            # R_z = np.array([
            #     [np.cos(theta_z), -np.sin(theta_z), 0],
            #     [np.sin(theta_z), np.cos(theta_z), 0],
            #     [0, 0, 1]
            # ])

            # R_x = np.array([
            #     [1, 0, 0],
            #     [0, np.cos(theta_x), -np.sin(theta_x)],
            #     [0, np.sin(theta_x), np.cos(theta_x)]
            # ])

            # R_y = np.array([
            # [np.cos(theta_y), 0, np.sin(theta_y)],
            # [0, 1, 0],
            # [-np.sin(theta_y), 0, np.cos(theta_y)]
            # ])
            # # Combine the rotation matrices (apply Z then X)
            # R = np.dot(R_z, R_x)
            # R = R_x




            R = np.eye(3)
            T = math_util.pack_Rp(R, p)
            return T

    def __init__(self, robot, tower_position):
        super().__init__(robot)
        self.db_manager = DatabaseManager()

        self.robot = robot

        # self.block_height = 0.0515
        self.block_height = 0.002 # Same value specified in the yumi_examples_main.py script
        self.block_pick_height = 0.016 # Height between the motion commander and the pick_up object right before closing the gripper
        self.block_grasp_Ts = make_block_grasp_Ts(self.block_pick_height)
        self.tower_position = tower_position

        self.reset()

        self.add_monitors(
            [
                BuildTowerContext.monitor_perception,
                BuildTowerContext.monitor_block_tower,
                BuildTowerContext.monitor_gripper_has_block,
                BuildTowerContext.monitor_suppression_requirements,
                BuildTowerContext.monitor_diagnostics,
            ]
        )
        self.db_manager.check_table_schema("sequences")

    def query_database(self, table_name, condition_value, column_to_query, condition_column):
        # Example: Query task information
        result = self.db_manager.query_task_info(table_name, condition_value, column_to_query, condition_column)
        print(f"Query result for task '{table_name}': {result}")
        return result

    def update_db_data(self, table_name, column_to_update, new_value, condition_column, condition_value):
        self.db_manager.update_table(table_name, column_to_update, new_value, condition_column, condition_value)
        

    def reset(self):
        self.blocks = OrderedDict()
        self.reached_target_T = False
        # print("loading blocks")
        for i, (name, cortex_obj) in enumerate(self.robot.registered_obstacles.items()):
            # print("{}) {}".format(i, name))

            # This behavior might be run either with CortexObjects (e.g. when synchronizing with a
            # sim/real world via ROS) or standard core API objects. If it's the latter, add the
            # CortexObject API.
            if not isinstance(cortex_obj, CortexObject):
                cortex_obj = CortexObject(cortex_obj)

            cortex_obj.sync_throttle_dt = 0.25
            self.blocks[name] = BuildTowerContext.Block(i, cortex_obj, self.block_grasp_Ts)
            # print("self.blocks[name] info",self.blocks[name])
        # print("self.blocks[name] info final",self.blocks[name])
        self.block_tower = BuildTowerContext.BlockTower(self.tower_position, self.block_height, self)

        self.active_block = None
        self.in_gripper = None
        self.placement_target_eff_T = None

        self.print_dt = 0.25
        self.next_print_time = None
        self.start_time = None

    @property
    def has_active_block(self):
        return self.active_block is not None

    def activate_block(self, name):
        self.active_block = self.blocks[name]

    def reset_active_block(self):
        if self.active_block is None:
            return

        self.active_block.chosen_grasp = None
        self.active_block = None

    @property
    def block_names(self):
        block_names = [name for name in self.blocks.keys()]
        return block_names

    @property
    def num_blocks(self):
        return len(self.blocks)

    def mark_block_in_gripper(self):
        eff_p = self.robot.arm.get_fk_p()
        blocks_with_dists = []
        for _, block in self.blocks.items():
            block_p, _ = block.obj.get_world_pose()
            blocks_with_dists.append((block, np.linalg.norm(eff_p - block_p)))

        closest_block, _ = min(blocks_with_dists, key=lambda v: v[1])
        self.in_gripper = closest_block

    def clear_gripper(self):
        self.in_gripper = None

    @property
    def is_gripper_clear(self):
        return self.in_gripper is None

    @property
    def gripper_has_block(self):
        return not self.is_gripper_clear

    @property
    def has_placement_target_eff_T(self):
        return self.placement_target_eff_T is not None

    @property
    def next_block_name(self):
        remaining_block_names = [b.name for b in self.find_not_in_tower()]
        
        if len(remaining_block_names) == 0:
            return None
        for name in self.block_tower.desired_stack:
            if name in remaining_block_names:
                break
        return name

    def find_not_in_tower(self):
        blocks = [block for (name, block) in self.blocks.items()]
    
        for b in self.block_tower.stack:
            blocks[b.i] = None
        not_in_tower = [b for b in blocks if b is not None]

        ordered_blocks = []
        for desired_name in self.block_tower.desired_stack:
            for block in not_in_tower:
                # print("block.name:_",block.name)
                if block.name == desired_name:
                    
                    ordered_blocks.append(block)
        # print("ordered_blocks :", ordered_blocks.name)
        return ordered_blocks   
        # return [b for b in blocks if b is not None]

    def print_tower_status(self):
        in_tower = self.block_tower.stack
        print("\nin tower:")
        for i, b in enumerate(in_tower):
            print(
                "%d) %s, aligned: %s, suppressed: %s"
                % (i, b.name, str(b.is_aligned), str(not b.collision_avoidance_enabled))
            )

        not_in_tower = self.find_not_in_tower()
        print("\nnot in tower:")
        for i, b in enumerate(not_in_tower):
            print(
                "%d) %s, aligned: %s, suppressed: %s"
                % (i, b.name, str(b.is_aligned), str(not b.collision_avoidance_enabled))
            )
        print()



    def monitor_perception(self):
        for _, block in self.blocks.items():
            obj = block.obj
            if not obj.has_measured_pose():
                continue
    
            measured_T = obj.get_measured_T()
            belief_T = obj.get_T()
    
            not_in_gripper = block != self.in_gripper
    
            eff_p = self.robot.arm.get_fk_p()
            sync_performed = False
            if not_in_gripper and np.linalg.norm(belief_T[:3, 3] - eff_p) > 0.05:
                sync_performed = True
                obj.sync_to_measured_pose()
            elif np.linalg.norm(belief_T[:3, 3] - measured_T[:3, 3]) > 0.15:
                sync_performed = True
                obj.sync_to_measured_pose()
    
        # Check for new blocks not already in context
        detected_blocks = {name for name in self.robot.registered_obstacles.keys()}
        existing_blocks = set(self.blocks.keys())
        new_blocks = detected_blocks - existing_blocks
    
        for name in new_blocks:
            cortex_obj = self.robot.registered_obstacles[name]
            if not isinstance(cortex_obj, CortexObject):
                cortex_obj = CortexObject(cortex_obj)
            cortex_obj.sync_throttle_dt = 0.25
    
            self.blocks[name] = BuildTowerContext.Block(
                len(self.blocks), cortex_obj, self.block_grasp_Ts
            )
            print(f"New block detected and added to pick preferences: {name}")
    def monitor_block_tower(self):
        """ Monitor the current state of the block tower.

        The block tower is determined as the collection of blocks at the tower location and their
        order by height above the table.
        """
        # self.update_blocks()
        stage = omni.usd.get_context().get_stage()
        tree_node_path = "/World/Holder"
        tree_prim = stage.GetPrimAtPath(tree_node_path)
        
        if not tree_prim.IsValid():
            print(f"Tree node at {tree_node_path} not found.")
        else:
            # print(f"Tree node found at {tree_node_path}")

            # Access properties or children of the tree node
            # For example, get the tree's transform or children
            xform = UsdGeom.Xform(tree_prim)
            # transform_matrix = xform.GetLocalTransformation()
            tray_xform = UsdGeom.Xformable(tree_prim)
            transform= tray_xform.GetLocalTransformation()
            getprim_pos = transform.ExtractTranslation()
            # self.Drop_off_location = (POS_HOLD[0]+0.065+(0.075/2),POS_HOLD[1],POS_HOLD[2]+0.065+(0.075/2))
            self.towerlocation=(getprim_pos[0]+0.012+(0.005/2) + 0.007,getprim_pos[1],getprim_pos[2]+0.065+(0.075/2))
        tower_xy = self.towerlocation[:2]
        
        
        # tower_xy = self.block_tower.tower_position[:2]
        # tower_yz = self.block_tower.tower_position[-2:] # changed this

        new_block_tower_sequence = []
        for name, block in self.blocks.items():
            if self.gripper_has_block and self.in_gripper.name == block.name:
                # Don't include any blocks currently in the gripper
                continue

            p, _ = block.obj.get_world_pose()
            block_xy = p[:2]
            block_z = p[2]
            # block_yz = p[-2:] # Changed this
            block_x = p[0] # Changed this
            # print("towerxy",tower_xy)
            dist_to_tower = np.linalg.norm(tower_xy - block_xy)
            # dist_to_tower = np.linalg.norm(tower_yz - block_yz) # changed this
            # print("distance to tower",name,":",dist_to_tower)
            # print(dist_to_tower)
            # thresh = self.block_height / 2
            thresh = (len(self.block_tower.stack) * 0.007) + 0.007
            # print("thresh",name,":",thresh)
            # thresh = 0.003 # Changed it to a distance higher that the distance between two slots in holder (0.007)

            if dist_to_tower <= thresh:
                # new_block_tower_sequence.append((block_z, block))
                new_block_tower_sequence.append((block_x, block)) # changed this
                

        if len(new_block_tower_sequence) > 1:
            new_block_tower_sequence.sort(key=lambda v: v[0])
            print(new_block_tower_sequence)

        self.block_tower.stash_stack()
        for _, block in new_block_tower_sequence:
            self.block_tower.stack.append(block)

        new_blocks, removed_blocks = self.block_tower.find_new_and_removed()
        for block in new_blocks:
            block.is_aligned = False

        for block in removed_blocks:
            block.is_aligned = None
        # Identify and print blocks not in the tower
        not_in_tower = self.find_not_in_tower()
        for i, block in enumerate(not_in_tower):
            block_name = block.name
            if block_name in self.block_tower.desired_stack:
                next_in_order = self.block_tower.desired_stack[self.block_tower.height]
                if block_name == next_in_order:
                    block.is_aligned = True  # Mark the block as aligned for placement
                else:
                    block.is_aligned = None
            
        print("\nBlocks not in tower:")
        for i, block in enumerate(not_in_tower):
            print(f"{i}) {block.name}, aligned: {block.is_aligned}")

    def monitor_gripper_has_block(self):
        if self.gripper_has_block:
            block = self.in_gripper
            _, block_p = math_util.unpack_T(block.obj.get_transform())
            eff_p = self.robot.arm.get_fk_p()
            if np.linalg.norm(block_p - eff_p) > 0.1:
                print("Block lost. Clearing gripper.")
                self.clear_gripper()

    def monitor_suppression_requirements(self):
        arm = self.robot.arm
        eff_T = arm.get_fk_T()
        eff_R, eff_p = math_util.unpack_T(eff_T)
        ax, ay, az = math_util.unpack_R(eff_R)

        target_p, _ = arm.target_prim.get_world_pose()

        toward_target = target_p - eff_p
        dist_to_target = np.linalg.norm(toward_target)

        blocks_to_suppress = []
        if self.gripper_has_block:
            blocks_to_suppress.append(self.in_gripper)

        for name, block in self.blocks.items():
            block_T = block.obj.get_transform()
            block_R, block_p = math_util.unpack_T(block_T)

            # If the block is close to the target and the end-effector is above the block (in z), then
            # suppress it.
            target_dist_to_block = np.linalg.norm(block_p - target_p)
            xy_dist = np.linalg.norm(block_p[:2] - target_p[:2])
            margin = 0.05
            # Add the block if either we're descending on the block, or they're neighboring blocks
            # during the descent.
            if (
                target_dist_to_block < 0.1
                and (xy_dist < 0.02 or eff_p[2] > block_p[2] + margin)
                or target_dist_to_block < 0.15
                and target_dist_to_block > 0.07
                and eff_p[2] > block_p[2] + margin
            ):
                if block not in blocks_to_suppress:
                    blocks_to_suppress.append(block)

        for block in blocks_to_suppress:
            if block.collision_avoidance_enabled:
                try:
                    arm.disable_obstacle(block.obj)
                    block.collision_avoidance_enabled = False
                except Exception as e:
                    print("error disabling obstacle")
                    import traceback

                    traceback.print_exc()

        for name, block in self.blocks.items():
            if block not in blocks_to_suppress:
                if not block.collision_avoidance_enabled:
                    arm.enable_obstacle(block.obj)
                    block.collision_avoidance_enabled = True

    def monitor_diagnostics(self):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            self.next_print_time = now + self.print_dt

        if now >= self.next_print_time:
            print("\n==========================================")
            print("time since start: %f sec" % (now - self.start_time))
            self.print_tower_status()
            self.next_print_time += self.print_dt

            if self.has_active_block:
                print("active block:", self.active_block.name)
            else:
                print("no active block")


class OpenGripperRd(DfRldsNode):
    def __init__(self, dist_thresh_for_open):
        super().__init__()
        self.dist_thresh_for_open = dist_thresh_for_open
        self.add_child("open_gripper", DfOpenGripper())

    def is_runnable(self):
        ct = self.context
        if self.context.is_gripper_clear and not self.context.robot.gripper.is_open():
            if ct.has_active_block and ct.active_block.has_chosen_grasp:
                grasp_T = ct.active_block.chosen_grasp
                eff_T = ct.robot.arm.get_fk_T()
                p1 = grasp_T[:3, 3]
                p2 = eff_T[:3, 3]
                dist_to_target = np.linalg.norm(p1 - p2)
                return dist_to_target < self.dist_thresh_for_open

    def decide(self):
        return DfDecision("open_gripper")


class ReachToBlockRd(DfRldsNode):
    def __init__(self):
        super().__init__()
        self.child_name = None

    def link_to(self, name, decider):
        self.child_name = name
        self.add_child(name, decider)

    def is_runnable(self):
        return self.context.is_gripper_clear

    def decide(self):
        return DfDecision(self.child_name)


class GoHome(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("go_home", make_go_home())

    def enter(self):
        self.context.robot.gripper.close()

    def decide(self):
        return DfDecision("go_home")


class ChooseNextBlockForTowerBuildUp(DfDecider):
    def __init__(self):
        super().__init__()
        # If conditions aren't good, we'll just go home.
        self.add_child("go_home", GoHome())
        self.child_name = None

    def link_to(self, name, decider):
        self.child_name = name
        self.add_child(name, decider)

    def decide(self):
        ct = self.context
        # print("test ct.next_block_name",ct.next_block_name)
        ct.active_block = ct.blocks[ct.next_block_name]
        # print("test ct.active _block",ct.active_block)

        # Check exceptions
        block_p, _ = ct.active_block.obj.get_world_pose()
        if np.linalg.norm(block_p) < 0.25:
            print("block too close to robot base: {}".format(ct.active_block.name))
            return DfDecision("go_home")
        elif np.linalg.norm(block_p) > 0.81:
            print("block too far away: {}".format(ct.active_block.name))
            return DfDecision("go_home")
        elif (
            self.context.block_tower.height >= 2
            and np.linalg.norm(block_p - self.context.block_tower.tower_position) < 0.15
        ):
            print("block too close to tower: {}".format(ct.active_block.name))
            return DfDecision("go_home")

        other_obj_Ts = [
            block.obj.get_transform() for block in ct.blocks.values() if ct.next_block_name != block.obj.name
        ]
        ct.active_block.chosen_grasp = ct.active_block.get_best_grasp(ct.robot.arm.get_fk_T(), other_obj_Ts)
        print("go home calculations;",ct.next_block_name,np.linalg.norm(block_p - self.context.block_tower.tower_position))
        return DfDecision(self.child_name, ct.active_block.chosen_grasp)

    def exit(self):
        self.context.active_block.chosen_grasp = None


class ChooseNextBlockForTowerTeardown(DfDecider):
    def __init__(self):
        super().__init__()
        self.child_name = None

    def link_to(self, name, decider):
        self.child_name = name
        self.add_child(name, decider)

    def decide(self):
        ct = self.context
        ct.active_block = ct.block_tower.top_block
        active_block_T = ct.active_block.obj.get_transform()
        ct.active_block.chosen_grasp = calc_grasp_for_block_T(ct, active_block_T, np.array([0.0, -1.0, 0.0]))
        return DfDecision(self.child_name, ct.active_block.chosen_grasp)

    def exit(self):
        self.context.active_block.chosen_grasp = None


class ChooseNextBlock(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("choose_next_block_for_tower", ChooseNextBlockForTowerBuildUp())
        self.add_child("choose_tower_block", ChooseNextBlockForTowerTeardown())

    def link_to(self, name, decider):
        for _, child in self.children.items():
            child.link_to(name, decider)

    def decide(self):
        if self.context.block_tower.current_stack_in_correct_order:
            print("tower in correct order")
            return DfDecision("choose_next_block_for_tower")
        
        else:
            print("tower in wrong order")
            return DfDecision("choose_tower_block")


class LiftState(DfState):
    """ A simple state which sends a target a distance command_delta_z above the current
    end-effector position until the end-effector has moved success_delta_z meters up.

    Args:
        command_delta_z: The delta offset up to shift the command away from the current end-effector
            position every cycle.
        success_delta_z: The delta offset up from the original end-effector position measured on
            entry required for exiting the state.
    """

    def __init__(self, command_delta_z, success_delta_z, cautious_command_delta_z=None):
        self.command_delta_z = command_delta_z
        self.cautious_command_delta_z = cautious_command_delta_z
        self.success_delta_z = success_delta_z

    def enter(self):
        # On entry, set the posture config to the current config so the movement is minimal.
        posture_config = self.context.robot.arm.articulation_subset.get_joints_state().positions.astype(float)
        self.context.robot.arm.set_posture_config(posture_config)

        self.success_z = self.context.robot.arm.get_fk_p()[2] + self.success_delta_z

    def closest_non_grasped_block_dist(self, eff_p):
        blocks_with_dists = []
        for name, block in self.context.blocks.items():
            block_p, _ = block.obj.get_world_pose()
            dist = np.linalg.norm(eff_p[:2] - block_p[:2])
            if dist > 0.03:
                # Only consider it if it's not grapsed (i.e. not too close to the gripper).
                blocks_with_dists.append((block, dist))

        closest_block, closest_dist = min(blocks_with_dists, key=lambda v: v[1])
        return closest_dist

    def step(self):
        pose = self.context.robot.arm.get_fk_pq()
        if pose.p[2] >= self.success_z:
            return None

        if self.cautious_command_delta_z is not None and self.closest_non_grasped_block_dist(pose.p) < 0.1:
            # Use the cautious command delta-z if it's specified and we're close to another block.
            pose.p[2] += self.cautious_command_delta_z
        else:
            pose.p[2] += self.command_delta_z

        self.context.robot.arm.send_end_effector(target_pose=pose)
        return self

    def exit(self):
        # On exit, reset the posture config back to the default value.
        self.context.robot.arm.set_posture_config_to_default()

        
# class SlideState(DfState):
#     """A state to slide the block along the x-axis by a specified distance."""

#     def __init__(self, slide_distance):
#         self.slide_distance = slide_distance
#         self.target_pose = None

#     def enter(self):
#         # Get the current end-effector pose.
#         eff_T = self.context.robot.arm.get_fk_T()
#         eff_R, eff_p = math_util.unpack_T(eff_T)

#         # Compute the target pose by moving along the x-axis.
#         eff_p[0] += self.slide_distance
#         self.target_pose = PosePq(eff_p, math_util.matrix_to_quat(eff_R))

#         # Send the sliding motion command.
#         self.context.robot.arm.send(MotionCommand(target_pose=self.target_pose))

#     def step(self):
#         # Check if the current pose matches the target pose.
#         eff_T = self.context.robot.arm.get_fk_T()
#         if math_util.transforms_are_close(
#             self.target_pose.to_T(), eff_T, p_thresh=0.005, R_thresh=0.01
#         ):
#             return None  # Exit this state.
#         return self  # Continue sliding.

#     def exit(self):
#         self.context.reached_target_T = False

class RotateState(DfState):
    """A state to rotate the cube by a specified angle around the z-axis."""

    def __init__(self, rotation_angle_degrees):
        self.rotation_angle_radians = np.radians(rotation_angle_degrees)
        self.target_pose = None

    def _calculate_rotation_matrix(self, angle, axis):
        """Generate a rotation matrix for a given angle (radians) and axis."""
        axis = math_util.normalized(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        x, y, z = axis
        rotation_matrix = np.array([
            [
                cos_theta + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_theta,
                x * z * one_minus_cos + y * sin_theta,
            ],
            [
                y * x * one_minus_cos + z * sin_theta,
                cos_theta + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_theta,
            ],
            [
                z * x * one_minus_cos - y * sin_theta,
                z * y * one_minus_cos + x * sin_theta,
                cos_theta + z * z * one_minus_cos,
            ],
        ])
        return rotation_matrix

    def enter(self):
        # Get the current end-effector pose.
        eff_T = self.context.robot.arm.get_fk_T()
        eff_R, eff_p = math_util.unpack_T(eff_T)

        # Compute the target rotation matrix.
        rotation_matrix = self._calculate_rotation_matrix(
            self.rotation_angle_radians, axis=np.array([0.0, 0.0, 1.0])  # Rotate around the z-axis.
        )
        target_R = rotation_matrix @ eff_R  # Apply the rotation.

        # Set the target pose.
        self.target_pose = PosePq(eff_p, math_util.matrix_to_quat(target_R))

        # Send the rotation motion command.
        self.context.robot.arm.send(MotionCommand(target_pose=self.target_pose))

    def step(self):
        # Check if the current pose matches the target pose.
        eff_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(
            self.target_pose.to_T(), eff_T, p_thresh=0.005, R_thresh=0.01
        ):
            return None  # Exit this state.
        return self  # Continue rotating.

    def exit(self):
        self.context.reached_target_T = True
        
        

class SlideState(DfState):
    """A state to slide the block in a specified direction by a specified distance."""
 
    def __init__(self, slide_distance, direction):
        self.slide_distance = slide_distance
        self.direction = direction
        self.target_pose = None
 
    def enter(self):
        # Get the current end-effector pose.
        eff_T = self.context.robot.arm.get_fk_T()
        eff_R, eff_p = math_util.unpack_T(eff_T)
 
        # Define direction unit vectors in local frame
        direction_map = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),
        }
 
        # Compute world direction vector by rotating local direction by eff_R
        if self.direction not in direction_map:
            raise ValueError(f"Invalid direction '{self.direction}'")
 
        local_dir = direction_map[self.direction]
        world_dir = eff_R @ local_dir
        eff_p += self.slide_distance * world_dir
 
        # Set target pose
        self.target_pose = PosePq(eff_p, math_util.matrix_to_quat(eff_R))
        self.context.robot.arm.send(MotionCommand(target_pose=self.target_pose))
 
    def step(self):
        eff_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(
            self.target_pose.to_T(), eff_T, p_thresh=0.005, R_thresh=0.01
        ):
            return None
        return self   
    def exit(self):
        self.context.reached_target_T = False
        ct=self.context
        db_manager = DatabaseManager()
        # if self.context.placement_target_eff_T == None:
        #         db_manager.update_table("Pick_Op_Parmeters", "operation_status", True, "object_id", ct.next_block_name)
        print(f"Updating operation_status for object_id: ")
        db_manager.update_table("Pick_Op_Parmeters", "Slide_state_status", True, "object_id", ct.next_block_name)



class PickBlockRd(DfStateMachineDecider, DfRldsNode):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    DfTimedDeciderState(DfCloseGripper(), activity_duration=1.5),#Change to 0.5 if the duration cahnge is not working
                    SlideState(slide_distance=0.008, direction="y"),  # along Y-axis
                    LiftState(command_delta_z=0.3, cautious_command_delta_z=0.03, success_delta_z=0.075),
                    DfWriteContextState(lambda ctx: ctx.mark_block_in_gripper()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )
        self.is_locked = False

    def is_runnable(self):
        ct = self.context
        if ct.has_active_block and ct.active_block.has_chosen_grasp:
            grasp_T = ct.active_block.chosen_grasp

            eff_T = self.context.robot.arm.get_fk_T()

            thresh_met = math_util.transforms_are_close(grasp_T, eff_T, p_thresh=0.005, R_thresh=0.005)
            return thresh_met

        return False

    def exit(self):
        self.context.reached_target_T = False  # <--- Reset after picking a new block 
        ct=self.context
        db_manager = DatabaseManager()
        # if self.context.placement_target_eff_T == None:
        #         db_manager.update_table("Pick_Op_Parmeters", "operation_status", True, "object_id", ct.next_block_name)
        print(f"Updating operation_status for object_id: ")
        db_manager.update_table("Pick_Op_Parmeters", "operation_status", True, "object_id", ct.next_block_name)




class TablePointValidator:
    def __init__(self, context):
        ct = context

        block_pts = [b.obj.get_world_pose()[0] for _, b in ct.blocks.items() if b != context.in_gripper]
        block_pts.append(ct.block_tower.tower_position)
        self.avoid_pts_with_dist_threshs = [(p[:2], 0.15) for p in block_pts]
        self.avoid_pts_with_dist_threshs.append((np.zeros(2), 0.35))

        self.center_p = np.array([0.3, 0.0])
        self.std_devs = np.array([0.2, 0.3])

    def validate_point(self, p):
        for p_avoid, d_thresh in self.avoid_pts_with_dist_threshs:
            d = np.linalg.norm(p_avoid - p)
            if d < d_thresh:
                return False

            # Lateral check
            if p[1] < 0 or p[1] > 0.3:
                return False

            # Depth check
            if p[0] > 0.7 or p[0] < 0.3:
                return False
        return True

    def sample_random_position_2d(self):
        while True:
            p = self.std_devs * (np.random.randn(2) + self.center_p)
            if self.validate_point(p):
                return p


class ReachToPlaceOnTower(DfDecider):
    def __init__(self,Travel_Height):
        super().__init__()
        
        db_manager = DatabaseManager()
        # ct = self.context      
       
        # distance = db_manager.query_task_info(table_name="Travel_Op_Parmeters", condition_value=ct.in_gripper.name, column_to_query="Travel_Height", condition_column="object_id")
        self.add_child("approach_grasp", DfApproachGrasp(approach_along_axis=1, direction_length=Travel_Height)) # Defined the derection of approach for pickup and the direction length specifying the distace from which the approach will start
        # self.add_child("approach_grasp", DfApproachGrasp(approach_along_axis=1, direction_length=distance)) # Defined the derection of approach for pickup and the direction length specifying the distace from which the approach will start

    def decide(self):
        ct = self.context
        db_manager=DatabaseManager()
        axis = db_manager.query_task_info(table_name="Travel_Op_Parmeters", condition_value=ct.in_gripper.name, column_to_query="Gripper_Rotation", condition_column="object_id")
        if axis=="y-axis":
            ct.placement_target_eff_T = calc_grasp_for_top_of_tower(ct,axis)
        else:
            ct.placement_target_eff_T = calc_grasp_for_top_of_tower(ct,"z-axis")
        eff_T=ct.robot.arm.get_fk_T()
        if math_util.transforms_are_close(ct.placement_target_eff_T, eff_T, p_thresh=0.01, R_thresh=0.02):
            self.context.reached_target_T = True
            return None
        return DfDecision("approach_grasp", ct.placement_target_eff_T)

    def exit(self):
        # self.context.placement_target_eff_T = None
        self.context.reached_target_T = True

        print("Exiting ReachToPlaceOnTower: reached_target_T set to True")

class ReachToPlaceOnTable(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("approach_grasp", DfApproachGrasp())

    def choose_random_T_on_table(self):
        ct = self.context

        table_point_validator = TablePointValidator(self.context)
        rp2d = table_point_validator.sample_random_position_2d()
        rp = np.array([rp2d[0], rp2d[1], ct.block_height / 2])

        ax = -math_util.normalized(np.array([rp[0], rp[1], 0.0]))
        az = np.array([0.0, 0.0, 1.0])
        ay = np.cross(az, ax)
        T = math_util.pack_Rp(math_util.pack_R(ax, ay, az), rp)

        return calc_grasp_for_block_T(ct, T, -T[:3, 3])

    def enter(self):
        self.context.placement_target_eff_T = self.choose_random_T_on_table()

    def decide(self):
        ct = self.context

        table_point_validator = TablePointValidator(self.context)
        if not table_point_validator.validate_point(ct.placement_target_eff_T[:2, 3]):
            ct.placement_target_eff_T = self.choose_random_T_on_table()

        return DfDecision("approach_grasp", ct.placement_target_eff_T)

    def exit(self):
        # self.context.placement_target_eff_T = None
        self.context.reached_target_T = True




class ReachToPlacementRd(DfRldsNode):
    def __init__(self,Travel_Height):
        super().__init__()
        self.add_child("reach_to_place_on_tower", ReachToPlaceOnTower(Travel_Height))
        self.add_child("reach_to_place_table", ReachToPlaceOnTable())

    def is_runnable(self):
        return self.context.gripper_has_block

    def enter(self):
        self.context.placement_target_eff_T = None

    def decide(self):
        ct = self.context

        if ct.block_tower.current_stack_in_correct_order and ct.block_tower.next_block == ct.in_gripper:
            return DfDecision("reach_to_place_on_tower")
        else:
            return DfDecision("reach_to_place_table")
    def exit(self):
        self.context.reached_target_T = True
        print("Exiting ReachToPlacementRd: reached_target_T set to True")
        ct=self.context
        db_manager = DatabaseManager()
        db_manager.update_table("Travel_Op_Parmeters", "operation_status", True, "object_id", ct.active_block.name)
    # def exit(self):
    #     self.context.placement_target_eff_T = None


def set_top_block_aligned(ct):
    top_block = ct.block_tower.top_block
    if top_block is not None:
        top_block.set_aligned()



class ScrewRd(DfStateMachineDecider, DfRldsNode):
    """A state sequence for performing a screwing motion with the cube."""
    def __init__(self,number_of_rotations=0):  # Default to 1 if not provided
        sequence = [DfSetLockState(set_locked_to=True, decider=self)]
        
        for _ in range(number_of_rotations):
            sequence.append(RotateState(rotation_angle_degrees=90))  # Rotate the cube 90 degrees
            sequence.append(DfTimedDeciderState(DfOpenGripper(), activity_duration=0.5))
            sequence.append(RotateState(rotation_angle_degrees=-90))  # Rotate back to original
            sequence.append(DfTimedDeciderState(DfCloseGripper(), activity_duration=0.5))

        sequence.append(DfSetLockState(set_locked_to=False, decider=self))

        super().__init__(DfStateSequence(sequence))
        self.is_locked = False

    def is_runnable(self):  
        ct = self.context      
        if self.context.gripper_has_block:
            number_of_rotations_db = ct.query_database(table_name="Screw_Op_Parmeters", condition_value=ct.active_block.name, column_to_query="number_of_rotations", condition_column="object_id")
            current_rotation_db = ct.query_database(table_name="Screw_Op_Parmeters", condition_value=ct.active_block.name, column_to_query="current_rotation", condition_column="object_id")

            if number_of_rotations_db > current_rotation_db:
                ct.screw_finished = True
                ct.update_db_data("Screw_Op_Parmeters", "operation_status", True, "object_id", ct.active_block.name)
                ct.update_db_data("Screw_Op_Parmeters", "current_rotation", number_of_rotations_db, "object_id", ct.active_block.name)
                return True
        return False


class DropRd(DfStateMachineDecider, DfRldsNode):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    DfTimedDeciderState(DfOpenGripper(), activity_duration=0.5),
                    LiftState(command_delta_z=0.1, success_delta_z=0.03),
                    DfWriteContextState(lambda ctx: ctx.clear_gripper()),
                    DfWriteContextState(set_top_block_aligned),
                    DfTimedDeciderState(DfCloseGripper(), activity_duration=0.25),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )
        self.is_locked = False

    def is_runnable(self):
        print("goint inside the runnable")
        ct = self.context
        if ct.gripper_has_block:
            print("<dropping block>")
            return True
        return False

    def exit(self):
        ct=self.context
        db_manager = DatabaseManager()
        db_manager.update_table("Drop_Op_Parmeters", "operation_status", True, "object_id", ct.active_block.name)
        self.context.reset_active_block()
        self.context.placement_target_eff_T = None
        self.context.reached_target_T = False  # <--- Reset after dropping 
      
        
        
class PlaceBlockRd(DfStateMachineDecider, DfRldsNode):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    DfTimedDeciderState(DfOpenGripper(), activity_duration=0.5),
                    LiftState(command_delta_z=0.1, success_delta_z=0.03),
                    DfWriteContextState(lambda ctx: ctx.clear_gripper()),
                    DfWriteContextState(set_top_block_aligned),
                    DfTimedDeciderState(DfCloseGripper(), activity_duration=0.25),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )
        self.is_locked = False

    def is_runnable(self):
        ct = self.context
        if ct.gripper_has_block and ct.has_placement_target_eff_T:
            eff_T = ct.robot.arm.get_fk_T()

            thresh_met = math_util.transforms_are_close(
                ct.placement_target_eff_T, eff_T, p_thresh=0.005, R_thresh=0.005
            )

            if thresh_met:
                print("<placing block>")
            return thresh_met

        return False

    def exit(self):
        ct = self.context
        self.context.reset_active_block()
        self.context.placement_target_eff_T = None




def make_pick_rlds():
    rlds = DfRldsDecider()

    open_gripper_rd = OpenGripperRd(dist_thresh_for_open=0.15)
    reach_to_block_rd = ReachToBlockRd()
    choose_block = ChooseNextBlock()
    approach_grasp = DfApproachGrasp()

    reach_to_block_rd.link_to("choose_block", choose_block)
    choose_block.link_to("approach_grasp", approach_grasp)

    rlds.append_rlds_node("reach_to_block", reach_to_block_rd)
    rlds.append_rlds_node("pick_block", PickBlockRd())
    rlds.append_rlds_node("open_gripper", open_gripper_rd)  # Always open the gripper if it's not.

    return rlds

def make_travel_rlds():
    rlds = DfRldsDecider()
    db_manager = DatabaseManager()
    
    object_name=db_manager.query_task_info(table_name="Operation_Sequence", condition_value="travel", column_to_query="object_name", condition_column="sequence_name")

    Travel_height=db_manager.query_task_info(table_name="Travel_Op_Parmeters", condition_value=object_name, column_to_query="Travel_Height", condition_column="object_id")

    nodename=db_manager.query_task_info(table_name="sequences", condition_value="travel", column_to_query="node_name", condition_column="sequence_name")
    print(nodename)
    
    if nodename in globals():  # Check if the name exists in global scope
        node_object = globals()[nodename](Travel_height)  # Instantiate class or call function

    # Append to rlds
    rlds.append_rlds_node("reach_to_placement", node_object)
    # rlds.append_rlds_node("reach_to_placement", ReachToPlacementRd())

    return rlds

# def make_screw_rlds():
#     rlds = DfRldsDecider()
#     db_manager = DatabaseManager()
#     nodename=db_manager.query_task_info(table_name="sequences", condition_value="screw", column_to_query="node_name", condition_column="sequence_name")
#     print(nodename)
#     # if not nodename:
#     #     print("No valid entry for 'screw' sequence in database. Skipping ScrewRd.")
#     #     return rlds  # Return empty rlds, avoiding errors
    
#     if nodename in globals():  # Check if the name exists in global scope
#         node_object = globals()[nodename]()  # Instantiate class or call function
#     rlds.append_rlds_node("screw_rd", node_object)
#     # screw_rd = ScrewRd() 
#     # rlds.append_rlds_node("screw_rd", screw_rd)
#     return rlds
def make_screw_rlds():
    rlds = DfRldsDecider()
    db_manager = DatabaseManager()
    
    object_name = db_manager.query_task_info(
        table_name="Operation_Sequence", condition_value="screw", column_to_query="object_name", condition_column="sequence_name"
    )

    # number_of_rotations = db_manager.query_task_info(
    #     table_name="Screw_Op_Parmeters", condition_value=object_name, column_to_query="number_of_rotations", condition_column="object_id"
    # )
    
    nodename=db_manager.query_task_info(table_name="sequences", condition_value="screw", column_to_query="node_name", condition_column="sequence_name")
    node_object = globals()[nodename](number_of_rotations=0)  # Instantiate class or call function

    # screw_node = ScrewRd(number_of_rotations=number_of_rotations)
    rlds.append_rlds_node("screw_rd", node_object)

    # rlds.append_rlds_node("screw_rd", screw_node)
    return rlds

def make_drop_rlds():
    rlds = DfRldsDecider()
    db_manager = DatabaseManager()
    nodename=db_manager.query_task_info(table_name="sequences", condition_value="drop", column_to_query="node_name", condition_column="sequence_name")
    print(nodename)
    
    if nodename in globals():  # Check if the name exists in global scope
        node_object = globals()[nodename]()  # Instantiate class or call function
    rlds.append_rlds_node("drop_block", node_object)
    # rlds.append_rlds_node("drop_block", DropRd())
    return rlds



class BlockPickAndPlaceDispatch(DfDecider):
    def __init__(self, sequence_operations):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.sequence_operations = sequence_operations  # Ensure this is properly set

        # Mapping of operations to RLD nodes.
        self.task_map = {
            "go_home": GoHome(),
            "pick": make_pick_rlds(),
            "travel": make_travel_rlds(),
            "screw": make_screw_rlds(),
            "drop": make_drop_rlds(), 
            # "slide": SlideState(slide_distance=0.05)  # Adding slide operation
        }

        # Create child nodes dynamically based on sequence_operations
        for operation in sequence_operations:
            operation_name = operation["sequence_name"]  # Extract sequence name
            if operation_name in self.task_map:
                self.add_child(operation_name, self.task_map[operation_name])
            else:
                raise ValueError(f"Task '{operation_name}' is not defined in the task map.")        

    def decide(self):
        ct = self.context

        for operation in self.sequence_operations:
            task_name = operation["sequence_name"]
            operation_id = operation["operation_id"]
            object_name = operation["object_name"]

            if task_name == "go_home" and ct.block_tower.is_complete:
                return DfDecision("go_home")
            if task_name == "pick" and ct.is_gripper_clear and not ct.block_tower.is_complete and ct.next_block_name == object_name:
                return DfDecision("pick")
            if task_name == "travel" and ct.gripper_has_block and not ct.reached_target_T and ct.in_gripper.name == object_name:
                return DfDecision("travel")
            if task_name == "screw" and ct.gripper_has_block and ct.reached_target_T and ct.in_gripper.name == object_name and not self.db_manager.query_task_info(
                table_name="Screw_Op_Parmeters", condition_value=ct.active_block.name, column_to_query="operation_status", condition_column="object_id" ):
                return DfDecision("screw")
            if task_name == "drop" and ct.gripper_has_block and ct.reached_target_T and ct.in_gripper.name == object_name:
                return DfDecision("drop")
            # if task_name == "slide" and ct.gripper_has_block and ct.reached_target_T and ct.in_gripper.name == object_name:
            #     return DfDecision("slide")
        
        return DfDecision("go_home")  # Default decision


def make_decider_network(robot, Drop_off_location=None):
    db_manager = DatabaseManager()
    
    # Retrieve sequences dynamically
    db_manager.cursor.execute("""
        SELECT sequence_id, operation_id, sequence_name, object_name 
        FROM Operation_Sequence
    """)
    rows = db_manager.cursor.fetchall()
    
    # Convert query results to dictionary format
    sequence_operations = [{
        "sequence_id": row[0],
        "operation_id": row[1],
        "sequence_name": row[2],
        "object_name": row[3]
    } for row in rows]
    
    return DfNetwork(
        BlockPickAndPlaceDispatch(sequence_operations), 
        context=BuildTowerContext(robot, tower_position=np.array([0.35, 0.0, 0.068]))
    )


   
class DatabaseManager:
    def __init__(self, db_name="/home/omniverse/.local/share/ov/pkg/isaac_sim-2023.1.1/sequences.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        # self.initialize_database()



    def query_task_info(self, table_name, condition_value, column_to_query, condition_column):
        if column_to_query and condition_column:
            query = f"SELECT {column_to_query} FROM {table_name} WHERE {condition_column} = ?"
            self.cursor.execute(query, (condition_value,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        
        elif column_to_query and not condition_column:
            query = f"SELECT {column_to_query} FROM {table_name} ORDER BY {condition_value} ASC"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        
        else:
            query = "SELECT * FROM tasks WHERE task_name = ?"
            self.cursor.execute(query, (condition_value,))
            row = self.cursor.fetchone()
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, row)) if row else None

    def update_table(self, table_name, column_to_update, new_value, condition_column, condition_value):
        """
        Update a specific column in any table with a new value based on a condition.

        Args:
            table_name (str): The name of the table to update.
            column_to_update (str): The column to update.
            new_value (any): The new value to set for the column.
            condition_column (str): The column to use for the condition.
            condition_value (any): The value to match in the condition column.
        """
        query = f"""
        UPDATE {table_name}
        SET {column_to_update} = ?
        WHERE {condition_column} = ?
        """
        self.cursor.execute(query, (new_value, condition_value))
        self.conn.commit()
        print(f"Updated {column_to_update} in {table_name} where {condition_column} = {condition_value} to {new_value}")

    def check_table_schema(self, table_name):
        """
        Print the schema of a table for debugging purposes.

        Args:
            table_name (str): The name of the table to check.
        """
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        schema = self.cursor.fetchall()
        print(f"Schema of {table_name}:")
        for column in schema:
            print(column)


    def close(self):
        self.conn.close()


# Copyright (c) 2022, NVIDIA  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
