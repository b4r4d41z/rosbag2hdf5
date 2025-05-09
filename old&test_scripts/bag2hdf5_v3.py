import rosbag
import h5py
import numpy as np
import json
from pathlib import Path
from bisect import bisect_left
import cv2
import pybullet as p
import pybullet_data

# Show 1st msg of transformed .bag
SHOW_FIRST_SAMPLE = True

def find_nearest(timestamp_list, target):
    pos = bisect_left(timestamp_list, target)
    if pos == 0:
        return 0
    if pos == len(timestamp_list):
        return len(timestamp_list) - 1
    before = timestamp_list[pos - 1]
    after = timestamp_list[pos]
    return pos if abs(after - target) < abs(before - target) else pos - 1

def load_instruction(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data.get('instruction', '')

def get_joint_info(robot_id):
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode('utf-8')
        joint_name_to_index[name] = i
    return joint_name_to_index

def get_arm_joint_names():
    left_arm_joints = [
        'l_arm_pitch', 'l_arm_roll', 'l_arm_yaw',
        'l_forearm_pitch', 'l_hand_yaw', 'l_hand_pitch', 'l_hand_roll'
    ]
    right_arm_joints = [
        'r_arm_pitch', 'r_arm_roll', 'r_arm_yaw',
        'r_forearm_pitch', 'r_hand_yaw', 'r_hand_pitch', 'r_hand_roll'
    ]
    return left_arm_joints, right_arm_joints

def compute_fk(joint_angles, robot_id, joint_name_to_index, left_arm_joints, right_arm_joints):
    for i, name in enumerate(left_arm_joints):
        if name in joint_name_to_index:
            p.resetJointState(robot_id, joint_name_to_index[name], joint_angles[i])
    for i, name in enumerate(right_arm_joints):
        if name in joint_name_to_index:
            p.resetJointState(robot_id, joint_name_to_index[name], joint_angles[i + 7])

    l_idx = joint_name_to_index.get('l_hand_roll')
    r_idx = joint_name_to_index.get('r_hand_roll')
    l_state = p.getLinkState(robot_id, l_idx, computeForwardKinematics=True)
    r_state = p.getLinkState(robot_id, r_idx, computeForwardKinematics=True)

    R_l = np.array(p.getMatrixFromQuaternion(l_state[5])).reshape(3, 3)
    R_r = np.array(p.getMatrixFromQuaternion(r_state[5])).reshape(3, 3)

    r6d_l = np.concatenate([R_l[:, 0], R_l[:, 1]])
    r6d_r = np.concatenate([R_r[:, 0], R_r[:, 1]])
    rpy_l = np.array(p.getEulerFromQuaternion(l_state[5]))
    rpy_r = np.array(p.getEulerFromQuaternion(r_state[5]))
    return r6d_l, rpy_l, r6d_r, rpy_r

def convert_bag_to_hdf5(bag_file_path, output_dir, instruction):
    bag_name = Path(bag_file_path).stem
    hdf5_file_path = output_dir / f"{bag_name}.h5"
    print(f"Processing bag: {bag_name}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = Path("/home/ivan/biped_s4/urdf/biped_v4_arm.urdf")
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)

    joint_name_to_index = get_joint_info(robot_id)
    left_arm_joints, right_arm_joints = get_arm_joint_names()

    cam1_timestamps, cam1_images = [], []
    cam_left_images, cam_left_timestamps = [], []
    cam_right_images, cam_right_timestamps = [], []
    qpos, qvel, effort, qpos_timestamps = [], [], [], []
    hand_state, hand_state_timestamps = [], []
    hand_cmd, hand_cmd_timestamps = [], []
    arm_traj_positions, arm_traj_timestamps = [], []

    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            timestamp = t.to_sec()
            if topic == "/cam_1/color/image_raw/compressed":
                img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cam1_images.append(img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8))
                cam1_timestamps.append(timestamp)
            elif topic == "/cam_4/color/compressed":
                img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cam_left_images.append(img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8))
                cam_left_timestamps.append(timestamp)
            elif topic == "/cam_3/color/compressed":
                img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cam_right_images.append(img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8))
                cam_right_timestamps.append(timestamp)
            elif topic == "/robot_arm_q_v_tau":
                qpos.append(np.array(msg.q, dtype=np.float64))
                qvel.append(np.array(msg.v, dtype=np.float64))
                effort.append(np.array(msg.vd, dtype=np.float64))
                qpos_timestamps.append(timestamp)
            elif topic == "/robot_hand_position":
                hand_state.append([msg.left_hand_position[0], msg.right_hand_position[0]])
                hand_state_timestamps.append(timestamp)
            elif topic == "/control_robot_hand_position":
                hand_cmd.append([msg.left_hand_position[0], msg.right_hand_position[0]])
                hand_cmd_timestamps.append(timestamp)
            elif topic == "/kuavo_arm_traj":
                arm_traj_positions.append(np.deg2rad(np.array(msg.position, dtype=np.float64)))
                arm_traj_timestamps.append(timestamp)

    aligned_qpos, aligned_qvel, aligned_effort = [], [], []
    aligned_cam_left, aligned_cam_right = [], []
    aligned_action = []

    for t in cam1_timestamps:
        idx_q = find_nearest(qpos_timestamps, t)
        idx_hand_state = find_nearest(hand_state_timestamps, t)
        idx_hand_cmd = find_nearest(hand_cmd_timestamps, t)
        idx_left_cam = find_nearest(cam_left_timestamps, t)
        idx_right_cam = find_nearest(cam_right_timestamps, t)
        idx_arm_traj = find_nearest(arm_traj_timestamps, t)

        full_qpos = np.concatenate([qpos[idx_q], hand_state[idx_hand_state]])
        aligned_qpos.append(full_qpos)
        aligned_qvel.append(qvel[idx_q])
        aligned_effort.append(effort[idx_q])
        aligned_cam_left.append(cam_left_images[idx_left_cam])
        aligned_cam_right.append(cam_right_images[idx_right_cam])

        r6d_l, rpy_l, r6d_r, rpy_r = compute_fk(
            arm_traj_positions[idx_arm_traj], robot_id, joint_name_to_index,
            left_arm_joints, right_arm_joints
        )
        gripper_left = hand_cmd[idx_hand_cmd][0]
        gripper_right = hand_cmd[idx_hand_cmd][1]
        action_vector = np.concatenate([r6d_l, rpy_l, [gripper_left], r6d_r, rpy_r, [gripper_right]])
        aligned_action.append(action_vector)

    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        observations_grp = hdf5_file.create_group("observations")
        images_grp = observations_grp.create_group("images")
        depth_grp = observations_grp.create_group("images_depth")

        hdf5_file.create_dataset("instruction", data=instruction)
        hdf5_file.create_dataset("base_action", data=np.zeros((len(cam1_images), 2)))
        hdf5_file.create_dataset("action", data=np.array(aligned_action))

        observations_grp.create_dataset("qpos", data=np.array(aligned_qpos))
        observations_grp.create_dataset("qvel", data=np.array(aligned_qvel))
        observations_grp.create_dataset("effort", data=np.array(aligned_effort))

        images_grp.create_dataset("cam_high", data=np.array(cam1_images))
        images_grp.create_dataset("cam_left_wrist", data=np.array(aligned_cam_left))
        images_grp.create_dataset("cam_right_wrist", data=np.array(aligned_cam_right))

        depth_grp.create_dataset("cam_high", data=np.zeros((len(cam1_images),)))
        depth_grp.create_dataset("cam_left_wrist", data=np.zeros((len(cam1_images),)))
        depth_grp.create_dataset("cam_right_wrist", data=np.zeros((len(cam1_images),)))

    print(f"Successfully created: {hdf5_file_path.name}")

    p.disconnect()
    return hdf5_file_path

def show_first_sample(hdf5_file_path):
    print(f"\n[INFO] Showing first sample from: {hdf5_file_path.name}")
    with h5py.File(hdf5_file_path, 'r') as f:
        if "action" in f:
            print(f"\n/action (shape {f['action'].shape}):")
            print(f['action'][0])
        if "base_action" in f:
            print(f"\n/base_action (shape {f['base_action'].shape}):")
            print(f['base_action'][0])
        if "instruction" in f:
            print("\n/instruction:")
            print(f['instruction'][()])
        if "observations/qpos" in f:
            print(f"\n/observations/qpos (shape {f['observations/qpos'].shape}):")
            print(f['observations/qpos'][0])
        if "observations/qvel" in f:
            print(f"\n/observations/qvel (shape {f['observations/qvel'].shape}):")
            print(f['observations/qvel'][0])
        if "observations/effort" in f:
            print(f"\n/observations/effort (shape {f['observations/effort'].shape}):")
            print(f['observations/effort'][0])
        if "observations/images/cam_high" in f:
            print(f"\n/observations/images/cam_high (shape {f['observations/images/cam_high'].shape}):")
            print(f['observations/images/cam_high'][0, :2, :2, :])
        if "observations/images/cam_left_wrist" in f:
            print(f"\n/observations/images/cam_left_wrist (shape {f['observations/images/cam_left_wrist'].shape}):")
            print(f['observations/images/cam_left_wrist'][0, :2, :2, :])
        if "observations/images/cam_right_wrist" in f:
            print(f"\n/observations/images/cam_right_wrist (shape {f['observations/images/cam_right_wrist'].shape}):")
            print(f['observations/images/cam_right_wrist'][0, :2, :2, :])
        if "observations/images_depth/cam_high" in f:
            print(f"\n/observations/images_depth/cam_high (shape {f['observations/images_depth/cam_high'].shape}):")
            print(f['observations/images_depth/cam_high'][0])
        if "observations/images_depth/cam_left_wrist" in f:
            print(f"\n/observations/images_depth/cam_left_wrist (shape {f['observations/images_depth/cam_left_wrist'].shape}):")
            print(f['observations/images_depth/cam_left_wrist'][0])
        if "observations/images_depth/cam_right_wrist" in f:
            print(f"\n/observations/images_depth/cam_right_wrist (shape {f['observations/images_depth/cam_right_wrist'].shape}):")
            print(f['observations/images_depth/cam_right_wrist'][0])

def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'dataset'
    json_path = output_dir / 'instruction.json'
    instruction = load_instruction(json_path)

    bag_dir = script_dir / 'your_ros_bag_files'
    bag_files = sorted(bag_dir.glob("*.bag"))
    print(f"Found {len(bag_files)} .bag files for processing.")

    for i, bag_file in enumerate(bag_files):
        hdf5_path = convert_bag_to_hdf5(bag_file, output_dir, instruction)
        if i == 0 and SHOW_FIRST_SAMPLE:
            show_first_sample(hdf5_path)

if __name__ == "__main__":
    main()
