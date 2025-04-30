import rosbag
import h5py
import numpy as np
import json
from pathlib import Path
from bisect import bisect_left
import cv2

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
    return data['instruction']

def convert_bag_to_hdf5(bag_file_path, output_dir, instruction):
    bag_name = Path(bag_file_path).stem
    hdf5_file_path = output_dir / f"{bag_name}.h5"

    cam1_timestamps, cam1_images = [], []
    cam_left_images, cam_left_timestamps = [], []
    cam_right_images, cam_right_timestamps = [], []

    qpos, qvel, effort, qpos_timestamps = [], [], [], []
    action, action_timestamps = [], []
    hand_state, hand_state_timestamps = [], []
    hand_cmd, hand_cmd_timestamps = [], []

    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            timestamp = t.to_sec()

            if topic == "/cam_1/color/image_raw/compressed":
                cam1_timestamps.append(timestamp)
                img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cam1_images.append(img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8))

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

            elif topic == "/kuavo_arm_traj":
                action.append(np.array(msg.position, dtype=np.float64))
                action_timestamps.append(timestamp)

            elif topic == "/control_robot_hand_position":
                hand_cmd.append([msg.left_hand_position[0], msg.right_hand_position[0]])
                hand_cmd_timestamps.append(timestamp)

    aligned_action, aligned_qpos = [], []
    aligned_qvel, aligned_effort = [], []
    aligned_cam_left, aligned_cam_right = [], []

    for t in cam1_timestamps:
        idx_q = find_nearest(qpos_timestamps, t)
        idx_action = find_nearest(action_timestamps, t)
        idx_hand_state = find_nearest(hand_state_timestamps, t)
        idx_hand_cmd = find_nearest(hand_cmd_timestamps, t)
        idx_left_cam = find_nearest(cam_left_timestamps, t)
        idx_right_cam = find_nearest(cam_right_timestamps, t)

        full_qpos = np.concatenate([qpos[idx_q], hand_state[idx_hand_state]])
        full_action = np.concatenate([action[idx_action], hand_cmd[idx_hand_cmd]])

        aligned_qpos.append(full_qpos)
        aligned_action.append(full_action)
        aligned_qvel.append(qvel[idx_q])
        aligned_effort.append(effort[idx_q])

        aligned_cam_left.append(cam_left_images[idx_left_cam])
        aligned_cam_right.append(cam_right_images[idx_right_cam])

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

        # Depth placeholders
        depth_grp.create_dataset("cam_high", data=np.zeros((len(cam1_images),)))
        depth_grp.create_dataset("cam_left_wrist", data=np.zeros((len(cam1_images),)))
        depth_grp.create_dataset("cam_right_wrist", data=np.zeros((len(cam1_images),)))

    print(f"Successfully created: {hdf5_file_path.name}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'dataset'  
    json_path = output_dir / 'instruction.json'
    instruction = load_instruction(json_path)

    bag_files = list((script_dir / 'your_ros_bag_fiels').glob("*.bag"))

    print(f"Found {len(bag_files)} .bag files for processing.")

    for bag_file in bag_files:
        convert_bag_to_hdf5(bag_file, output_dir, instruction)
