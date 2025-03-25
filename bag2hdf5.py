import rosbag
import h5py
import numpy as np
from pathlib import Path
from bisect import bisect_left

def create_hdf5_directory():
    script_dir = Path(__file__).resolve().parent
    hdf5_dir = script_dir / 'hdf5'
    hdf5_dir.mkdir(exist_ok=True)
    return hdf5_dir

def find_nearest(timestamp_list, target):
    pos = bisect_left(timestamp_list, target)
    if pos == 0:
        return 0
    if pos == len(timestamp_list):
        return len(timestamp_list) - 1
    before = timestamp_list[pos - 1]
    after = timestamp_list[pos]
    return pos if abs(after - target) < abs(before - target) else pos - 1

def convert_bag_to_hdf5(bag_file_path, hdf5_dir, total_files, index):
    bag_name = Path(bag_file_path).stem
    hdf5_file_path = hdf5_dir / f"{bag_name}.h5"

    #print(f"Processing {index}/{total_files}: {bag_file_path} -> {hdf5_file_path}")

    cam1_timestamps = []
    cam1_images = []

    other_images = {
        "cam_left_wrist": [],
        "cam_right_wrist": [],
    }
    other_images_timestamps = {
        "cam_left_wrist": [],
        "cam_right_wrist": [],
    }

    qpos_data = []
    qpos_timestamps = []
    hand_state_data = []
    hand_state_timestamps = []

    action_data = []
    action_timestamps = []
    hand_cmd_data = []
    hand_cmd_timestamps = []

    vlen_dtype = h5py.special_dtype(vlen=np.uint8)

    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            timestamp = t.to_sec()

            if topic == "/cam_1/color/image_raw/compressed":
                cam1_timestamps.append(timestamp)
                cam1_images.append(np.frombuffer(msg.data, dtype=np.uint8))

            elif topic == "/cam_4/color/compressed":
                other_images["cam_left_wrist"].append(np.frombuffer(msg.data, dtype=np.uint8))
                other_images_timestamps["cam_left_wrist"].append(timestamp)

            elif topic == "/cam_3/color/compressed":
                other_images["cam_right_wrist"].append(np.frombuffer(msg.data, dtype=np.uint8))
                other_images_timestamps["cam_right_wrist"].append(timestamp)

            elif topic == "/robot_arm_q_v_tau":
                qpos_data.append(np.array(msg.q, dtype=np.float64))  # 14 значений
                qpos_timestamps.append(timestamp)

            elif topic == "/robot_hand_position":
                left = np.frombuffer(msg.left_hand_position, dtype=np.uint8)[0:1]
                right = np.frombuffer(msg.right_hand_position, dtype=np.uint8)[0:1]
                hand_state_data.append(np.array([left[0], right[0]], dtype=np.float64))
                hand_state_timestamps.append(timestamp)

            elif topic == "/kuavo_arm_traj":
                action_data.append(np.array(msg.position, dtype=np.float64))  # 14 значений
                action_timestamps.append(timestamp)

            elif topic == "/control_robot_hand_position":
                left = np.frombuffer(msg.left_hand_position, dtype=np.uint8)[0:1]
                right = np.frombuffer(msg.right_hand_position, dtype=np.uint8)[0:1]
                hand_cmd_data.append(np.array([left[0], right[0]], dtype=np.float64))
                hand_cmd_timestamps.append(timestamp)

    # Convert all to numpy arrays
    qpos_data = np.array(qpos_data)
    qpos_timestamps = np.array(qpos_timestamps)
    hand_state_data = np.array(hand_state_data)
    hand_state_timestamps = np.array(hand_state_timestamps)

    action_data = np.array(action_data)
    action_timestamps = np.array(action_timestamps)
    hand_cmd_data = np.array(hand_cmd_data)
    hand_cmd_timestamps = np.array(hand_cmd_timestamps)

    for key in other_images:
        other_images_timestamps[key] = np.array(other_images_timestamps[key])

    aligned_qpos = []
    aligned_action = []
    aligned_cam_left = []
    aligned_cam_right = []

    for t in cam1_timestamps:
        i_q = find_nearest(qpos_timestamps, t) if len(qpos_timestamps) > 0 else -1
        i_hand_q = find_nearest(hand_state_timestamps, t) if len(hand_state_timestamps) > 0 else -1
        i_a = find_nearest(action_timestamps, t) if len(action_timestamps) > 0 else -1
        i_hand_a = find_nearest(hand_cmd_timestamps, t) if len(hand_cmd_timestamps) > 0 else -1
        i_l = find_nearest(other_images_timestamps["cam_left_wrist"], t) if len(other_images_timestamps["cam_left_wrist"]) > 0 else -1
        i_r = find_nearest(other_images_timestamps["cam_right_wrist"], t) if len(other_images_timestamps["cam_right_wrist"]) > 0 else -1

        qpos_vector = np.zeros(16)
        if i_q != -1:
            qpos_vector[:14] = qpos_data[i_q]
        if i_hand_q != -1:
            qpos_vector[14:] = hand_state_data[i_hand_q]

        action_vector = np.zeros(16)
        if i_a != -1:
            action_vector[:14] = action_data[i_a]
        if i_hand_a != -1:
            action_vector[14:] = hand_cmd_data[i_hand_a]

        aligned_qpos.append(qpos_vector)
        aligned_action.append(action_vector)
        aligned_cam_left.append(other_images["cam_left_wrist"][i_l] if i_l != -1 else np.zeros(1, dtype=np.uint8))
        aligned_cam_right.append(other_images["cam_right_wrist"][i_r] if i_r != -1 else np.zeros(1, dtype=np.uint8))

    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        observations_grp = hdf5_file.create_group("observations")
        images_grp = observations_grp.create_group("images")

        observations_grp.create_dataset("qpos", data=np.array(aligned_qpos))
        hdf5_file.create_dataset("action", data=np.array(aligned_action))
        images_grp.create_dataset("cam_high", (len(cam1_images),), dtype=vlen_dtype, data=cam1_images)
        images_grp.create_dataset("cam_left_wrist", (len(aligned_cam_left),), dtype=vlen_dtype, data=aligned_cam_left)
        images_grp.create_dataset("cam_right_wrist", (len(aligned_cam_right),), dtype=vlen_dtype, data=aligned_cam_right)

    print(f"Successfully created!: ({index}/{total_files})")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    samples_dir = script_dir / 'samples'
    hdf5_dir = create_hdf5_directory()

    bag_files = list(samples_dir.glob("*.bag"))
    total_files = len(bag_files)

    print(f"Found {total_files} .bag files in {samples_dir}")

    for index, bag_file in enumerate(bag_files, start=1):
        convert_bag_to_hdf5(bag_file, hdf5_dir, total_files, index)
