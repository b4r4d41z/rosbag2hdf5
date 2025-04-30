import os
import numpy as np
import h5py
import rosbag
import pybullet as p
import pybullet_data

def build_model(urdf_path):
    """
    Load the URDF model and initialize PyBullet.
    """
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(os.path.abspath("../biped_s4/meshes"))
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id

def get_joint_info(robot_id):
    """
    Retrieve joint indices and names for the robot.
    """
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    return joint_name_to_index

def get_arm_joint_names():
    """
    Define the names of the 7 joints for each arm, excluding finger joints.
    """
    left_arm_joints = [
        'l_arm_pitch', 'l_arm_roll', 'l_arm_yaw',
        'l_forearm_pitch', 'l_hand_yaw', 'l_hand_pitch', 'l_hand_roll'
    ]
    right_arm_joints = [
        'r_arm_pitch', 'r_arm_roll', 'r_arm_yaw',
        'r_forearm_pitch', 'r_hand_yaw', 'r_hand_pitch', 'r_hand_roll'
    ]
    return left_arm_joints, right_arm_joints

def compute_fk(joint_angles_14, robot_id, joint_name_to_index, left_arm_joints, right_arm_joints):
    """
    Compute the 6D and RPY orientations for both end-effectors using PyBullet.
    """
    for i, name in enumerate(left_arm_joints):
        idx = joint_name_to_index.get(name)
        if idx is not None:
            p.resetJointState(robot_id, idx, joint_angles_14[i])
    for i, name in enumerate(right_arm_joints):
        idx = joint_name_to_index.get(name)
        if idx is not None:
            p.resetJointState(robot_id, idx, joint_angles_14[i + 7])

    l_idx = joint_name_to_index.get('l_hand_roll')
    r_idx = joint_name_to_index.get('r_hand_roll')

    l_state = p.getLinkState(robot_id, l_idx, computeForwardKinematics=True)
    r_state = p.getLinkState(robot_id, r_idx, computeForwardKinematics=True)

    R_l = np.array(p.getMatrixFromQuaternion(l_state[5])).reshape(3, 3)
    R_r = np.array(p.getMatrixFromQuaternion(r_state[5])).reshape(3, 3)

    r6d_l = np.concatenate([R_l[:, 0], R_l[:, 1]])
    rpy_l = np.array(p.getEulerFromQuaternion(l_state[5]))

    r6d_r = np.concatenate([R_r[:, 0], R_r[:, 1]])
    rpy_r = np.array(p.getEulerFromQuaternion(r_state[5]))

    return r6d_l, rpy_l, r6d_r, rpy_r

def debug_print_example_fk(q_sample, robot_id, joint_name_to_index, left_arm_joints, right_arm_joints):
    """
    Print detailed FK result for a single joint vector.
    """
    print("\n===== DEBUG =====")
    print("[INFO] Using topic: /robot_arm_q_v_tau")
    print("[INFO] Extracted 14 joint angles:")
    print(np.round(q_sample, 4))

    # Compute FK
    for i, name in enumerate(left_arm_joints):
        idx = joint_name_to_index.get(name)
        if idx is not None:
            p.resetJointState(robot_id, idx, q_sample[i])
    for i, name in enumerate(right_arm_joints):
        idx = joint_name_to_index.get(name)
        if idx is not None:
            p.resetJointState(robot_id, idx, q_sample[i + 7])

    l_idx = joint_name_to_index['l_hand_roll']
    r_idx = joint_name_to_index['r_hand_roll']

    l_state = p.getLinkState(robot_id, l_idx, computeForwardKinematics=True)
    r_state = p.getLinkState(robot_id, r_idx, computeForwardKinematics=True)

    R_l = np.array(p.getMatrixFromQuaternion(l_state[5])).reshape(3, 3)
    R_r = np.array(p.getMatrixFromQuaternion(r_state[5])).reshape(3, 3)

    r6d_l = np.concatenate([R_l[:, 0], R_l[:, 1]])
    r6d_r = np.concatenate([R_r[:, 0], R_r[:, 1]])
    rpy_l = np.array(p.getEulerFromQuaternion(l_state[5]))
    rpy_r = np.array(p.getEulerFromQuaternion(r_state[5]))

    print("\n[LEFT] Rotation matrix:")
    print(np.round(R_l, 4))
    print("[LEFT] 6D orientation:", np.round(r6d_l, 4))
    print("[LEFT] RPY:", np.round(rpy_l, 4))

    print("\n[RIGHT] Rotation matrix:")
    print(np.round(R_r, 4))
    print("[RIGHT] 6D orientation:", np.round(r6d_r, 4))
    print("[RIGHT] RPY:", np.round(rpy_r, 4))
    print("===== DEBUG =====\n")

def process_bag_file(bag_file_path, output_hdf5_path, urdf_path, topic_name):
    """
    Process a single .bag file: extract joint angles, compute FK, and save to HDF5.
    """
    robot_id = build_model(urdf_path)
    joint_name_to_index = get_joint_info(robot_id)
    left_arm_joints, right_arm_joints = get_arm_joint_names()

    joint_angles_list = []
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            joint_angles_list.append(msg.q)

    if not joint_angles_list:
        print(f"No messages found in topic {topic_name}")
        return

    q_data = np.array(joint_angles_list)
    T = q_data.shape[0]

    # DEBUG print (one time)
    debug_print_example_fk(q_data[0], robot_id, joint_name_to_index, left_arm_joints, right_arm_joints)

    eef_pose_dataset = np.zeros((T, 18))
    for t in range(T):
        r6d_l, rpy_l, r6d_r, rpy_r = compute_fk(q_data[t], robot_id, joint_name_to_index, left_arm_joints, right_arm_joints)
        eef_pose_dataset[t] = np.concatenate([r6d_l, rpy_l, r6d_r, rpy_r])

    with h5py.File(output_hdf5_path, 'w') as f:
        f.create_dataset('eef_pose', data=eef_pose_dataset)

    p.disconnect()

def main():
    input_dir = "your_ros_bag_fiels"
    output_dir = "dataset"
    urdf_path = "../biped_s4/urdf/biped_v4_arm.urdf"
    topic_name = "/robot_arm_q_v_tau"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".bag"):
            bag_file_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".hdf5"
            output_hdf5_path = os.path.join(output_dir, output_filename)
            print(f"Processing {filename}...")
            process_bag_file(bag_file_path, output_hdf5_path, urdf_path, topic_name)
            print(f"Saved FK results to {output_hdf5_path}")

if __name__ == "__main__":
    main()
