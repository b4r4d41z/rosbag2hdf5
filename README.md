 
## .bag → .h5 for ROS1 (noetic) 🐢
This project is a Python script that transforms data from .bag to .h5 format. The script works with ROS1 Noetic.
---
### UPD:

- 5.9.2025 - /qpos have simular structure as /action. 
- 4.30.2025 - /action dataset now contains of 6D representation and RPY data of EEF. 
---

Final structure of the h5 file:

```
/                        Group
/action                  Dataset {541, 20}
/base_action             Dataset {541, 2}
/instruction             Dataset {SCALAR}
/observations            Group
/observations/effort     Dataset {541, 14}
/observations/images     Group
/observations/images/cam_high Dataset {541, 480, 640, 3}
/observations/images/cam_left_wrist Dataset {541, 480, 640, 3}
/observations/images/cam_right_wrist Dataset {541, 480, 640, 3}
/observations/images_depth Group
/observations/images_depth/cam_high Dataset {541}
/observations/images_depth/cam_left_wrist Dataset {541}
/observations/images_depth/cam_right_wrist Dataset {541}
/observations/qpos       Dataset {541, 20}
/observations/qvel       Dataset {541, 14}
```

Where `/action` contains of `(T; r6d_l, rpy_l, gripper_left, r6d_r, rpy_r, gripper_right)`. 
Gripper data was taken from `/control_robot_hand_position topic` 

This structure was specifically designed for the [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) project.

## How to use

1. Replace the file in the `your_ros_bag_fiels` folder in the project directory with your own `.bag` files, or change the path in `bag2hdf5.py` on line 127 to the directory containing your files:

```
    bag_files = list((script_dir / 'your_ros_bag_fiels').glob("*.bag"))
```

2. Create your own task description — an instruction — in `.json` format similar to the one provided in the `dataset` folder.

3. Then run the script:

```
python3 bag2hdf5.py
```

4. During execution, your data will be created in the in `hyour_ros_bag_fielsdf5` folder.

## Additional info:

In the `samples` folder of the project directory, there is example `.bag` file with data collected on the [Kuavo robot](https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/).


**You can easely check this data in Foxglov Studio**

![Kuavo](/rosbag2hdf5/kuavo.png)
