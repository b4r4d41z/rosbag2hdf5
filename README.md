## .bag → .h5 for ROS1 (noetic) 🐢

This project is a Python script that transforms data from .bag to .h5 format. The script works with ROS1 Noetic.

Final structure of the h5 file:

```
/                        Group
/action                  Dataset {645, 16}
/observations            Group
/observations/images     Group
/observations/images/cam_high Dataset {645}
/observations/images/cam_left_wrist Dataset {645}
/observations/images/cam_right_wrist Dataset {645}
/observations/qpos       Dataset {645, 16}
```

This structure was specifically designed for the [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) project.

## How to use

1. Replace the file in the `samples` folder in the project directory with your own `.bag` files, or change the path in `bag2hdf5.py` on line 147 to the directory containing your files:

```
samples_dir = script_dir / 'samples'
```

2. Then run the script:

```
python3 bag2hdf5.py
```

3. During execution, an `hdf5` folder will be created in the project directory containing your data.

## Additional info:

In the `samples` folder of the project directory, there is example `.bag` file with data collected on the [Kuavo robot](https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/).


**You can easely check this data in Foxglov Studio**

![Kuavo](/rosbag2hdf5/kuavo.png)