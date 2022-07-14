# Gelsight ROS

This package implements a variety of processing techniques for GelSight sensors in ROS.

As of right now, only the R1.5 is supported. You may be able to make others work by converting your sensor output to support a `cv2.VideoCapture` format.

## Dependencies

- \>= Python 3.8
- [Pybind11 catkin](https://github.com/ipab-slmc/pybind11_catkin)
- [PyTorch](https://pytorch.org/get-started/locally/)

## Topics / Features

> The following can all be enabled / disabled in `config/gelsight.yml`.

Depth reconstruction (`sensor_msgs/PointCloud2`)
- Two methods available: 'functional' and 'model'
- Functional method uses raw coefficients to estimate pixel -> gradient mapping
- Model method uses a trained MLP to infer pixel -> gradient mapping (see [model creation](#model-creation))

Pose detection (`geometry_msgs/PoseStamped`)
- Filters out a portion of the depth map, then performs PCA and returns the principal axis as a pose msg

Marker tracking (`gelsight_ros/GelsightMarkersStamped`)
- Uses OpenCV's adaptive threshold to mask markers
- GelsightMarkers contains a list of pixel coordinates in the raw image

Flow detection (`gelsight_ros/GelsightFlowStamped`)
- Uses Gelsight's marker tracking algorithm to contrust pairwise matching of markers
- GelsightFlow contains the reference and current marker frames with the data having a one-to-one correspondence

## [Model creation](#model-creation)

This package provides a few utilities to collect, label and train on data from the sensor. For more information on this process, see this [paper](https://ieeexplore.ieee.org/abstract/document/9560783).

`scripts/record.py`: Allows you to collect a series of images and store them to a chosen folder.

`scripts/label_data.py`: Allows you to label the found circles. Directions documented at the top of the file.

`scripts/plot_grad.py`: Allow you verify the labeled data by ploting the gradients of a single, labeled image.

`scripts/train.py`: Allows you to train a simple MLP on the labeled data.

## Usage

Before running this package, you have to configure the sensor stream. For R1.5, it's recommend you use the [mjpeg_streamer](https://github.com/jacksonliam/mjpg-streamer) service on the raspberry pi and set `http_stream/url` in `gelsight.yml`.

Modify additional parameters as needed, then launch:

```bash
roslaunch gelsight_ros gelsight.launch
```

## TODO

- Support additional dataset formats (COCO)
- Support additional model types (pix2pix)
- Translate expensive processes to C++

## References

Please check the official [GelSight SDK](https://github.com/gelsightinc/gsrobotics) for more information.
