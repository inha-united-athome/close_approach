<div align="center">
  <h1> Close Approach For Manipulation</h1>
  <img src="https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 24.04">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/ROS2-jazzy-22314E?logo=ros&logoColor=white" alt="ROS2 jazzy">
</div>

<div align="center">
  <img src="assets/example.jpg" width="45%" alt="image 1">
  <img src="assets/example.jpg" width="45%" alt="image 2">
</div>

# What is this?

A global navigation stack is usually designed to move a robot safely through the environment, not to guarantee a manipulation-ready pose in front of an object.
For mobile manipulation, however, the final few centimeters matter.
The robot must approach the target with the correct distance, lateral offset, and heading so that the arm can reach the object without unnecessary motion or unstable configurations.

This becomes more difficult with a large mobile manipulator.
Because the robot footprint is large, Nav2 alone often cannot place the base close enough to the object.
Even when the navigation goal is reached, the remaining pose error can be too large for manipulation.
The robot may stop slightly too far away, face the target at an unsuitable angle, or fail to align the arm workspace with the object.

`close_approach` is designed for this final local approach phase.
It uses onboard depth and point cloud data to estimate the target geometry around the robot, filters the local region of interest, removes irrelevant planes, extracts geometric cues, estimates the remaining alignment error, and controls the base until the robot reaches a manipulation-ready pose.

In short, this package bridges the gap between global navigation and manipulation.
Nav2 brings the robot near the target area; `close_approach` performs the precise, perception-guided alignment required before manipulation can begin.

# Evaluation
`TODO`

# Runtime Architecture

The current launch path is `launch/approach.launch.py`. It runs the local
approach pipeline as separate ROS 2 nodes:

- `pc_detector_node`: point-cloud based target geometry, longitudinal `x_error`,
  debug clouds, OBB marker, and target-edge marker
- `edge_detector_node`: image-edge based yaw error from compressed camera images
- `approach_manager_node`: `/approach` action server and approach state machine
- `approach_controller_node`: PID conversion from `/approach/control_error` to
  `/cmd_vel`
- `approach_debug_logger_node`: per-action CSV, debug image, and PCD capture

The legacy monolithic `approach_node` is still present in the package, but it is
not used by `approach.launch.py`.

```bash
ros2 launch close_approach approach.launch.py
```

The action interface remains:

```bash
ros2 action send_goal /approach inha_interfaces/action/Approach "{goal_distance: 0.3}" --feedback
```

## Error Flow

Point-cloud distance error:

```text
/camera/camera_head/depth/color/points
  -> pc_detector_node
  -> /approach/pc_error
  -> approach_manager_node
  -> /approach/control_error.x_error
  -> approach_controller_node
  -> /cmd_vel.linear.x
```

Yaw error:

```text
/camera/camera_head/color/image_raw/compressed
  -> edge_detector_node
  -> /approach/edge_error.theta_error
  -> approach_manager_node
  -> /approach/control_error.theta_error
  -> approach_controller_node
  -> /cmd_vel.angular.z
```

`approach_head_control_node`, when started separately, listens to
`/approach/_action/feedback`. It is compatible with the manager-based action
server because the `/approach` action name, type, and feedback fields are kept.

## Point-Cloud ROI

`pc_detector_node` uses a trapezoid-like spatial ROI in `base_nav`.
The near side has a smaller lateral half-width and the far side has a larger
lateral half-width:

```yaml
roi_x_min: 0.1
roi_x_max: 2.0
roi_y_abs_near: 0.3
roi_y_abs_max: 0.8
roi_z_max: 1.5
```

The allowed `|y|` grows linearly from `roi_y_abs_near` to `roi_y_abs_max` as
`x` increases. This reduces near-field robot/self clutter while keeping a wider
search region for farther targets.

The published `/approach/debug_cloud` is after downsampling, outlier removal,
ground removal, spatial ROI, and optional LiDAR fusion. The published
`/approach/filtered_cloud` is after clustering, projection to 2D, and front
slicing, and is closer to the cloud used for OBB and `x_error` estimation.

## Edge Detector Control

`edge_detector_node` is disabled by default and avoids image decoding while
disabled. `approach_manager_node` enables it at action start and disables it
when the action stops through:

```text
/approach/edge_detector/set_enable
```

Manual control:

```bash
ros2 service call /approach/edge_detector/set_enable inha_interfaces/srv/SetEnable "{enable: true}"
ros2 service call /approach/edge_detector/set_enable inha_interfaces/srv/SetEnable "{enable: false}"
```

The edge yaw gate is parameterized:

```yaml
max_abs_yaw_deg: 30.0
```

Only Hough line angles with `abs(angle) < max_abs_yaw_deg` are used for yaw
estimation. The debug image topic is:

```text
/approach/edge_debug/compressed
```

# Logs
## Approach debug logs

`approach_debug_logger_node` writes one timestamped directory per action under
`/home/thor/inha_log/module/close_approach` by default. The output path and
topics are configured in `config/approach_debug_logger.yaml`.

```text
/home/thor/inha_log/module/close_approach/approach_YYYYMMDD_HHMMSS_mmm/
├── samples.csv
├── sample_000000.jpg
├── sample_000000.pcd
├── sample_000001.jpg
└── sample_000001.pcd
```

- `samples.csv`: state, edge yaw, point-cloud error, control error, command
  velocity, and paired image/PCD filenames
- `sample_*.jpg`: latest compressed debug image from
  `/approach/edge_debug/compressed`
- `sample_*.pcd`: latest point cloud from the configured logger cloud topic,
  `/approach/debug_cloud` by default

To log the final cloud used closer to `x_error` estimation, set:

```yaml
cloud_topic: /approach/filtered_cloud
```

## Retreat logs

`retreat_node` writes one timestamped directory per action under
`/home/thor/inha_logs/module/close_approach/retreat` by default. Logging can be
configured with the `log_enabled` and `log_dir` parameters.

- `trail.csv`: reverse trail snapshot used by the retreat controller
- `trace.csv`: per-control-cycle pose, distance, tracking error, and command data
- `summary.csv`: action result, distance overshoot, and aggregate tracking errors
