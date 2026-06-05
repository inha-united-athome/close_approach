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

# Logs
## Close approach logs

When the `debug` parameter is enabled, `approach_node` writes one timestamped
directory per action under `/home/thor/inha_logs/module/close_approach`.
Point cloud snapshots are rate-limited per processing stage by the
`debug_save_period_sec` parameter.

```text
/home/thor/inha_logs/module/close_approach/action_YYYYMMDD_HHMMSS_mmm/
├── measure_YYYYMMDD_HHMMSS_mmm.txt
├── YYYYMMDD_HHMMSS_mmm_XXXXXX_roi_filtered.pcd
├── YYYYMMDD_HHMMSS_mmm_XXXXXX_clustered.pcd
└── YYYYMMDD_HHMMSS_mmm_XXXXXX_final.pcd
```

- `measure_*.txt`: per-cycle sensor latency, processing time, total delay,
  control interval, SE(2) error, velocity command, and processing-stage timing
- `*_roi_filtered.pcd`: point cloud after ROI filtering and sensor fusion
- `*_clustered.pcd`: point cloud after target cluster selection
- `*_final.pcd`: point cloud after projection and front slicing

## Retreat logs

`retreat_node` writes one timestamped directory per action under
`/home/thor/inha_logs/module/close_approach/retreat` by default. Logging can be
configured with the `log_enabled` and `log_dir` parameters.

- `trail.csv`: reverse trail snapshot used by the retreat controller
- `trace.csv`: per-control-cycle pose, distance, tracking error, and command data
- `summary.csv`: action result, distance overshoot, and aggregate tracking errors
