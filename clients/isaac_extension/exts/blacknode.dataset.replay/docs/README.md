# Blacknode Dataset Replay

Adds **Window > Blacknode Dataset Replay** to Isaac Sim. The window connects
directly to a Blacknode dataset stream and provides joint mapping, home-pose
calibration, direction, limits, and USD drive controls.

Use **Use selected prim** to assign the selected articulation root, then
**Discover Joints** to list articulation DOFs. Each dataset joint can be mapped
to an Isaac DOF, reversed with `+1`/`-1`, limited, and tuned with stiffness,
damping, and maximum force. Joint-angle fields and sliders update the simulated
articulation while calibration mode is active.

Pose the robot to match the physical home and choose **Set Home Pose**. For an
incremental correction, toggle **Calibrate**, enter or slide angle nudges, and
choose **Add to Home**. **Go Home** reapplies the stored pose. The status area
uses green for a healthy stream and red for connection, mapping, or articulation
errors.

Window settings persist through Isaac's settings store. Portable articulation
calibrations can also be saved and loaded as versioned JSON files under
`~/.blacknode/calibrations/isaac/`. The extension connects directly to the
`StreamPublisher` WebSocket; ROS 2 is optional for other workflows and is not
required by this replay window.
