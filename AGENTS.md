# blacknode-dataset Agent Instructions

This is an independent Blacknode extension-package repository. Check and commit
its Git state separately from a containing Blacknode checkout.

## Scope

Own Blacknode-native dataset manifests, episode journaling, synchronized sample
capture, validation, summaries, LeRobot-format export, and explicit Hugging
Face upload. Do not own robot control, ROS transport, or camera acquisition.

## Rules

- Recording must never command motion; it only consumes stream handles.
- Keep capture independent from LeRobot. Compatibility belongs in exporters and
  optional interoperability tests, never the recorder runtime.
- Journal incomplete episodes before conversion. Save atomically and retain
  recoverable data after interruption or encoding failure.
- Preserve source timestamps, stable joint names, units, robot identity,
  calibration references, action targets, and freshness counters.
- Refuse stale or malformed state/action samples. Do not silently resize vectors
  or reorder joints between episodes.
- Network upload is explicit. Never persist or log Hugging Face credentials.
- Tests must use synthetic robot samples and generated camera frames; routine
  validation must not require hardware, ROS, or network access.

## Verification

From the Blacknode root:

```powershell
python -m pytest packages/blacknode-dataset/tests
Get-ChildItem packages\blacknode-dataset\templates\*.json | ForEach-Object { blacknode validate $_.FullName }
```
