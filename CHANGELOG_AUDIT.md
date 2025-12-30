# Pipeline Audit Change Log

## 2025-12-29

- utils/report_generator.py:1-292
  - Rebuilt the report generator into a single, consistent implementation.
  - Removed duplicated blocks and undefined variables, and added safe handling for missing calibration data.
  - Ensures per-subject reports and cross-subject summary generate without runtime errors.
