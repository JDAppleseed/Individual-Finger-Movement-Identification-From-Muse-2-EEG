# AGENTS.md

This repo uses subject IDs like "8-M16". For future requests, use this note to
update defaults consistently and safely.

## What changed for the 8-M16 update
- Set the default subject ID strings from "1-F35" to "8-M16" in these files:
  - 1_stream_and_record.py
  - 1b_extract_windows.py
  - 2_train_model.py
  - 4_generate_reports.py
  - 5_review_events.py
  - 5_validate_events.py
  - app/config_model.py
- Updated the default demographic fields to match "8-M16":
  - 1_stream_and_record.py: set GENDER = "M", AGE = 16

## Paths to check next time
Use these paths when updating the subject default again.

- 1_stream_and_record.py
  - SUBJECT_ID_OVERRIDE: update to the new subject ID.
  - GENDER / AGE: keep in sync with the new subject ID.
- 1b_extract_windows.py
  - DEFAULT_SUBJECT_ID and doc examples in comments.
- 2_train_model.py
  - argparse default for --subject-id.
- 4_generate_reports.py
  - argparse default for --subject-id.
- 5_review_events.py
  - argparse default for --subject-id.
- 5_validate_events.py
  - argparse default for --subject-id.
- app/config_model.py
  - default_step1b_settings() and default_train_settings() subject_id.

## How to update for a new subject
1) Replace all default subject IDs (currently "8-M16") with the new subject ID
   in the paths listed above.
2) If the subject ID encodes demographics (e.g., "F35"), update GENDER and AGE
   in 1_stream_and_record.py to match.
3) Leave utils/subject_registry.json unchanged unless you explicitly want to
   reset or override auto-increment behavior.
