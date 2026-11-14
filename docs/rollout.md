# Cohort roll-out

A workflow distilled from one chat can be rolled out over a cohort: the run
editor (or `medmcp replay <name> --batch <manifest>`) reads a manifest written by
a stack, binds the workflow's inputs per subject or subject-session,
pre-flights every item, and runs the steps without a model in between. Each item
writes its results to its own folder under the output folder
(`results/<workflow>/<subject>[/<session>]` unless another is chosen).

## Manifests

These formats are the core's; any stack can write them. The core names no
stack and no tool. medmcp-cohort is the stack that writes them today
(`plan_batch`, `select_series`, `review_selection`, `confirm_selection`).

| Manifest | One row per | Inputs bound by |
| --- | --- | --- |
| Batch plan (CSV with a `status` column) | subject (or subject-session) | the plan's file columns (matched to the recorded example, or chosen in the editor) |
| Selection (CSV with `subject_id` and `path`; optional `session_id`, `collection`, `sequence`, `contrast`, `plane`) | selected series | `path`; one input only (see below) |
| Confirmed selection (`<name>_confirmed.csv`: a selection plus `role`) with its record `<name>_confirmed.json` | role per subject-session | the `role` column |

Only `ok` rows of a plan run; flagged rows are listed and left out.

The record of a confirmed selection is a JSON object:

- `medmcp_selection_confirmation`: `1` (required; the format version);
- `confirmed_csv`: `{path, sha256}` of the confirmed CSV (required);
- optional `confirmed_at`, `note`, `rule_version`, `role_labels` (the labels of
  the `role` column), `overrides` and `excluded` (lists), `counts`
  (`subject_sessions_confirmed`, `series_confirmed`), and `manifest`, `index`,
  `criteria` as `{path, sha256}` references to what was confirmed from.

Anything else in it belongs to the stack that wrote it; the run keeps the whole
record.

## Selections: confirm before a multi-input roll-out

A plain selection can hold two series of one role in a visit (a repeat, a
reformat, a second plane). A tool that reads one scan is not affected: the
workflow runs once per selected series. A workflow that reads several scans per
subject-session (a T1, a T1 with contrast, a FLAIR) needs exactly one series per
role, and which one is decided with the person, not by the roll-out:

1. A selection tool picks the candidate series.
2. A review tool shows one series per subject, session and role, with the
   alternatives, ties and missing roles.
3. A confirmation tool applies the person's decisions and writes
   `<name>_confirmed.csv` and `<name>_confirmed.json` (the record: SHA-256 of
   the confirmed CSV, what it was confirmed from with hashes, the rule version,
   the overrides, the subject-sessions left out and the counts).

The rules the core applies:

- **A multi-input workflow over a selection without a `role` column is
  refused**: "have the selection reviewed and confirmed first". No roles are
  asked for and none are guessed.
- **A confirmed selection binds by its `role` column.** The editor offers
  exactly its role labels (`T1-contrast`, `T1+contrast`, `FLAIR`, …) for each
  input; a label it does not have is refused. A one-input workflow over a
  confirmed selection with several roles asks which role it reads.
- **A confirmed selection is read only with its record.** The core hashes the
  CSV bytes it binds and compares them with `confirmed_csv.sha256` in the
  `.json` beside it. A missing record, a record without the marker, or a CSV
  that differs is refused: "the confirmed selection was changed after
  confirmation". Confirm again instead of editing the confirmed file.
- **The checks are repeated when the run starts**, whatever bound the rows, so
  a file edited between loading and Run, or a plain selection sent with a
  multi-input workflow, does not start a run. Every series an item of a
  confirmed roll-out reads must also be a row of the confirmed CSV: a path
  edited in the run editor is refused, since the record would otherwise cite a
  selection the run did not use.
- Plans are not affected by these rules.

The run editor shows, once a confirmed selection is loaded, when it was
confirmed, under which review rule version and how many overrides and left-out
subject-sessions it has; a refused manifest shows the reason under the loader.

## What the run records

Every run keeps a folder `.vibe/runs/<id>/` with `run.json`, the workflow as it
ran (`recipe.yaml`) and the manifest as given (`manifest.csv`, with its
SHA-256). For a confirmed selection it also keeps the record
(`confirmation.json`), and `run.json` cites it under
`manifest.confirmed_selection`:

- `path` and `sha256` of `<name>_confirmed.json`, and `confirmed_csv_sha256`
  (equal to `manifest.sha256`: the bytes that were bound);
- `confirmed_at`, `note`, `rule_version`, `roles`, the number of `overrides` and
  `excluded` subject-sessions, `subject_sessions_confirmed`, `series_confirmed`;
- `source_manifest`, `criteria` and `index`, each `{path, sha256}` as the record
  states them.

A difference found while copying (the record changed after the run was
checked) is written to `error` there rather than hidden. The Markdown report
(`GET /api/runs/{id}/export?format=md`) prints the same as a "Confirmed
selection" line, which a methods section can cite; the JSON export inlines the
whole record.
