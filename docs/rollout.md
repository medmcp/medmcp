# Cohort roll-out

A workflow distilled from one chat can be rolled out over a cohort: the run
editor (or `medmcp replay <name> --batch <manifest>`) reads a manifest written by
the cohort stack, binds the workflow's inputs per subject or subject-session,
pre-flights every item, and runs the steps without a model in between. Each item
writes its results to its own folder under the output folder
(`results/<workflow>/<subject>[/<session>]` unless another is chosen).

## Manifests

| Manifest | Written by | One row per | Inputs bound by |
| --- | --- | --- | --- |
| `batch_plan.csv` | `plan_batch` | subject (or subject-session), with a `status` | the plan's file columns (matched to the recorded example, or chosen in the editor) |
| `<name>_series.csv` | `select_series` | selected series | `path`; one input only (see below) |
| `<name>_confirmed.csv` + `<name>_confirmed.json` | `review_selection`, then `confirm_selection` | role per subject-session | the `role` column |

Only `ok` rows of a plan run; flagged rows are listed and left out.

## Selections: confirm before a multi-input roll-out

A plain selection can hold two series of one role in a visit (a repeat, a
reformat, a second plane). A tool that reads one scan is not affected: the
workflow runs once per selected series. A workflow that reads several scans per
subject-session (a T1, a T1 with contrast, a FLAIR) needs exactly one series per
role, and which one is decided with the person, not by the roll-out:

1. `select_series` picks the candidate series.
2. `review_selection(manifest_path, roles=[...])` shows one series per subject,
   session and role, with the alternatives, ties and missing roles.
3. `confirm_selection(review_path, overrides, note)` applies the person's
   decisions and writes `<name>_confirmed.csv` (the selection's columns plus
   `role`, `decided_by` and the index's own `index_sequence`/`index_contrast`/
   `index_plane`) and `<name>_confirmed.json` (the record: SHA-256 of the
   confirmed CSV, the source selection, criteria and index with their hashes,
   the rule version, every decision and override, the subject-sessions left out
   and the counts).

The rules the core applies:

- **A multi-input workflow over a selection without a `role` column is
  refused**: "run review_selection and confirm_selection first". No roles are
  asked for and none are guessed.
- **A confirmed selection binds by its `role` column.** The editor offers
  exactly its role labels (`T1-contrast`, `T1+contrast`, `FLAIR`, …) for each
  input; a label it does not have is refused. A one-input workflow over a
  confirmed selection with several roles asks which role it reads.
- **A confirmed selection is read only with its record.** The core hashes the
  CSV bytes it binds and compares them with `confirmed_csv.sha256` in the
  `.json` beside it. A missing record, a record of something else, or a CSV
  that differs is refused: "the confirmed selection was changed after
  confirmation". Confirm again instead of editing the confirmed file.
- **The checks are repeated when the run starts**, whatever bound the rows, so
  a file edited between loading and Run, or a plain selection sent with a
  multi-input workflow, does not start a run. Every series an item of a
  confirmed roll-out reads must also be a row of the confirmed CSV: a path
  edited in the run editor is refused, since the record would otherwise cite a
  selection the run did not use.
- Plans (`batch_plan.csv`) are not affected by these rules.

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
