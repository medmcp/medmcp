---
name: dcm-to-bids
description: >
  Convert a DICOM directory tree into a BIDS-organised dataset with NIfTI images.
  Covers single-step NIfTI conversion and full BIDS layout creation.
compatibility: Requires the medmcp-dicom MCP server (console script medmcp-dicom).
license: Apache-2.0
---

## Workflow — dcm-to-bids

### Step 1 — Gather inputs and confirm

Ask the user for two things only. Provide suggested paths for both:
- **dicom_root**: root directory of the DICOM data (tool scans recursively)
- **output_dir**: where to write the BIDS dataset

Once you have both paths, **compute** the proposed dataset name as the last
path component of `dicom_root` with `_bids` appended. For example, if
`dicom_root` is `/data/study_2024`, the dataset name is `study_2024_bids`.

Then present the concrete values back to the user — fill in the actual strings,
not placeholders:

> **Proposed settings**
> - DICOM source: `/data/study_2024`
> - BIDS output: `/data/study_2024_bids`
> - Dataset name: `study_2024`
>
> Reply **yes** to proceed, or provide a different dataset name.

**Do not call any tool until the user replies "yes" (or equivalent explicit
confirmation). A different dataset name counts as a reply — confirm again with
the updated name before proceeding.**

### Step 2 — Full BIDS conversion

Call `build_bids_dataset(dicom_root, output_dir, dataset_name, anonymize=True)`.
- `anonymize=True` (default): subject labels are `sub-001`, `sub-002` … only.
- `anonymize=False`: source patient IDs are also written to `participants.tsv`.
  Only set this if the user explicitly asks for non-anonymous output.

Follow the `_render` instructions in the result.

If `unclassified` is non-empty: display the unclassified series in a separate
section and ask the user how to handle them (rename manually, skip, or note for
later). Do **not** re-run the whole conversion just to fix suffix labels.

---

## Gotchas

- **Call on the dataset root**, not a subdirectory — the tool scans recursively.
- **dcm2niix required**: `dcm2niix` must be on the PATH.
  If not installed, the tool returns a clear error with installation instructions.
  Do not retry; report the error and the install hint to the user.
- **PHI**: `anonymize=True` is the default. Never set `anonymize=False` unless
  the user explicitly requests non-anonymous output.
- **Research software**: flag any context that suggests clinical decision-making
  and stop. This package is not licensed for clinical use.
- **Unclassified series**: series that do not match any suffix rule are still
  converted and placed in `anat/` with a fallback suffix. They appear in the
  `unclassified` list so the user can review them.
- **Multi-echo sequences**: each echo is written as a separate NIfTI with an
  `_echo-N` label appended to the filename. This is BIDS-valid but the user
  should verify the echo ordering.
- **DWI series**: dcm2niix automatically produces `.bval` and `.bvec` files
  alongside the NIfTI. These are moved to the BIDS `dwi/` directory.
- **Duplicate series per session**: if two series map to the same
  sub/ses/suffix combination, `_run-01` / `_run-02` labels are added
  automatically.
