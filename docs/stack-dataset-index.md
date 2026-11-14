# Stack dataset index

A stack can keep an index of a folder of scans: a table with one row per scan and
a small summary file beside it. When such a stack is active and a chat message
names a workspace folder, the core tells the assistant what the index says about
the folder, or which tool builds the index first, so it does not list the folder
by shell. The core knows no stack's files or tools by name: everything comes from
the stack's declaration.

## For someone running MedMCP

- Nothing to configure. Without an active stack that declares an index, no note is
  added and the assistant looks at folders as before.
- A message names a folder by an absolute path, by the name of a top-level
  workspace folder as people type it, or through the file open in the viewer. At
  most three folders are described per message.
- The note is sent with the message and is not part of the chat's stored text.

## For a stack author

Declare the index in the image label `org.medmcp.stack`:

```json
{"name": "medmcp-example",
 "dataset_index": {"summary": ".index/summary.json",
                   "build": "build_index",
                   "query": ["summarize_index", "pick_scans"]}}
```

| Key | Meaning |
| --- | --- |
| `summary` | Path of the summary file, relative to the indexed folder. No `..`, not absolute. |
| `build` | The tool that indexes a folder. |
| `query` | Up to four tools that answer from an index. May be empty. |

Tool names are plain identifiers (letters, digits, `_`). A declaration that does
not fit is dropped and the stack installs without it.

The summary file is a JSON object. The core reads one key:

- `description`: one line saying what the folder holds, for example
  `"3 subjects, 18 sessions, 315 scans; sequences: T1 54, FLAIR 18"`. It is quoted
  to the assistant on one line, with square brackets replaced and cut at 400
  characters. Every other key is the stack's own.

What the assistant is told:

- The folder has a summary: that it already has a table of its scans, the
  description, and to call the `query` tools instead of reading its files.
- The folder holds scans (`.nii`, `.nii.gz`, `.dcm`, `.ima`, `DICOMDIR`) but has no
  summary: to call the `build` tool first.
- The folder holds no scans: nothing.
