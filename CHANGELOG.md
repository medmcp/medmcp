# Changelog

Notable, user-visible changes to MedMCP. Format follows
[Keep a Changelog](https://keepachangelog.com/); entries land under
**Unreleased** as PRs merge and move under a version heading at release time.

## Unreleased

## 0.2.1 — 2026-08-25

### Changed

- Published images link back to this repository and carry version and commit labels.

## 0.2.0 — 2026-08-25

### Added

- **External MCP servers** — connect the agent to remote MCP services (Settings →
  Advanced). Off by default, behind a consent dialog; tokens stay on this machine.
- **TotalSegmentator stack** — whole-body CT/MR anatomy segmentation, installable
  from the Tool stacks window.
- `/healthz` reports the running version and the build it came from.

### Changed

- Settings drawer reorganised; external MCP servers moved into a window of their own.
- Wording tidied across the interface.

### Fixed

- Uninstalling a tool stack now removes it for good — it could previously come back.

### Security

- Stacks that ask for network access need your consent at install, and are marked
  **internet** while installed.
- Saved workflows can only touch files inside your workspace.
- Requests to the workspace from other websites are refused (`MEDMCP_ALLOWED_HOSTS`
  / `MEDMCP_ALLOWED_ORIGINS` allow a reverse proxy).

## 0.1.0

First public release. MedMCP is an on-premise agentic framework that exposes
validated medical imaging tools through a natural-language interface: a locally
served model plans and sequences the work, every computation is delegated to a
tested implementation, and no imaging data, patient metadata, or results leave
your infrastructure.

**Not licensed for clinical use.**

### Workspace

- A single browser workspace at `http://localhost:8100` with four resizable
  panels: file explorer, medical-image viewer, workflow manager, and agent chat.
- **File explorer** over your workspace directory — multi-select, rename, delete,
  new folder, upload, and drag-and-drop into the viewer and workflow inputs.
- **Image viewer** — volumes (`.nii.gz`, `.nrrd`, `.dcm`, …) rendered with
  Niivue as multiplanar slices plus a 3D view, with PDF, image, and text preview
  for everything else. Drag a segmentation from the explorer onto the image to
  overlay it with a per-label colormap and adjustable opacity. A reset button
  returns the volume to how it looked when it opened, leaving your viewer
  settings and any overlay untouched.
- **Chat** with a local model over Ollama (Meta's Muse Glimmer 30B by
  default — open-weight, Apache 2.0, and built for tool use), with
  streamed responses, tool-call cards, and a context meter. Chats persist and
  can be resumed, renamed, archived, deleted, branched into a parallel session,
  or rewound to before an earlier message (restoring the files it touched).
  Ask for more than one thing and the agent writes out a task list before it
  starts, ticking off each part as it lands.

### Imaging tool stacks

- Imaging capabilities install as **tool stacks** — containerized MCP servers
  browsed and installed from a searchable stacks window, or installed
  host-native as isolated `uv` tools. Stacks are discovered automatically; no
  config editing.
- Stacks bake their model weights at build time and run offline.

### Provenance and reusable workflows

- Every session records an append-only **provenance** trail: an environment
  manifest, one normalized event per completed tool call, and a persisted
  mirror of every permission decision, rendered on demand as a report.
- **Distill** a session into a reusable workflow: MedMCP filters the exploratory
  and failed calls, lifts concrete file paths into named inputs, and writes a
  recipe plus a readable description of what it does.
- **Replay** a workflow deterministically on new data with no model in the loop
  — including in batch over a whole cohort — after previewing and confirming the
  resolved steps. A run asks only for the inputs that are genuinely yours to
  choose: where a step simply wrote next to its input, the destination is worked
  out from the file you give it, and the results land beside the data you are
  replaying on.
- **Share** a workflow as one self-contained `.workflow.yaml` file, and import
  one you were sent as a reviewable draft — including the fields it fills in for
  itself, so a colleague is asked no more than you were.

### Deployment

- Runs from prebuilt images with a single `docker compose` command, or
  host-native via `just` recipes.
- Multi-architecture images (x86-64 and ARM) built on one shared CUDA base, with
  GPU access through the NVIDIA Container Toolkit's CDI interface, so rootless
  Docker is supported.
- A local proxy between the agent and the model server repairs tool-call
  failures in the bundled model's parser — it works around the name collision
  that triggers them, retries turns that are cut short, and neutralises malformed
  tool arguments. On by default, since the failure it prevents is silent; set
  `MEDMCP_LLM_SHIM=0` to talk to the model server directly.

### Security

- **Nothing is changed or sent without your approval.** Writing a file, editing
  one, fetching a URL, or running a command with side effects each require an
  explicit click. There is no "always allow" and no session-wide approval; each
  call is approved on its own, and the dialog shows the call's arguments
  alongside a generated plain-language risk summary. Read-only shell commands
  (`ls`, `cat`, `grep`, …) run without a prompt inside your workspace; the same
  commands pointed outside it ask first, and `find -exec` counts as execution.
- **Invented file paths are sent back to the agent, not to you.** Before a tool
  call reaches the approval dialog, MedMCP checks the paths it would use. If one
  cannot be right — an input that does not exist, a destination folder that is
  missing — the call is refused and the agent is told why, including what the
  nearest real folder holds and, where the path merely had the wrong prefix,
  which file it probably meant. The agent corrects itself and tries again, so
  the calls you are asked to approve are ones whose paths resolve. If it fails
  to correct after two attempts the call is passed through to you anyway, so a
  confused agent surfaces rather than looping out of sight.
- **File paths are checked before you approve.** Local models invent
  plausible-looking paths, and until now the first sign was the tool failing
  after you had already approved it. The approval dialog now says, for each path
  the call would use, whether it is actually there: an input that does not exist
  is flagged in red, a destination folder that is missing likewise, and a file
  that would be silently overwritten is called out. A missing path also shows
  what the nearest real folder actually contains, which usually makes the
  intended path obvious — the model tends to get the folder right and the
  filename wrong, or to invent one subject in an otherwise correct tree. Paths
  pointing outside the workspace are flagged as unverifiable. The check is a
  plain filesystem lookup — it never asks a model, so unlike the risk summary
  beside it, it cannot be wrong about what is on disk. It is advice, not a gate:
  the decision stays yours.
- **No data egress.** `web_search` is disabled and `web_fetch` requires
  approval. Tool stacks run with networking denied, all Linux capabilities
  dropped, and no privilege escalation — a tool call cannot reach the network
  even if the agent is steered into attempting one.
- **Localhost only.** The server binds to the loopback interface and has no
  authentication; the containerized deployment publishes its port only to the
  host loopback. Do not expose it to a network without adding real auth.
- The filesystem API refuses any path resolving outside the workspace root.
- Permission decisions are written to an audit trail that cannot be silenced.
