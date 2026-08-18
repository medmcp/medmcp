# Changelog

Notable, user-visible changes to MedMCP. Format follows
[Keep a Changelog](https://keepachangelog.com/); entries land under
**Unreleased** as PRs merge and move under a version heading at release time.

## Unreleased

### Changed

- **The agent now keeps a visible task list when you ask for more than one
  thing.** Multi-part requests — strip this scan, register it, then report the
  volumes — are written out as a checklist before the work starts and ticked off
  as each part lands, so you can see what it understood you to be asking and how
  far it has got. Single requests are unaffected.

### Fixed

- **Workflow runs only ask for the inputs you actually have to provide.**
  A workflow distilled from a session used to ask for the scan *and* for the
  directory that scan sits in — so a two-decision pipeline arrived with several
  boxes to fill, and the folder was frozen to wherever the workflow happened to
  be recorded. Where a step simply wrote next to its input, that folder is now
  worked out from the file you give it and no longer appears in the form, so the
  results land beside the file you are replaying on. It remains part of the
  workflow, and *Change where results are written* opens it if you want the
  output somewhere else. Where the folder can't be worked out unambiguously —
  two inputs sitting in the same directory, say — nothing is guessed and it is
  asked for as before.
- **Tool stacks no longer go missing on the first run after installing them.**
  A stack that takes a moment to start the first time — the usual case, since its
  image has just been pulled and nothing is cached yet — could be given up on
  before it answered, leaving that chat with none of its tools. Stacks are now
  given a startup budget a cold start fits into, and existing installations are
  repaired automatically — no reinstall needed. Starting a new chat also retries,
  so a stack that was only slow comes back on its own.
- **The agent no longer goes silent on multi-step requests.** Asking for a
  multi-step imaging workflow could leave the chat with no reply at all — no text,
  no error, nothing to retry. The local model server was failing to read the
  model's tool calls and closing the connection without reporting it, so roughly
  half of all tool-using turns were lost. Affected turns now complete, and a turn
  that genuinely cannot be recovered says so instead of hanging.
- **Agent replies are no longer discarded.** Responses from the model were being
  dropped before they reached the chat because the reasoning output was read from
  the wrong field.

### Added

- **`medmcp-llm-shim`** — an optional local proxy that sits between the agent and
  the model server and repairs the tool-call failures above: it works around the
  name collision that triggers them, retries turns that are cut short, and
  neutralises malformed tool arguments that would otherwise leave a chat
  permanently unable to reply. **On by default** — the failure it prevents is
  silent, so there is nothing for you to notice or search for when it happens.
  Set `MEDMCP_LLM_SHIM=0` to disable it and talk to the model server directly.
- **Reset view** button in the image viewer. After panning, zooming, rotating the
  3D view, or scrolling far through the slices, one click returns the volume to
  how it looked when it opened. Your viewer settings and any overlay stay as they
  are — only the camera moves.

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
  overlay it with a per-label colormap and adjustable opacity.
- **Chat** with a local model over Ollama (Meta's Muse Glimmer 30B by
  default — open-weight, Apache 2.0, and built for tool use), with
  streamed responses, tool-call cards, and a context meter. Chats persist and
  can be resumed, renamed, archived, deleted, branched into a parallel session,
  or rewound to before an earlier message (restoring the files it touched).

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
  resolved steps.
- **Share** a workflow as one self-contained `.workflow.yaml` file, and import
  one you were sent as a reviewable draft.

### Deployment

- Runs from prebuilt images with a single `docker compose` command, or
  host-native via `just` recipes.
- Multi-architecture images (x86-64 and ARM) built on one shared CUDA base, with
  GPU access through the NVIDIA Container Toolkit's CDI interface, so rootless
  Docker is supported.

### Security

- **No auto-approval.** Every tool call — bash, file writes, web fetches —
  requires an explicit click. There is no "always allow" and no session-wide
  approval; each call is approved on its own. The approval dialog shows the
  call's arguments alongside a generated plain-language risk summary.
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
