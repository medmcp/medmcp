# Changelog

Notable, user-visible changes to MedMCP. Format follows
[Keep a Changelog](https://keepachangelog.com/); entries land under
**Unreleased** as PRs merge and move under a version heading at release time.

## Unreleased

### Changed

- Agent: mistral-vibe 2.26 with its Unified Harness runtime; the MedMCP agent profile (`.vibe/agents/medmcp.toml`) carries the system prompt and the approval posture, and the build-time vibe patches are gone.
- Agent: stack tools are discovered and called by the model through the harness (`search_tool_functions`, `run_typescript`); each stack call still asks for approval on its own and shows as its own card.
- Model: the shim renders tool definitions and parses the model's tool calls itself (`MEDMCP_SHIM_TOOLS=glimmer`, default), so Ollama's Glimmer parser no longer truncates turns; `MEDMCP_SHIM_TOOLS=ollama` restores the old path.
- Chats: the workspace keeps its own transcript of every chat; export, titles and distillation read it, and chats written by older versions still read.
- Chats: a tool-catalogue lookup shows as a note, not a card.

### Fixed

- Viewer: the intensity window survives a panel resize instead of falling back to Auto.
- Chats: opening a chat right after closing one no longer fails on vibe 2.26.1's post-close race; the server waits it out.
- Approval dialog: vibe 2.26's session-scoped options are filtered by what they do, so none approves more than the call in hand.
- Tooling: ruff 0.17.1, pyright 1.1.414, pre-commit 4.7.0, types-pyyaml.

## 0.6.0 — 2026-10-19

### Added

- Approval dialog: a read-only shell command (`ls`, `cat`, `grep`, …) can be approved for the rest of the chat; one that redirects to a file or leaves the workspace still asks. `MEDMCP_READ_ONLY_APPROVAL=0` turns it off.
- Explorer: upload a whole folder (subfolders kept), upload into any folder, and download a file or a folder as a streamed `.zip`.
- Explorer: right-click on empty space opens New folder / Upload files… / Upload folder… / Refresh; the header's New folder and Upload files act on the selected folder; folders upload by drag-and-drop.
- Chats: Export opens a chat as a printable page (messages, tool calls with inputs, approval and result) with Save as PDF.

### Fixed

- Chats: returning to a chat after a page reload or reconnect opens that chat again instead of silently starting a new, empty one.
- Chats: a message sent while a reply is still running no longer leaves the chat stuck on "A turn is already running".
- Approval dialog: tool-call explanations from a slow local model are no longer cut off after 20 s (now 90 s).
- Viewer: a file rewritten by a tool is reloaded instead of served from the browser cache.
- Stacks: containers get 8 GB of shared memory; Docker's 64 MB default killed full-resolution TotalSegmentator runs.

### Changed

- Image build: the agent (mistral-vibe) is patched so every chat shares one scratchpad path (the model's cached prompt is reused) and a chat-long approval never covers a redirect; the build fails if a vibe update no longer matches.
- Chats: consecutive calls of one tool share a single card, and shell commands no longer get View/Overlay chips.

## 0.5.0 — 2026-10-16

### Added

- Viewer: CSV and TSV files open as a sortable table, XLSX spreadsheets with one tab per sheet, Markdown as a rendered document, JSON as a collapsible tree; other text files get line numbers, and a file of unknown type is shown as text when its bytes are text. Chat result chips offer View for these too.

## 0.4.3 — 2026-10-16

### Fixed

- Chat: generated titles work again; chats no longer keep the first prompt as their name.
- Workflows: distilling a chat without a title names it first, so the workflow is named after the task.
- Approval dialog: the path checks (input exists, output folder exists, would overwrite) are shown again.
- Tool stacks and Update windows: a connection dropped mid-install or mid-update is reported instead of leaving the window busy.
- Settings: a change the server rejects snaps the switch back.
- Explorer: New folder names the folder inline in the tree.
- Explorer: Analyze `.hdr`/`.img` and `.v16` volumes get the volume icon.
- Explorer: a drive mounted inside the workspace, or a folder holding one, cannot be deleted or moved.
- Compose: a wrong docker socket path fails `up` instead of silently breaking every stack.
- Workflow requirements: the installed-image digest resolves for a registry with a port.
- LLM shim: a 4xx from Ollama is passed through instead of retried.
- `scripts/install_ollama.sh`: requires Ollama ≥ 0.32.15 and supports arm64.

## 0.4.2 — 2026-10-13

### Added

- Tool stacks: `medmcp-cardiac` (cine MRI ventricle segmentation and function, AHA wall analysis, CT calcium score) in both install catalogs.

## 0.4.1 — 2026-10-12

### Added

- Viewer: 4D volumes (cine MRI, time series) get a frame slider with play/pause in the status bar; ←/→ step frames (Shift = 10), space plays; a 4D label overlay follows the image frame and the legend counts the frame on screen.

## 0.4.0 — 2026-10-07

### Added

- Viewer: label overlays size their colour table to the data (a 117-structure TotalSegmentator result no longer collapses above label 64), and a legend lists the structures present with names (from the `_labels.csv` the stack writes beside the segmentation), volumes in mL, click-to-jump and isolate.
- Viewer: continuous overlays (probability maps, co-registered images, warps) render with a colormap and threshold instead of being forced into label colours; the type can be switched by hand.
- Viewer: a status bar reads out voxel, millimetre position, intensity and label under the pointer, and the intensity window with presets (Auto, Full range, and CT soft tissue/lung/bone/brain/liver on Hounsfield data).
- Viewer: files a tool result names appear as chips in the chat with View and Overlay actions, so the agent's output opens without hunting in the explorer.
- Viewer: PNG snapshot of the current view, ↑/↓ and PageUp/PageDown step slices (Shift = 10), `o` hides the overlay for a before/after look.

### Fixed

- Chat: a tool result, and its View/Overlay chips, stays after a reload or resume; the replayed call arrived without its output before.
- No more browser tooltips popping up over the workspace: hover titles are gone, icon buttons keep an accessible label.
- Delete confirmations are the app's own dialog, not the browser's popup.
- Viewer: the PNG snapshot is flattened onto black; the raw canvas kept partial alpha along the crosshair and edges, which image viewers showed washed out.
- Viewer: label overlays drew several ids in one colour and a hidden label showed when its neighbour was visible (Niivue's generic shader filters the colour table); label maps now go through its atlas shader, which looks ids up exactly.
- Viewer: a label overlay is always sampled nearest-neighbour, whatever the interpolation setting; linear sampling blended ids into colours no label has.
- Viewer: the legend says which `_labels.csv` would name the labels when none is found; the viewer carries no label table of its own.
- Viewer: `.dcm` files showed a Niivue error; they now say to convert the series to NIfTI with the DICOM stack.
- Viewer: a panel resize no longer re-downloads the volume.

## 0.3.1 — 2026-10-06

### Added

- Choose the local model: click the model name in the chat (or Settings → Model) to switch between seven models. A model downloads the first time you pick it and can be deleted again; Muse Glimmer stays the default.
- Optional cloud model: Settings → Advanced → Cloud model runs chats on Claude, ChatGPT or another OpenAI-compatible service with your own API key. Off by default; it asks for consent and shows a red banner while on, because chats then leave your machine.

### Changed

- Shorter text in Settings and in the External MCP window.
- A chat's provenance record now says where its model ran.

## 0.3.0 — 2026-09-09

### Security

- This is the first signed release. Updating to it from the host is unverified once; every update from it onward is verified.
- Published images and the compose artifact are signed in CI (Sigstore keyless, bound to this repository's release workflow); the workspace verifies the signature before applying an update or installing a stack, pins the update to the verified digest, and refuses an image that does not verify.

### Added

- The workspace says when a new MedMCP release is out (header notice, Settings → Version) and shows its notes.
- Update from the UI: *Update now* pulls the release, restarts on it, and rolls back if it does not start. `docker compose` installs only; others get the host command.
- Settings → Advanced can switch the daily release check off; `MEDMCP_UPDATE_CHECK=0` turns every check off for a deployment; `MEDMCP_UPDATE_URL` points the check at a mirror.

## 0.2.3 — 2026-09-03

### Changed

- Workflow runs keep going when the page is closed or reloaded; the page picks them up again.
- The run editor checks every row's input files before anything starts and flags a missing file per row.
- One run editor replaces the Single/Batch switch and the separate preview screen: rows, checks, and the resolved steps sit on one screen with one Run button. Stop is now an explicit action rather than closing the page.
- Saving a chat as a workflow is instant and names the workflow after the chat; no model pass runs and only `recipe.yaml` is written.
- Workflows no longer have a draft/active state: every workflow can be run, renamed, exported and deleted. Existing `draft/` and `active/` folders are folded together on first use; an earlier release will not list them afterwards.

### Removed

- The Refine action and the generated `SKILL.md` narrative; a workflow is its recorded steps.
- `medmcp promote` (and `just promote`) and the `--no-llm` flag of `medmcp distill`.

## 0.2.2 — 2026-09-02

### Added

- Chats are given a descriptive title after the first reply and refined once a few turns later. A name you set yourself is kept.
- Replies stream as they are written again; the model-server proxy keeps repairing tool calls.
- The chat says when the model server is being retried, and when a turn stopped at its step limit.

### Changed

- mistral-vibe 2.24.5 and Ollama 0.33.2 (`docker compose` installs now pin the tested Ollama release; 0.32.15 is the minimum for host installs).
- The Chats menu is wider and stays fully on screen at any window size.

### Security

- File writes and edits keep asking for approval. The updated agent runtime would otherwise have approved them automatically.
- Host installs: the read-only shell-command allowlist no longer re-fills itself on every start.

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
