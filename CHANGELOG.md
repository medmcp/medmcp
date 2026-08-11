# Changelog

Notable, user-visible changes to MedMCP. Format follows
[Keep a Changelog](https://keepachangelog.com/); entries land under
**Unreleased** as PRs merge and move under a version heading at release time.

## Unreleased

### Added

- **Rewind**: hover a user message in a resumed chat to rewind the conversation
  to before it — previews which workspace files would be restored, asks for
  confirmation, then truncates in place (#93).
- **Branch a chat**: a git-branch icon on the open chat's row in the Chats menu
  duplicates its full history into a new session, for trying a second analysis
  path on the same context; branches inherit the source's title (#92).
- Stack connection failures now surface as a chat message on session start
  ("The following MCP servers failed to connect: …") instead of tools being
  silently missing (#89).

### Changed

- One chat, one entry: context compaction no longer splits a chat into multiple
  sessions in the Chats menu, and reopening a compacted chat restores the full
  post-compaction context — including after an agent restart (#90).
- Deleting a chat now removes everything it produced: the whole transcript
  chain, the provenance record, and the agent's own stored copy (#89, #90).
- Renaming a chat writes the title into the agent's session metadata too (#89).
- The context meter reports the model's actually-served window (Ollama
  `num_ctx`), not the model family's nominal maximum (#89).
- Replayed chats show the user's message text without the internal
  `[workspace context: …]` viewer note, natively (#89).
- Overlaying a segmentation is now drag-and-drop only: drag the volume from the
  file explorer onto the image. The dropdown of every volume in the workspace is
  gone, and the bar above the image appears only once something is overlaid —
  showing that volume's name, its opacity, and a button to remove it (#99).
- Provenance recording is treated as always-on: its off switch moved out of the
  settings drawer's General list into a collapsed **Advanced** section at the
  bottom, so a chat's audit trail is no longer one stray click from stopping
  (#100).
- Saved workflows now belong to the replay engine alone. The agent can no longer
  invoke one as a skill — a workflow runs from the workflow panel, replaying its
  recorded tool calls exactly, or it does not run. Promoting a draft now just
  marks it reviewed and worth keeping. The settings drawer's "Personal
  workflows" master switch and per-workflow switches are gone with the skill
  loading they controlled (#101).

### Fixed

- The approval dialog shows the tool call's arguments again. The agent announces
  a call before it has finished writing its arguments, so they arrived a moment
  later and were dropped — leaving you to approve an action with only its name
  visible. Tool-call cards fill in the same way, and provenance records the
  arguments again (so those sessions distill into working workflows) (#102).
- Sent messages no longer appear twice (the agent runtime echoes live prompts
  since vibe 2.23; the echo is now merged into the already-rendered bubble) —
  as a side effect, messages sent in the current session become rewindable
  immediately instead of after a reload (#94).
- A branched chat and its original are now truly independent: opening the
  original no longer attaches to the branch, deleting the original no longer
  deletes the branch, and workflow distillation no longer mixes the two (#94).
- Workflow distillation covers tool calls made after a context compaction (#89).

### Security

- Every bash command the agent runs now requires interactive approval: the
  read-only allowlist was removed after finding that an output redirect from an
  allowlisted command (e.g. `echo "" > file`) wrote files without a prompt (#89).
- The permission dialog no longer offers any way to approve more than the call
  in front of you: both "Always allow" (which persisted an auto-approval into
  the config) and "Allow for remainder of this session" are gone, leaving
  **Allow** and **Deny**. Every tool call is approved on its own (#89, #98).
- The agent's built-in `skill-creator` skill is disabled: workflow distillation
  stays the single, reviewable skill-authoring path (#89).

### Dependencies

- mistral-vibe 2.17 → 2.23.3, fastapi 0.141, ruff 0.16 (#84, #88, #89).
