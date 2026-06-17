# stacks.d — container-stack manifests

Each `stacks.d/<name>.toml` declares an MCP tool stack that runs as a **container**
(launched by the core via `docker run -i`) instead of a `uv tool` install. The
core's `load_mcp_servers()` (`src/medmcp/settings.py`) reads these and merges them
with the entry-point scan; a uv-tool install of the same name wins (local dev).

`${VAR}` references in `command`/`args`/`skills_path` are expanded against the
environment at load time — notably `${MEDMCP_WORKSPACE}` so the bind-mount lands
at the host path the workspace server already uses (path parity).

Example (`medmcp-dicom.toml`):

```toml
name = "medmcp-dicom"
command = "docker"
args = [
    "run", "--rm", "-i",
    "-v", "${MEDMCP_WORKSPACE}:${MEDMCP_WORKSPACE}",
    "ghcr.io/medmcp/dicom:dev",
]
# GPU stacks add:  "--device", "nvidia.com/gpu=all"  (CDI, rootless).
# skills_path = "/app/stacks.d/medmcp-dicom/skills"   # vendored SKILL.md set
```
