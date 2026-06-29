![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=MedMCP&reversal=false&fontSize=46&fontAlignY=28&desc=An%20Agentic%20Framework%20for%20Democratizing%20Medical%20Imaging%20Pipelines&descSize=24&descAlignY=55&fontColor=FFFFFF)

<p align="center">
  <a href="https://medmcp.ai"><b>medmcp.ai</b></a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#security">Security</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

MedMCP is an open, community-driven agentic framework that exposes validated medical imaging tools through a natural-language interface.
It is designed to enable clinicians, radiologists, and domain researchers to apply state-of-the-art image analysis methods without requiring expertise in command-line interfaces, Python environment management, or library-specific APIs.

Everything runs **on-premise**: a locally served model plans and sequences the work, all computation is delegated to tested implementations, and no imaging data, patient metadata, or results leave your infrastructure. You work through a single workspace that contains a *file explorer, image viewer, replay engine for personal workflows, and the chat interface*.

> [!WARNING]
> MedMCP is under active development and **not licensed for clinical use**.

---

## Quick start

The easiest way to run MedMCP is with the prebuilt Docker images.

**Start with a single command:** Set `MEDMCP_WORKSPACE` to the folder where your imaging data lives and results should be saved (any absolute path):

```bash
MEDMCP_WORKSPACE="$HOME/medmcp-data" \
  docker compose -f oci://ghcr.io/medmcp/compose:main up -d
```
Then open **http://localhost:8100**.

**Requirements:** Linux OS with an NVIDIA GPU (≥ 24 GB VRAM recommended for the local Gemma 4 26B model), a recent driver (≥ R570 / CUDA 12.8), and Docker with GPU access via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/) (CDI; rootless Docker works).

**Stop** with `docker compose -f oci://ghcr.io/medmcp/compose:main down`.

**Update** by re-running the start command with `--pull always`.

> Want to build from source or run host-native? See **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Features

- **Chat & Agent**: a familiar interface to interact with your local models, tools, and skills.
- **File explorer**: a builtin file explorer to organize your data.
- **Image viewer**: a builtin image viewer for medical images (`.nii.gz`, `.nrrd`, `.dcm`, ...) and other file formats (`.pdf`, `.csv`, ...).
- **Replay engine**: a replay engine for distilling and replaying processing pipelines into shareable workflows.
- **Easy to extend**: easily install new imaging capabilities through the UI.

---

## Security

MedMCP assumes the local model can be steered by prompt injection (e.g. text pasted from untrusted documents), so its safety model is built around explicit user control:

- **No auto-approval**: every tool call (bash, file writes, web fetches) requires an explicit click.
- **Localhost only**: the server binds to localhost. Do not expose port 8100 over a network without adding real authentication.
- **No data egress**: `web_search` is disabled and `web_fetch` requires approval. Nothing leaves your infrastructure by default.

To report a vulnerability, see **[SECURITY.md](SECURITY.md)**.

---

## Contributing

We welcome community contributions. We believe

See **[CONTRIBUTING.md](CONTRIBUTING.md)** to set up a development environment and submit a pull request.

---

## Team

MedMCP is developed and maintained by:

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/jqmcginnis"><img src="https://github.com/jqmcginnis.png" width="64" height="64" alt="Julian McGinnis"/></a><br/>
      <b>Julian McGinnis</b><br/>
      Technical University of Munich<br/>
      <a href="https://jqmcginnis.github.io/">Website</a> · <a href="https://github.com/jqmcginnis">GitHub</a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/pfriedri"><img src="https://github.com/pfriedri.png" width="64" height="64" alt="Paul Friedrich"/></a><br/>
      <b>Paul Friedrich</b><br/>
      University of Basel<br/>
      <a href="https://pfriedri.github.io">Website</a> · <a href="https://github.com/pfriedri">GitHub</a>
    </td>
  </tr>
</table>

… together with the wider [open-source community](CONTRIBUTING.md).

### Contributors

[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://jqmcginnis.github.io/"><img src="https://avatars.githubusercontent.com/u/33037028?v=4?s=100" width="100px;" alt="Julian McGinnis"/><br /><sub><b>Julian McGinnis</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-dev/commits?author=jqmcginnis" title="Code">💻</a> <a href="#ideas-jqmcginnis" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/medmcp/medmcp-dev/commits?author=jqmcginnis" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://pfriedri.github.io"><img src="https://avatars.githubusercontent.com/u/101359393?v=4?s=100" width="100px;" alt="Paul Friedrich"/><br /><sub><b>Paul Friedrich</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-dev/commits?author=pfriedri" title="Code">💻</a> <a href="#ideas-pfriedri" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/medmcp/medmcp-dev/commits?author=pfriedri" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification — contributions of any kind are welcome!

---

## License

MedMCP is released under the [Apache License 2.0](LICENSE).
