![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=MedMCP&reversal=false&fontSize=46&fontAlignY=28&desc=An%20Agentic%20Framework%20for%20Democratizing%20Medical%20Imaging%20Pipelines&descSize=24&descAlignY=55&fontColor=FFFFFF)

<p align="center">
  <a href="https://medmcp.ai"><b>medmcp.ai</b></a> ·
  <a href="#installation">Installation</a> ·
  <a href="#security">Security</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

MedMCP is an open, agentic framework that puts validated medical-imaging tools behind a natural-language interface. It lets clinicians, radiologists, and researchers run state-of-the-art image-analysis pipelines without the command line, Python environments, or library-specific APIs.

Everything runs **on-premise**: a locally served model plans and sequences the work, all computation is delegated to tested implementations, and no imaging data, patient metadata, or results leave your infrastructure. You work through a single browser workspace — file explorer, image viewer, workflows, and chat — at `http://localhost:8100`.

Learn more at **[medmcp.ai](https://medmcp.ai)**.

> [!WARNING]
> MedMCP is under active development and **not licensed for clinical use**.

---

## Installation

The easiest way to run MedMCP is with the prebuilt Docker images — nothing to build, no source to download.

**You need:** Linux with an NVIDIA GPU (≥ 24 GB VRAM recommended for the local Gemma 4 26B model), a recent driver (≥ R570 / CUDA 12.8), and Docker with GPU access via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/) (CDI; rootless Docker works).

**Start it.** Set `MEDMCP_WORKSPACE` to the folder where your imaging data lives and results should be saved (any absolute path):

```bash
MEDMCP_WORKSPACE="$HOME/medmcp-data" \
  docker compose -f oci://ghcr.io/medmcp/compose:main up -d
```

The first start downloads the model (~18 GB), so give it a few minutes. Then open **http://localhost:8100**.

**Add imaging tools.** In the UI, open **Settings → Stacks → Available** and install the stacks you need (e.g. `dicom`, `neuro`). Each one downloads the first time the agent uses it.

**Stop** with `docker compose -f oci://ghcr.io/medmcp/compose:main down`. **Update** by re-running the start command with `--pull always`.

> Want to build from source or run host-native? See **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Team

MedMCP is developed and maintained by:

- **Julian McGinnis** — Technical University of Munich
- **Paul Friedrich** — University of Basel

… together with the [open-source community](CONTRIBUTING.md).

---

## Security

MedMCP assumes the local model can be steered by prompt injection (e.g. text pasted from untrusted documents), so its safety model is built around explicit user control:

- **No auto-approval** — every tool call (bash, file writes, web fetches) requires an explicit click. There is no auto-approval path.
- **Localhost only** — the server binds to localhost. Do not expose port 8100 over a network without adding real authentication.
- **No data egress** — `web_search` is disabled and `web_fetch` requires approval. Nothing leaves your infrastructure by default.

To report a vulnerability, see **[SECURITY.md](SECURITY.md)**.

---

## Contributing

Contributions are welcome. MedMCP grows through a shared schema for tool and skill metadata, CI-based testing, and a central registry — so every new contribution is immediately available to all users. See **[CONTRIBUTING.md](CONTRIBUTING.md)** to set up a development environment and submit a pull request.

---

## License

MedMCP is released under the [Apache License 2.0](LICENSE).
