# MedMCP workspace frontend

The Vite + React 19 + TypeScript single-page app served by the MedMCP workspace
server (`medmcp-workspace`, `src/medmcp/server.py`). It is the project's only
frontend; the server serves this app's production build from `dist/` and returns
a build hint if it is missing.

## Layout

Four resizable panels (`react-resizable-panels`), wired together in `App.tsx`:

| Panel | Component | What it does |
| --- | --- | --- |
| Top-left | `FileExplorer.tsx` | Workspace file tree (`react-arborist`) — multi-select, context menu, drag source for the viewer and workflow inputs |
| Top-right | `Viewer.tsx` | Medical-image viewer — [Niivue](https://github.com/niivue/niivue) for volumes (multiplanar + 3D, drag-and-drop segmentation overlays), plus PDF/image/text preview |
| Bottom-left | `WorkflowPanel.tsx` | Distilled workflows — list, inspect, export/import, and run via the replay engine (including batch runs) |
| Bottom-right | `Chat.tsx` | Agent chat over a WebSocket, with streamed markdown, tool-call cards, and the tool-approval box |

Supporting modules: `api.ts` (REST calls), `chatSocket.ts` (the `/ws/chat`
client with auto-reconnect and resume), `types.ts` (wire types shared with the
server), `dragState.ts` (drag fallback for paths `dataTransfer` can strip), and
`components/SettingsDrawer.tsx` / `components/ViewerSettings.tsx`.

## Commands

Run these from the repository root (they wrap the same npm scripts):

```bash
just workspace-build   # npm install && npm run build  → frontend/dist
just workspace-dev     # npm run dev — hot reload, proxies /api and /ws to :8100
```

Or directly in this directory:

```bash
npm install
npm run dev      # dev server on :5173, proxied to the workspace server on :8100
npm run build    # tsc -b && vite build
npm run lint     # eslint
```

`npm run dev` only serves the UI — start the backend separately with
`just workspace` so the proxied `/api` and `/ws` routes resolve.

## Conventions

- **TypeScript strict**, plus the `react-hooks` v6 ESLint rules. `npm run build`
  runs `tsc -b` first, so type errors fail the build. Both run in CI.
- **Design language** mirrors the project site: navy surfaces with brand blue
  `#2B4FA3` / red `#D22229`. Fonts are Inter and JetBrains Mono, bundled via
  `@fontsource` — never load a font from a CDN, since MedMCP must run fully
  offline.
- **Attribution**: `rollup-plugin-license` emits `dist/LICENSES.txt` covering
  every dependency in the minified bundle. See `THIRD_PARTY_NOTICES.md` at the
  repository root.
