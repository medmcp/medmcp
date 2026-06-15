import { useCallback, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Chat } from './components/Chat'
import { FileExplorer } from './components/FileExplorer'
import { SettingsDrawer } from './components/SettingsDrawer'
import { Viewer } from './components/Viewer'
import { WorkflowPanel } from './components/WorkflowPanel'
import { GearIcon } from './components/icons'

/**
 * Four-panel workspace: explorer (top left), viewer (top right),
 * workflows (bottom left), chat (bottom right).
 */
export default function App() {
  const [openPath, setOpenPath] = useState<string | null>(null)
  // Files multi-selected in the explorer — feeds the workflow batch editor.
  const [selectedPaths, setSelectedPaths] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  // The vibe session that received the last prompt — what "Save chat as
  // workflow" distills. Survives a reconnect (which starts an empty session).
  const [distillSessionId, setDistillSessionId] = useState<string | null>(null)
  // Bumped whenever something may have written to the workspace (agent tool
  // calls, replay steps) so the explorer/viewer reload their file tree.
  const [fsVersion, setFsVersion] = useState(0)
  const notifyFsChanged = useCallback(() => setFsVersion((v) => v + 1), [])
  // True while a separator that borders the viewer is being dragged. The viewer
  // is the only WebGL/iframe panel; compositing it every frame is what makes
  // such a drag feel heavy, so we hide its canvas (via the `is-resizing` class)
  // until the layout settles. onLayoutChange fires per pointer move (drag in
  // progress); onLayoutChanged fires once the drag ends.
  const [resizing, setResizing] = useState(false)
  const startResize = useCallback(() => setResizing(true), [])
  const endResize = useCallback(() => setResizing(false), [])

  return (
    <div className={`app-shell${resizing ? ' is-resizing' : ''}`}>
      <header className="app-header">
        <span className="app-logo">MedMCP</span>
        <span className="app-subtitle">workspace</span>
        <button
          className="btn-icon app-header-gear"
          title="Settings"
          onClick={() => setSettingsOpen(true)}
        >
          <GearIcon />
        </button>
      </header>
      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <Group
        orientation="vertical"
        className="app-main"
        resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
        onLayoutChange={startResize}
        onLayoutChanged={endResize}
      >
        <Panel defaultSize="60%" minSize="20%">
          <Group
            orientation="horizontal"
            resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
            onLayoutChange={startResize}
            onLayoutChanged={endResize}
          >
            <Panel defaultSize="25%" minSize="12%">
              <FileExplorer
                onOpenFile={setOpenPath}
                refreshSignal={fsVersion}
                onSelectionChange={setSelectedPaths}
              />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Viewer path={openPath} refreshSignal={fsVersion} />
            </Panel>
          </Group>
        </Panel>
        <Separator className="sep sep-h" />
        <Panel minSize="15%">
          <Group orientation="horizontal" resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}>
            <Panel defaultSize="25%" minSize="12%">
              <WorkflowPanel
                distillSessionId={distillSessionId}
                onWorkspaceChanged={notifyFsChanged}
                selectedPaths={selectedPaths}
              />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Chat
                onPromptedSession={setDistillSessionId}
                viewedPath={openPath}
                onToolActivity={notifyFsChanged}
              />
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  )
}
