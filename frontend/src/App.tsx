import { useCallback, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Chat } from './components/Chat'
import { FileExplorer } from './components/FileExplorer'
import { SettingsDrawer } from './components/SettingsDrawer'
import { Viewer } from './components/Viewer'
import { WorkflowPanel } from './components/WorkflowPanel'
import { GearIcon } from './components/icons'

/** localStorage key holding the last active chat session id (for auto-resume). */
const ACTIVE_SESSION_KEY = 'medmcp.activeSession'

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
  // Chat session continuity: on load resume the last session (auto), and let
  // the user start a fresh one. `resumeId` is what the next Chat mount should
  // resume (stored id, or null for new); bumping `chatKey` remounts <Chat/> so
  // it drops its socket and reconnects with that target. Chat captures the
  // resume id at mount, so updating it here only takes effect on the next mount.
  const [resumeId, setResumeId] = useState<string | null>(() =>
    localStorage.getItem(ACTIVE_SESSION_KEY),
  )
  const [chatKey, setChatKey] = useState(0)
  const handleSessionEstablished = useCallback((id: string) => {
    localStorage.setItem(ACTIVE_SESSION_KEY, id)
    setResumeId(id)
  }, [])
  const startNewChat = useCallback(() => {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
    setResumeId(null)
    setChatKey((k) => k + 1)
  }, [])

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
                key={chatKey}
                onPromptedSession={setDistillSessionId}
                viewedPath={openPath}
                onToolActivity={notifyFsChanged}
                resumeSessionId={resumeId}
                onSessionEstablished={handleSessionEstablished}
                onNewChat={startNewChat}
              />
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  )
}
