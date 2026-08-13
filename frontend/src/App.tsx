import { useCallback, useEffect, useRef, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Chat } from './components/Chat'
import { FileExplorer } from './components/FileExplorer'
import { StackMarketplace } from './components/StackMarketplace'
import { SettingsDrawer } from './components/SettingsDrawer'
import { Viewer } from './components/Viewer'
import { WorkflowPanel } from './components/WorkflowPanel'
import { GearIcon, StoreIcon } from './components/icons'

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
  const [marketOpen, setMarketOpen] = useState(false)
  // The vibe session that received the last prompt — what "Save chat as
  // workflow" distills. Survives a reconnect (which starts an empty session).
  const [distillSessionId, setDistillSessionId] = useState<string | null>(null)
  // Bumped whenever something may have written to the workspace (agent tool
  // calls, replay steps) so the explorer/viewer reload their file tree.
  const [fsVersion, setFsVersion] = useState(0)
  const notifyFsChanged = useCallback(() => setFsVersion((v) => v + 1), [])
  // True while a separator is being dragged. The viewer is the only WebGL panel;
  // its canvas is a large GPU layer that's expensive to composite every frame, so
  // we drop it (via the `is-resizing` class) for the whole drag and let it redraw
  // once on release.
  //
  // The flag must stay true for the ENTIRE pointer drag. We can't clear it on the
  // library's `onLayoutChanged`: that fires on layout "settles" mid-drag (e.g.
  // when you pause), which would un-hide the canvas and make it lag again — the
  // exact "fine, then janky" symptom. So we set it on `onLayoutChange` (per move)
  // and clear it on the real pointer release, with a timed fallback for keyboard
  // resizes that have no pointerup.
  const [resizing, setResizing] = useState(false)
  const resizeEndTimer = useRef<number | null>(null)
  const startResize = useCallback(() => {
    setResizing(true)
    if (resizeEndTimer.current != null) clearTimeout(resizeEndTimer.current)
    resizeEndTimer.current = window.setTimeout(() => setResizing(false), 400)
  }, [])
  useEffect(() => {
    const end = () => {
      if (resizeEndTimer.current != null) {
        clearTimeout(resizeEndTimer.current)
        resizeEndTimer.current = null
      }
      setResizing(false)
    }
    window.addEventListener('pointerup', end, true)
    window.addEventListener('pointercancel', end, true)
    return () => {
      window.removeEventListener('pointerup', end, true)
      window.removeEventListener('pointercancel', end, true)
    }
  }, [])
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
  const openSession = useCallback((id: string) => {
    setResumeId(id)
    setChatKey((k) => k + 1)
  }, [])

  return (
    <div className={`app-shell${resizing ? ' is-resizing' : ''}`}>
      <header className="app-header">
        <span className="app-logo">MedMCP</span>
        <span className="app-subtitle">workspace</span>
        <button
          className="btn-icon app-header-gear"
          title="Tool stacks"
          onClick={() => setMarketOpen(true)}
        >
          <StoreIcon />
        </button>
        <button className="btn-icon" title="Settings" onClick={() => setSettingsOpen(true)}>
          <GearIcon />
        </button>
      </header>
      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <StackMarketplace open={marketOpen} onClose={() => setMarketOpen(false)} />
      <Group
        orientation="vertical"
        className="app-main"
        resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
        onLayoutChange={startResize}
      >
        <Panel defaultSize="60%" minSize="20%">
          <Group
            orientation="horizontal"
            resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
            onLayoutChange={startResize}
          >
            <Panel defaultSize="25%" minSize="12%">
              <FileExplorer
                onOpenFile={setOpenPath}
                refreshSignal={fsVersion}
                onSelectionChange={setSelectedPaths}
                isResizing={resizing}
              />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Viewer path={openPath} isResizing={resizing} />
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
                onOpenFile={setOpenPath}
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
                currentSessionId={resumeId}
                onSelectSession={openSession}
              />
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  )
}
