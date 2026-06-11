import { useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Chat } from './components/Chat'
import { FileExplorer } from './components/FileExplorer'
import { Viewer } from './components/Viewer'

/** Three-panel workspace: explorer (top left), viewer (top right), chat (bottom). */
export default function App() {
  const [openPath, setOpenPath] = useState<string | null>(null)

  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-logo">MedMCP</span>
        <span className="app-subtitle">workspace</span>
      </header>
      <Group
        orientation="vertical"
        className="app-main"
        resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
      >
        <Panel defaultSize="60%" minSize="20%">
          <Group orientation="horizontal" resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}>
            <Panel defaultSize="25%" minSize="12%">
              <FileExplorer onOpenFile={setOpenPath} />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Viewer path={openPath} />
            </Panel>
          </Group>
        </Panel>
        <Separator className="sep sep-h" />
        <Panel minSize="15%">
          <Chat />
        </Panel>
      </Group>
    </div>
  )
}
