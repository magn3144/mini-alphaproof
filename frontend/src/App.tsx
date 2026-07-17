import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import './App.css'

type Mode = 'manual' | 'agent'
type SessionStatus =
  | 'ready'
  | 'running'
  | 'verifying'
  | 'solved'
  | 'exhausted'
  | 'cancelled'
  | 'error'
type ActionStatus = 'pending' | 'running' | 'succeeded' | 'invalid'

type Hypothesis = {
  name: string
  type: string
  value: string | null
}

type Goal = {
  tag: string | null
  type: string
  hypotheses: Hypothesis[]
}

type StateNode = {
  kind: 'state'
  id: string
  nodeType: 'OR' | 'AND'
  branchLabel: string | null
  observation: string
  goals: Goal[]
  terminal: boolean
  proven: boolean
  children: TreeNode[]
}

type ActionNode = {
  kind: 'action'
  id: string
  action: string
  status: ActionStatus
  error: string | null
  proven: boolean
  children: TreeNode[]
}

type TreeNode = StateNode | ActionNode

type Snapshot = {
  sessionId: string
  mode: Mode
  status: SessionStatus
  error: string | null
  root: StateNode
  complete: boolean
  proofScript: string | null
}

type ViewState = {
  theorem: string
  snapshot: Snapshot | null
  error: string
  busyId: string | null
  pan: { x: number; y: number }
  zoom: number
  collapsed: string[]
  proofDismissed: boolean
}

type Edge = {
  from: string
  to: string
  path: string
}

type EdgePair = {
  from: string
  to: string
}

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'

const emptyView = (): ViewState => ({
  theorem: '',
  snapshot: null,
  error: '',
  busyId: null,
  pan: { x: 0, y: 0 },
  zoom: 1,
  collapsed: [],
  proofDismissed: false,
})

async function readResponse<T>(response: Response): Promise<T> {
  const data = await response.json()
  if (!response.ok || data.error) {
    throw new Error(data.error ?? `Request failed with ${response.status}`)
  }
  return data as T
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  return readResponse<T>(
    await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  )
}

async function getSnapshot(sessionId: string): Promise<Snapshot> {
  const query = new URLSearchParams({ sessionId })
  return readResponse<Snapshot>(await fetch(`${API_BASE}/state?${query}`))
}

function collectEdgePairs(root: TreeNode, collapsed: Set<string>): EdgePair[] {
  const pairs: EdgePair[] = []

  function visit(node: TreeNode) {
    if (collapsed.has(node.id)) {
      return
    }
    for (const child of node.children) {
      pairs.push({ from: node.id, to: child.id })
      visit(child)
    }
  }

  visit(root)
  return pairs
}

function targetIsControl(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    Boolean(
      target.closest(
        'input, textarea, button, .proof-card, .action-card, .modal-panel, .top-bar',
      ),
    )
  )
}

function targetIsNodeControl(target: EventTarget | null) {
  return (
    target instanceof HTMLElement && Boolean(target.closest('input, button'))
  )
}

function App() {
  const [mode, setMode] = useState<Mode>('manual')
  const [views, setViews] = useState<Record<Mode, ViewState>>({
    manual: emptyView(),
    agent: emptyView(),
  })
  const [startingMode, setStartingMode] = useState<Mode | null>(null)
  const [edges, setEdges] = useState<Edge[]>([])
  const [treeSize, setTreeSize] = useState({ width: 0, height: 0 })

  const view = views[mode]
  const snapshot = view.snapshot
  const collapsed = useMemo(() => new Set(view.collapsed), [view.collapsed])
  const agentSessionId = views.agent.snapshot?.sessionId
  const agentStatus = views.agent.snapshot?.status
  const treeRef = useRef<HTMLDivElement | null>(null)
  const cardRefs = useRef(new Map<string, HTMLDivElement>())
  const dragRef = useRef<{
    pointerId: number
    startX: number
    startY: number
    panX: number
    panY: number
  } | null>(null)

  const updateView = (
    targetMode: Mode,
    update: (current: ViewState) => ViewState,
  ) => {
    setViews((current) => ({
      ...current,
      [targetMode]: update(current[targetMode]),
    }))
  }

  useEffect(() => {
    if (
      !agentSessionId ||
      !agentStatus ||
      !['running', 'verifying'].includes(agentStatus)
    ) {
      return
    }

    let stopped = false
    const poll = async () => {
      try {
        const data = await getSnapshot(agentSessionId)
        if (!stopped) {
          updateView('agent', (current) => ({
            ...current,
            snapshot: data,
            error: '',
          }))
        }
      } catch (error) {
        if (!stopped) {
          updateView('agent', (current) => ({
            ...current,
            error: error instanceof Error ? error.message : String(error),
          }))
        }
      }
    }

    const timer = window.setInterval(() => void poll(), 300)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [agentSessionId, agentStatus])

  const edgePairs = useMemo(
    () => (snapshot ? collectEdgePairs(snapshot.root, collapsed) : []),
    [collapsed, snapshot],
  )

  useLayoutEffect(() => {
    const updateEdges = () => {
      const tree = treeRef.current
      if (!tree || !snapshot) {
        setEdges([])
        return
      }

      const treeRect = tree.getBoundingClientRect()
      const nextEdges = edgePairs.flatMap((pair) => {
        const from = cardRefs.current.get(pair.from)
        const to = cardRefs.current.get(pair.to)
        if (!from || !to) {
          return []
        }

        const fromRect = from.getBoundingClientRect()
        const toRect = to.getBoundingClientRect()
        const startX =
          (fromRect.left - treeRect.left + fromRect.width / 2) / view.zoom
        const startY = (fromRect.bottom - treeRect.top) / view.zoom
        const endX =
          (toRect.left - treeRect.left + toRect.width / 2) / view.zoom
        const endY = (toRect.top - treeRect.top) / view.zoom
        const midY = startY + Math.max(36, (endY - startY) * 0.52)
        return [
          {
            from: pair.from,
            to: pair.to,
            path: `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`,
          },
        ]
      })

      setEdges(nextEdges)
      setTreeSize({ width: tree.scrollWidth, height: tree.scrollHeight })
    }

    updateEdges()
    const observer = new ResizeObserver(updateEdges)
    if (treeRef.current) {
      observer.observe(treeRef.current)
    }
    for (const element of cardRefs.current.values()) {
      observer.observe(element)
    }
    window.addEventListener('resize', updateEdges)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', updateEdges)
    }
  }, [edgePairs, snapshot, view.zoom])

  const registerCard = (id: string) => (element: HTMLDivElement | null) => {
    if (element) {
      cardRefs.current.set(id, element)
    } else {
      cardRefs.current.delete(id)
    }
  }

  async function startProof(targetMode: Mode) {
    const target = views[targetMode]
    if (!target.theorem.trim()) {
      updateView(targetMode, (current) => ({
        ...current,
        error: 'Enter a theorem.',
      }))
      return
    }

    setStartingMode(targetMode)
    updateView(targetMode, (current) => ({ ...current, error: '' }))
    try {
      const data = await postJson<Snapshot>(`/${targetMode}/start`, {
        theorem: target.theorem,
      })
      updateView(targetMode, (current) => ({
        ...current,
        snapshot: data,
        error: '',
        busyId: null,
        pan: { x: 0, y: 0 },
        zoom: 1,
        collapsed: [],
        proofDismissed: false,
      }))
    } catch (error) {
      updateView(targetMode, (current) => ({
        ...current,
        error: error instanceof Error ? error.message : String(error),
      }))
    } finally {
      setStartingMode(null)
    }
  }

  async function mutateManual(
    path: string,
    busyId: string,
    payload: Record<string, string>,
  ) {
    const manual = views.manual.snapshot
    if (!manual) {
      return
    }
    updateView('manual', (current) => ({
      ...current,
      busyId,
      error: '',
    }))
    try {
      const data = await postJson<Snapshot>(path, {
        sessionId: manual.sessionId,
        ...payload,
      })
      updateView('manual', (current) => ({
        ...current,
        snapshot: data,
        busyId: null,
      }))
    } catch (error) {
      updateView('manual', (current) => ({
        ...current,
        busyId: null,
        error: error instanceof Error ? error.message : String(error),
      }))
    }
  }

  async function cancelAgent() {
    const agent = views.agent.snapshot
    if (!agent) {
      return
    }
    try {
      const data = await postJson<Snapshot>('/cancel', {
        sessionId: agent.sessionId,
      })
      updateView('agent', (current) => ({ ...current, snapshot: data }))
    } catch (error) {
      updateView('agent', (current) => ({
        ...current,
        error: error instanceof Error ? error.message : String(error),
      }))
    }
  }

  async function resetProof(targetMode: Mode) {
    const target = views[targetMode]
    if (target.snapshot) {
      try {
        await postJson('/reset', { sessionId: target.snapshot.sessionId })
      } catch (error) {
        updateView(targetMode, (current) => ({
          ...current,
          error: error instanceof Error ? error.message : String(error),
        }))
        return
      }
    }
    updateView(targetMode, () => emptyView())
  }

  function toggleNode(nodeId: string) {
    updateView(mode, (current) => ({
      ...current,
      collapsed: current.collapsed.includes(nodeId)
        ? current.collapsed.filter((id) => id !== nodeId)
        : [...current.collapsed, nodeId],
    }))
  }

  function dismissProof() {
    updateView(mode, (current) => ({ ...current, proofDismissed: true }))
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (targetIsControl(event.target)) {
      return
    }
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      panX: view.pan.x,
      panY: view.pan.y,
    }
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) {
      return
    }
    updateView(mode, (current) => ({
      ...current,
      pan: {
        x: drag.panX + event.clientX - drag.startX,
        y: drag.panY + event.clientY - drag.startY,
      },
    }))
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null
    }
  }

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    const pointerX = event.clientX - rect.left
    const pointerY = event.clientY - rect.top

    updateView(mode, (current) => {
      const zoom = Math.min(
        2,
        Math.max(0.35, current.zoom * Math.exp(-event.deltaY * 0.001)),
      )
      const worldX = (pointerX - current.pan.x) / current.zoom
      const worldY = (pointerY - current.pan.y) / current.zoom
      return {
        ...current,
        zoom,
        pan: {
          x: pointerX - worldX * zoom,
          y: pointerY - worldY * zoom,
        },
      }
    })
  }

  return (
    <main
      className="app-shell"
      style={{ backgroundPosition: `${view.pan.x}px ${view.pan.y}px` }}
    >
      <nav className="top-bar" aria-label="Proof mode">
        <div className="mode-tabs">
          {(['manual', 'agent'] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              className={mode === tab ? 'active' : ''}
              onClick={() => setMode(tab)}
            >
              {tab === 'manual' ? 'Manual' : 'Agent'}
              {views[tab].snapshot && <span className="session-dot" />}
            </button>
          ))}
        </div>
        {snapshot && (
          <div className={`session-status status-${snapshot.status}`}>
            <span>{statusLabel(snapshot.status)}</span>
            {mode === 'agent' &&
              ['running', 'verifying'].includes(snapshot.status) && (
                <button type="button" onClick={() => void cancelAgent()}>
                  Stop
                </button>
              )}
            {!['running', 'verifying'].includes(snapshot.status) && (
              <button type="button" onClick={() => void resetProof(mode)}>
                New theorem
              </button>
            )}
          </div>
        )}
      </nav>

      <div
        className="tree-viewport"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
      >
        <div
          className="tree-world"
          style={{
            transform: `translate(${view.pan.x}px, ${view.pan.y}px) scale(${view.zoom})`,
          }}
        >
          {snapshot ? (
            <div className="proof-tree" ref={treeRef} key={snapshot.sessionId}>
              <svg
                className="edge-layer"
                width={treeSize.width}
                height={treeSize.height}
                viewBox={`0 0 ${treeSize.width} ${treeSize.height}`}
                aria-hidden="true"
              >
                {edges.map((edge) => (
                  <path key={`${edge.from}-${edge.to}`} d={edge.path} />
                ))}
              </svg>
              <Tree
                node={snapshot.root}
                mode={mode}
                busyId={view.busyId}
                collapsed={collapsed}
                registerCard={registerCard}
                onToggle={toggleNode}
                onCreate={(nodeId) =>
                  mutateManual('/action/create', nodeId, { nodeId })
                }
                onUpdate={(actionId, action) =>
                  mutateManual('/action/update', actionId, {
                    actionId,
                    action,
                  })
                }
                onRun={(actionId) =>
                  mutateManual('/action/run', actionId, { actionId })
                }
              />
            </div>
          ) : (
            <StartPanel
              mode={mode}
              theorem={view.theorem}
              error={view.error}
              starting={startingMode === mode}
              onChange={(theorem) =>
                updateView(mode, (current) => ({ ...current, theorem }))
              }
              onStart={() => startProof(mode)}
            />
          )}
        </div>
      </div>

      {snapshot && (snapshot.error || view.error) && (
        <div className="floating-error">{snapshot.error ?? view.error}</div>
      )}

      {snapshot?.complete && snapshot.proofScript && !view.proofDismissed && (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal="true"
          onClick={dismissProof}
        >
          <section
            className="modal-panel"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-heading">
              <p>Proof complete</p>
              <button type="button" onClick={() => void resetProof(mode)}>
                New theorem
              </button>
            </div>
            <pre className="proof-script">{snapshot.proofScript}</pre>
          </section>
        </div>
      )}
    </main>
  )
}

function StartPanel({
  mode,
  theorem,
  error,
  starting,
  onChange,
  onStart,
}: {
  mode: Mode
  theorem: string
  error: string
  starting: boolean
  onChange: (theorem: string) => void
  onStart: () => Promise<void>
}) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void onStart()
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void onStart()
    }
  }

  return (
    <form className="start-panel" onSubmit={handleSubmit}>
      <div className="start-heading">
        <span>{mode === 'manual' ? 'Manual proof' : 'Agent search'}</span>
        <small>
          {mode === 'manual'
            ? 'Explore tactics and branches yourself.'
            : 'Watch AlphaProof build its search tree.'}
        </small>
      </div>
      <textarea
        value={theorem}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="Theorem"
        placeholder="theorem test_example (n : Nat) : n = n := by sorry"
        spellCheck={false}
      />
      <div className="start-actions">
        <button type="submit" disabled={starting}>
          {starting ? 'Starting' : mode === 'manual' ? 'Start proof' : 'Run agent'}
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}
    </form>
  )
}

function Tree({
  node,
  mode,
  busyId,
  collapsed,
  registerCard,
  onToggle,
  onCreate,
  onUpdate,
  onRun,
}: {
  node: TreeNode
  mode: Mode
  busyId: string | null
  collapsed: Set<string>
  registerCard: (id: string) => (element: HTMLDivElement | null) => void
  onToggle: (nodeId: string) => void
  onCreate: (nodeId: string) => Promise<void>
  onUpdate: (actionId: string, action: string) => Promise<void>
  onRun: (actionId: string) => Promise<void>
}) {
  const isCollapsed = collapsed.has(node.id)
  const collapsible = node.children.length > 0

  return (
    <div className="tree-node">
      {node.kind === 'state' ? (
        <StateCard
          node={node}
          mode={mode}
          busy={busyId === node.id}
          collapsed={isCollapsed}
          collapsible={collapsible}
          registerCard={registerCard}
          onCreate={onCreate}
          onToggle={onToggle}
        />
      ) : (
        <ActionCard
          node={node}
          mode={mode}
          busy={busyId === node.id}
          collapsed={isCollapsed}
          collapsible={collapsible}
          registerCard={registerCard}
          onUpdate={onUpdate}
          onRun={onRun}
          onToggle={onToggle}
        />
      )}

      {collapsible && !isCollapsed && (
        <div className="children-row">
          {node.children.map((child) => (
            <Tree
              key={child.id}
              node={child}
              mode={mode}
              busyId={busyId}
              collapsed={collapsed}
              registerCard={registerCard}
              onToggle={onToggle}
              onCreate={onCreate}
              onUpdate={onUpdate}
              onRun={onRun}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function StateCard({
  node,
  mode,
  busy,
  collapsed,
  collapsible,
  registerCard,
  onCreate,
  onToggle,
}: {
  node: StateNode
  mode: Mode
  busy: boolean
  collapsed: boolean
  collapsible: boolean
  registerCard: (id: string) => (element: HTMLDivElement | null) => void
  onCreate: (nodeId: string) => Promise<void>
  onToggle: (nodeId: string) => void
}) {
  return (
    <section
      ref={registerCard(node.id)}
      onClick={(event) => {
        if (collapsible && !targetIsNodeControl(event.target)) {
          onToggle(node.id)
        }
      }}
      className={[
        'proof-card',
        `node-${node.nodeType.toLowerCase()}`,
        node.terminal ? 'terminal' : '',
        node.proven ? 'proven' : '',
        collapsible ? 'collapsible' : '',
        collapsed ? 'collapsed' : '',
      ].join(' ')}
    >
      <div className="card-heading">
        <span className={`node-badge badge-${node.nodeType.toLowerCase()}`}>
          {node.nodeType}
        </span>
        {node.branchLabel && <span className="branch-label">{node.branchLabel}</span>}
        {collapsible && (
          <span className="collapse-indicator" aria-hidden="true">
            {collapsed ? `▸ ${node.children.length}` : '▾'}
          </span>
        )}
        {mode === 'manual' && node.nodeType === 'OR' && !node.terminal && (
          <button
            type="button"
            className="add-action"
            aria-label="Add tactic"
            title="Add tactic"
            disabled={busy}
            onClick={() => void onCreate(node.id)}
          >
            +
          </button>
        )}
      </div>
      <LeanState node={node} />
    </section>
  )
}

function ActionCard({
  node,
  mode,
  busy,
  collapsed,
  collapsible,
  registerCard,
  onUpdate,
  onRun,
  onToggle,
}: {
  node: ActionNode
  mode: Mode
  busy: boolean
  collapsed: boolean
  collapsible: boolean
  registerCard: (id: string) => (element: HTMLDivElement | null) => void
  onUpdate: (actionId: string, action: string) => Promise<void>
  onRun: (actionId: string) => Promise<void>
  onToggle: (nodeId: string) => void
}) {
  const [action, setAction] = useState(node.action)
  useEffect(() => setAction(node.action), [node.action])

  const editable = mode === 'manual' && ['pending', 'invalid'].includes(node.status)
  const changed = action.trim() !== node.action

  return (
    <section
      ref={registerCard(node.id)}
      onClick={(event) => {
        if (collapsible && !targetIsNodeControl(event.target)) {
          onToggle(node.id)
        }
      }}
      className={[
        'action-card',
        `action-${node.status}`,
        node.proven ? 'proven' : '',
        collapsible ? 'collapsible' : '',
        collapsed ? 'collapsed' : '',
      ].join(' ')}
    >
      <div className="action-heading">
        <span>Tactic</span>
        <span className="action-status">{node.status}</span>
        {collapsible && (
          <span className="collapse-indicator" aria-hidden="true">
            {collapsed ? `▸ ${node.children.length}` : '▾'}
          </span>
        )}
      </div>
      {editable ? (
        <div className="action-editor">
          <input
            value={action}
            onChange={(event) => setAction(event.target.value)}
            aria-label="Tactic"
            placeholder="intro h"
            disabled={busy}
            spellCheck={false}
          />
          <div className="action-buttons">
            <button
              type="button"
              className="secondary-button"
              disabled={busy || !changed}
              onClick={() => void onUpdate(node.id, action)}
            >
              Save
            </button>
            <button
              type="button"
              disabled={busy || changed || !node.action}
              onClick={() => void onRun(node.id)}
            >
              Run
            </button>
          </div>
        </div>
      ) : (
        <pre className="action-text">{node.action || 'Generating tactic…'}</pre>
      )}
      {node.error && <p className="error-text">{node.error}</p>}
    </section>
  )
}

function LeanState({ node }: { node: StateNode }) {
  if (node.terminal || node.goals.length === 0) {
    return <pre className="lean-state closed">{node.observation}</pre>
  }

  return (
    <div className="lean-state">
      {node.goals.map((goal, index) => (
        <div className="goal" key={`${goal.tag ?? 'goal'}-${index}`}>
          {goal.tag && <div className="goal-tag">case {goal.tag}</div>}
          {goal.hypotheses.map((hypothesis, hypothesisIndex) => (
            <div
              className="hypothesis"
              key={`${hypothesis.name}-${hypothesisIndex}`}
            >
              <span className="hypothesis-name">{hypothesis.name}</span>
              <span className="syntax">:</span>
              <span className="code-fragment">{hypothesis.type}</span>
              {hypothesis.value && (
                <>
                  <span className="syntax">:=</span>
                  <span className="code-fragment">{hypothesis.value}</span>
                </>
              )}
            </div>
          ))}
          <div className="target">
            <span className="turnstile">{'⊢'}</span>
            <span className="code-fragment">{goal.type}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function statusLabel(status: SessionStatus) {
  const labels: Record<SessionStatus, string> = {
    ready: 'Ready',
    running: 'Searching…',
    verifying: 'Verifying proof…',
    solved: 'Proof found',
    exhausted: 'Search exhausted',
    cancelled: 'Search stopped',
    error: 'Search failed',
  }
  return labels[status]
}

export default App
