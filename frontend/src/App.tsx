import {
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import './App.css'

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

type ProofNode = {
  id: string
  edgeAction: string | null
  observation: string
  goals: Goal[]
  terminal: boolean
  expanded: boolean
  proven: boolean
  children: ProofNode[]
}

type Snapshot = {
  sessionId: string
  root: ProofNode
  complete: boolean
  proofScript: string | null
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

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok || data.error) {
    throw new Error(data.error ?? `Request failed with ${response.status}`)
  }
  return data as T
}

function collectEdgePairs(root: ProofNode): EdgePair[] {
  const pairs: EdgePair[] = []

  function visit(node: ProofNode) {
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
    Boolean(target.closest('input, textarea, button, .modal-panel'))
  )
}

function App() {
  const [theorem, setTheorem] = useState('')
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [startError, setStartError] = useState('')
  const [nodeErrors, setNodeErrors] = useState<Record<string, string>>({})
  const [busyNode, setBusyNode] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [edges, setEdges] = useState<Edge[]>([])
  const [treeSize, setTreeSize] = useState({ width: 0, height: 0 })

  const treeRef = useRef<HTMLDivElement | null>(null)
  const cardRefs = useRef(new Map<string, HTMLDivElement>())
  const dragRef = useRef<{
    pointerId: number
    startX: number
    startY: number
    panX: number
    panY: number
  } | null>(null)

  const edgePairs = useMemo(
    () => (snapshot ? collectEdgePairs(snapshot.root) : []),
    [snapshot],
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
        const startX = fromRect.left - treeRect.left + fromRect.width / 2
        const startY = fromRect.bottom - treeRect.top
        const endX = toRect.left - treeRect.left + toRect.width / 2
        const endY = toRect.top - treeRect.top
        const midY = startY + Math.max(48, (endY - startY) * 0.52)

        return [
          {
            from: pair.from,
            to: pair.to,
            path: `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`,
          },
        ]
      })

      setEdges(nextEdges)
      setTreeSize({
        width: tree.scrollWidth,
        height: tree.scrollHeight,
      })
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
  }, [edgePairs, snapshot])

  const registerCard = (id: string) => (element: HTMLDivElement | null) => {
    if (element) {
      cardRefs.current.set(id, element)
    } else {
      cardRefs.current.delete(id)
    }
  }

  async function startProof() {
    if (!theorem.trim()) {
      setStartError('Enter a theorem.')
      return
    }

    setStarting(true)
    setStartError('')
    try {
      const data = await postJson<Snapshot>('/start', { theorem })
      setSnapshot(data)
      setNodeErrors({})
      setPan({ x: 0, y: 0 })
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error))
    } finally {
      setStarting(false)
    }
  }

  async function submitAction(nodeId: string, action: string) {
    if (!snapshot) {
      return
    }

    setBusyNode(nodeId)
    setNodeErrors((errors) => ({ ...errors, [nodeId]: '' }))
    try {
      const data = await postJson<Snapshot>('/step', {
        sessionId: snapshot.sessionId,
        nodeId,
        action,
      })
      setSnapshot(data)
    } catch (error) {
      setNodeErrors((errors) => ({
        ...errors,
        [nodeId]: error instanceof Error ? error.message : String(error),
      }))
    } finally {
      setBusyNode(null)
    }
  }

  async function resetProof() {
    if (snapshot) {
      await postJson('/reset', { sessionId: snapshot.sessionId }).catch(() => null)
    }
    setSnapshot(null)
    setNodeErrors({})
    setStartError('')
    setTheorem('')
  }

  function handleStartSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void startProof()
  }

  function handleTheoremKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void startProof()
    }
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
      panX: pan.x,
      panY: pan.y,
    }
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) {
      return
    }

    setPan({
      x: drag.panX + event.clientX - drag.startX,
      y: drag.panY + event.clientY - drag.startY,
    })
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null
    }
  }

  return (
    <main
      className="app-shell"
      style={{ backgroundPosition: `${pan.x}px ${pan.y}px` }}
    >
      <div
        className="tree-viewport"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <div
          className="tree-world"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px)` }}
        >
          {snapshot ? (
            <div className="proof-tree" ref={treeRef}>
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
              <TreeNode
                node={snapshot.root}
                busyNode={busyNode}
                error={nodeErrors}
                onAction={submitAction}
                registerCard={registerCard}
              />
            </div>
          ) : (
            <form className="start-panel" onSubmit={handleStartSubmit}>
              <textarea
                value={theorem}
                onChange={(event) => setTheorem(event.target.value)}
                onKeyDown={handleTheoremKeyDown}
                aria-label="Theorem"
                placeholder="theorem test_example (n : Nat) : n = n := by sorry"
                spellCheck={false}
              />
              <div className="start-actions">
                <button type="submit" disabled={starting}>
                  {starting ? 'Starting' : 'Start proof'}
                </button>
              </div>
              {startError && <p className="error-text">{startError}</p>}
            </form>
          )}
        </div>
      </div>

      {snapshot?.complete && snapshot.proofScript && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <section className="modal-panel">
            <div className="modal-heading">
              <p>Proof complete</p>
              <button type="button" onClick={resetProof}>
                New proof
              </button>
            </div>
            <pre className="proof-script">{snapshot.proofScript}</pre>
          </section>
        </div>
      )}
    </main>
  )
}

function TreeNode({
  node,
  busyNode,
  error,
  onAction,
  registerCard,
}: {
  node: ProofNode
  busyNode: string | null
  error: Record<string, string>
  onAction: (nodeId: string, action: string) => Promise<void>
  registerCard: (id: string) => (element: HTMLDivElement | null) => void
}) {
  return (
    <div className="tree-node">
      <section
        ref={registerCard(node.id)}
        className={[
          'proof-card',
          node.terminal ? 'terminal' : '',
          node.proven ? 'proven' : '',
        ].join(' ')}
      >
        {node.edgeAction && <pre className="action-chip">{node.edgeAction}</pre>}
        <LeanState node={node} />
        {!node.terminal && !node.expanded && (
          <ActionInput
            disabled={busyNode === node.id}
            error={error[node.id]}
            onSubmit={(action) => onAction(node.id, action)}
          />
        )}
      </section>

      {node.children.length > 0 && (
        <div className="children-row">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              busyNode={busyNode}
              error={error}
              onAction={onAction}
              registerCard={registerCard}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ActionInput({
  disabled,
  error,
  onSubmit,
}: {
  disabled: boolean
  error?: string
  onSubmit: (action: string) => Promise<void>
}) {
  const [action, setAction] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!action.trim()) {
      return
    }
    await onSubmit(action)
  }

  return (
    <form className="action-form" onSubmit={handleSubmit}>
      <input
        value={action}
        onChange={(event) => setAction(event.target.value)}
        aria-label="Tactic"
        placeholder="intro h"
        disabled={disabled}
        spellCheck={false}
      />
      <button type="submit" disabled={disabled || !action.trim()}>
        Run
      </button>
      {error && <p className="error-text">{error}</p>}
    </form>
  )
}

function LeanState({ node }: { node: ProofNode }) {
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
            <span className="turnstile">{'\u22a2'}</span>
            <span className="code-fragment">{goal.type}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default App
