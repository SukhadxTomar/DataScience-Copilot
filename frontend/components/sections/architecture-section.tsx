"use client"

import { useRef } from "react"
import { motion, useInView } from "framer-motion"

const nodes = [
  { id: "profile",     label: "Profile",     x: 50,  y: 50,  color: "#4F6BFF" },
  { id: "planner",     label: "Planner",     x: 200, y: 50,  color: "#4F6BFF" },
  { id: "insights",    label: "Insights",    x: 350, y: 10,  color: "#22D3EE" },
  { id: "problem",     label: "Problem Spec",x: 350, y: 50,  color: "#22D3EE" },
  { id: "cleaning",    label: "Cleaning",    x: 500, y: 30,  color: "#22D3EE" },
  { id: "features",    label: "Features",    x: 500, y: 70,  color: "#22D3EE" },
  { id: "training",    label: "Training",    x: 650, y: 30,  color: "#22D3EE" },
  { id: "evaluation",  label: "Evaluation",  x: 650, y: 70,  color: "#22D3EE" },
  { id: "explain",     label: "SHAP",        x: 800, y: 30,  color: "#10B981" },
  { id: "recommend",   label: "Recommend",   x: 800, y: 70,  color: "#10B981" },
  { id: "report",      label: "Report",      x: 950, y: 50,  color: "#10B981" },
]

const edges = [
  ["profile","planner"],
  ["planner","insights"],["planner","problem"],
  ["problem","cleaning"],["problem","features"],
  ["cleaning","features"],
  ["features","training"],["features","evaluation"],
  ["training","evaluation"],["training","explain"],
  ["evaluation","explain"],["evaluation","recommend"],
  ["explain","report"],["recommend","report"],
]

function getNode(id: string) { return nodes.find(n => n.id === id)! }

export function ArchitectureDiagramSection() {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section className="py-28 px-4 bg-surface" id="architecture" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">Architecture</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Live Pipeline Graph
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            Static LangGraph nodes, dynamic execution plan. The planner selects capabilities at runtime; the validator checks the DAG before anything runs.
          </p>
        </motion.div>

        {/* Desktop SVG diagram */}
        <div className="hidden md:block overflow-x-auto rounded-xl border border-border bg-bg p-6">
          <svg viewBox="0 0 1050 110" className="w-full" style={{ minWidth: 700 }}>
            {/* edges */}
            {edges.map(([a, b], i) => {
              const na = getNode(a), nb = getNode(b)
              return (
                <motion.line
                  key={i}
                  x1={na.x * 1} y1={na.y} x2={nb.x * 1} y2={nb.y}
                  stroke="#27272A" strokeWidth="1"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={inView ? { pathLength: 1, opacity: 1 } : {}}
                  transition={{ duration: 0.6, delay: 0.3 + i * 0.04 }}
                />
              )
            })}
            {/* pulse dots along edges */}
            {inView && edges.map(([a, b], i) => {
              const na = getNode(a), nb = getNode(b)
              return (
                <motion.circle
                  key={`pulse-${i}`}
                  r="2.5" fill="#4F6BFF"
                  initial={{ opacity: 0 }}
                  animate={{
                    opacity: [0, 1, 0],
                    cx: [na.x, nb.x],
                    cy: [na.y, nb.y],
                  }}
                  transition={{
                    duration: 1.5,
                    delay: 1 + i * 0.15,
                    repeat: Infinity,
                    repeatDelay: 3,
                    ease: "linear",
                  }}
                />
              )
            })}
            {/* nodes */}
            {nodes.map((n, i) => (
              <motion.g key={n.id}
                initial={{ opacity: 0, scale: 0.5 }}
                animate={inView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.3, delay: 0.1 + i * 0.06 }}
                style={{ transformOrigin: `${n.x}px ${n.y}px` }}
              >
                <circle cx={n.x} cy={n.y} r="14" fill="#131518" stroke={n.color} strokeWidth="1.5" />
                <text x={n.x} y={n.y + 24} textAnchor="middle" fill="#8A8F98" fontSize="7" fontFamily="monospace">
                  {n.label}
                </text>
              </motion.g>
            ))}
          </svg>
        </div>

        {/* Mobile fallback */}
        <div className="md:hidden grid grid-cols-2 gap-3">
          {nodes.map((n) => (
            <div key={n.id} className="flex items-center gap-2 rounded-lg border border-border bg-bg px-3 py-2">
              <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ background: n.color }} />
              <span className="font-mono text-xs text-text-muted">{n.label}</span>
            </div>
          ))}
        </div>

        {/* legend */}
        <div className="mt-6 flex flex-wrap gap-6 justify-center text-xs font-mono text-text-muted">
          <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-accent" />Bootstrap / Planner</span>
          <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-accent-cyan" />ML Capabilities</span>
          <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-success" />Explainability / Output</span>
        </div>
      </div>
    </section>
  )
}
