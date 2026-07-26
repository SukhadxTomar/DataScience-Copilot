"use client"

import { motion } from "framer-motion"
import { CheckCircle2, RotateCcw } from "lucide-react"

const phases = [
  { n: "01", label: "Backend Foundation",          desc: "Upload endpoint, dataset profiler, filesystem storage",                    status: "done"        },
  { n: "02", label: "LLM Abstraction + EDA Agent", desc: "Provider-independent LLM client, EDA insights agent, 3-node LangGraph",   status: "done"        },
  { n: "03", label: "ML Tool Agents",              desc: "Cleaning, feature engineering, training (4 models), evaluation tools",     status: "done"        },
  { n: "04", label: "Dynamic Planner",             desc: "Capability registry, DAG validator, planner agent, executor with retries", status: "done"        },
  { n: "05", label: "Explainability & Reporting",  desc: "SHAP importances, business recommendations, final report — frontend now", status: "in_progress" },
]

const statusStyle: Record<string, { icon: React.ReactNode; label: string; cls: string }> = {
  done:        { icon: <CheckCircle2 className="w-3.5 h-3.5" />, label: "Complete",     cls: "text-success border-success/30 bg-success/10" },
  in_progress: { icon: <RotateCcw    className="w-3.5 h-3.5" />, label: "In Progress",  cls: "text-warning border-warning/30 bg-warning/10" },
}

export function RoadmapSection() {
  return (
    <section className="py-28 px-4 bg-surface" id="roadmap">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">Build Roadmap</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Five phases to production
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            Phases 1–4 are complete. Phase 5 — explainability, recommendations, and this frontend — is in progress.
          </p>
        </motion.div>

        <div className="space-y-3">
          {phases.map((p, i) => {
            const s = statusStyle[p.status]
            return (
              <motion.div
                key={p.n}
                initial={{ opacity: 0, x: -16 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.35, delay: i * 0.07 }}
                className="flex items-start gap-5 rounded-xl border border-border bg-bg p-5 hover:border-accent/30 transition-colors duration-200"
              >
                <span className="font-mono text-2xl font-semibold text-border w-8 flex-shrink-0 mt-0.5">{p.n}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 flex-wrap mb-1">
                    <span className="font-semibold text-sm text-text-primary">{p.label}</span>
                    <span className={`inline-flex items-center gap-1.5 font-mono text-[10px] px-2 py-0.5 rounded-full border ${s.cls}`}>
                      {s.icon}{s.label}
                    </span>
                  </div>
                  <p className="text-xs text-text-muted">{p.desc}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
