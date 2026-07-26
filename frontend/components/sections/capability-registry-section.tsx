"use client"

import { motion } from "framer-motion"

// Real capability contracts from backend/app/planner/registry.py
const capabilities = [
  { name: "insights",        requires: ["profile"],                                                    produces: ["insights"],           llm: true  },
  { name: "problem_spec",    requires: ["problem_text", "profile"],                                    produces: ["problem_spec"],        llm: true  },
  { name: "cleaning",        requires: ["problem_spec"],                                               produces: ["cleaning_report"],     llm: false },
  { name: "feature_plan",    requires: ["cleaning_report", "problem_spec"],                            produces: ["feature_plan"],        llm: true  },
  { name: "features",        requires: ["feature_plan", "problem_spec", "cleaning_report"],            produces: ["feature_report"],      llm: false },
  { name: "training",        requires: ["feature_report", "problem_spec"],                             produces: ["training_report"],     llm: false },
  { name: "evaluation",      requires: ["feature_report", "training_report", "problem_spec"],          produces: ["evaluation_report"],   llm: false },
  { name: "explain",         requires: ["training_report", "feature_report", "problem_spec"],          produces: ["explanation_report"],  llm: false },
  { name: "recommendations", requires: ["problem_spec", "evaluation_report", "explanation_report"],    produces: ["recommendations"],     llm: true  },
  { name: "report",          requires: ["profile","problem_spec","cleaning_report","feature_report","training_report","evaluation_report","explanation_report","recommendations"], produces: ["report"], llm: false },
  { name: "summarize",       requires: ["problem_spec","cleaning_report","feature_report","training_report","evaluation_report"],         produces: ["summary"],            llm: false },
]

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] } },
}

export function CapabilityRegistrySection() {
  return (
    <section className="py-28 px-4 bg-bg" id="capabilities">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">Capability Registry</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Every step is a contract
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            The planner can only select registered capabilities. A DAG validator checks <code className="font-mono text-xs text-accent-cyan">requires</code> / <code className="font-mono text-xs text-accent-cyan">produces</code> soundness before execution begins.
          </p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          transition={{ staggerChildren: 0.05 }}
          className="rounded-xl border border-border overflow-hidden"
        >
          {/* header */}
          <div className="grid grid-cols-[1fr_2fr_1fr_auto] gap-4 px-5 py-3 bg-surface border-b border-border text-xs font-mono text-text-muted uppercase tracking-widest">
            <span>capability</span>
            <span>requires</span>
            <span>produces</span>
            <span>llm</span>
          </div>

          {capabilities.map((cap, i) => (
            <motion.div
              key={cap.name}
              variants={item}
              className={`grid grid-cols-[1fr_2fr_1fr_auto] gap-4 px-5 py-3.5 border-b border-border last:border-0 hover:bg-surface/60 transition-colors duration-150 ${i % 2 === 0 ? "bg-bg" : "bg-surface/30"}`}
            >
              <span className="font-mono text-sm text-text-primary">{cap.name}</span>
              <div className="flex flex-wrap gap-1.5">
                {cap.requires.map(r => (
                  <span key={r} className="font-mono text-xs px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20">{r}</span>
                ))}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {cap.produces.map(p => (
                  <span key={p} className="font-mono text-xs px-1.5 py-0.5 rounded bg-success/10 text-success border border-success/20">{p}</span>
                ))}
              </div>
              <span className={`font-mono text-xs ${cap.llm ? "text-warning" : "text-text-muted"}`}>
                {cap.llm ? "yes" : "no"}
              </span>
            </motion.div>
          ))}
        </motion.div>

        <p className="mt-4 text-xs font-mono text-text-muted text-center">
          Bootstrap artifacts always present: <span className="text-accent-cyan">run_id · dataset_id · problem_text · profile</span>
        </p>
      </div>
    </section>
  )
}
