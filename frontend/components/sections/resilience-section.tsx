"use client"

import { motion } from "framer-motion"
import { RotateCcw, CheckCircle2, AlertCircle } from "lucide-react"

const timeline = [
  { label: "profile",        status: "done",    desc: "Dataset profiled" },
  { label: "planner",        status: "done",    desc: "ExecutionPlan generated & validated" },
  { label: "problem_spec",   status: "done",    desc: "Problem type + target decided" },
  { label: "cleaning",       status: "done",    desc: "Checkpoint saved ✓" },
  { label: "feature_plan",   status: "failed",  desc: "LLM timeout — plan_error recorded" },
  { label: "planner",        status: "replan",  desc: "Replan 1/2 — retrying from cursor" },
  { label: "feature_plan",   status: "done",    desc: "Retry succeeded — checkpoint saved ✓" },
  { label: "features",       status: "done",    desc: "Checkpoint saved ✓" },
  { label: "training",       status: "done",    desc: "Best model: XGBoost — checkpoint saved ✓" },
  { label: "evaluation",     status: "done",    desc: "Held-out metrics recorded" },
  { label: "explain",        status: "done",    desc: "SHAP importances computed" },
  { label: "recommendations",status: "done",    desc: "Business recommendations generated" },
  { label: "report",         status: "done",    desc: "report.md written — run complete" },
]

const statusColor: Record<string, string> = {
  done:   "text-success border-success/40 bg-success/10",
  failed: "text-red-400 border-red-400/40 bg-red-400/10",
  replan: "text-warning border-warning/40 bg-warning/10",
}

export function ResilienceSection() {
  return (
    <section className="py-28 px-4 bg-surface" id="resilience">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">Resilience</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Checkpoint, fail, resume
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            State is checkpointed to SQLite after every node. A capability failure triggers a bounded replan (up to 2 retries) before surfacing to human input — the run never silently disappears.
          </p>
        </motion.div>

        <div className="relative">
          {/* vertical line */}
          <div className="absolute left-[18px] top-0 bottom-0 w-px bg-border" />

          <div className="space-y-1">
            {timeline.map((t, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: i * 0.04 }}
                className="flex items-start gap-4 pl-10 relative"
              >
                {/* dot */}
                <div className={`absolute left-[10px] top-3 h-[18px] w-[18px] rounded-full border flex items-center justify-center ${statusColor[t.status]}`}>
                  {t.status === "done"   && <CheckCircle2 className="w-2.5 h-2.5" />}
                  {t.status === "failed" && <AlertCircle  className="w-2.5 h-2.5" />}
                  {t.status === "replan" && <RotateCcw    className="w-2.5 h-2.5" />}
                </div>

                <div className="py-2.5 flex items-center gap-3 flex-wrap">
                  <span className="font-mono text-xs text-text-primary">{t.label}</span>
                  <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded border ${statusColor[t.status]}`}>
                    {t.status}
                  </span>
                  <span className="text-xs text-text-muted">{t.desc}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="mt-10 grid grid-cols-3 gap-4 text-center">
          {[
            { label: "Plan attempts", value: "3" },
            { label: "Runtime replans", value: "2" },
            { label: "Fallback", value: "needs_input" },
          ].map(s => (
            <div key={s.label} className="rounded-lg border border-border bg-bg p-4">
              <div className="font-mono text-xl font-semibold text-accent">{s.value}</div>
              <div className="mt-1 text-xs text-text-muted">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
