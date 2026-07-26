"use client"

import { motion } from "framer-motion"
import { Upload, MessagesSquare, Workflow, Database, BarChart3, Download } from "lucide-react"

const steps = [
  {
    icon: Upload,
    step: "01",
    label: "Upload a Dataset",
    desc: "CSV, Parquet, or Excel up to 200 MB. The profiler runs immediately — column types, missing values, cardinality, and data-quality warnings.",
  },
  {
    icon: MessagesSquare,
    step: "02",
    label: "Describe the Business Problem",
    desc: "Plain language. \"Predict which customers will churn next month.\" The platform infers problem type, target column, and columns to exclude.",
  },
  {
    icon: Workflow,
    step: "03",
    label: "Planner Generates an ExecutionPlan",
    desc: "A Planner Agent selects capabilities from a registry, orders them by requires/produces contracts, and a DAG validator checks soundness before anything runs.",
  },
  {
    icon: Database,
    step: "04",
    label: "Agents Execute the Pipeline",
    desc: "EDA → Cleaning → Feature Engineering → Training (LR, RF, XGBoost, LightGBM) → Evaluation → SHAP Explainability. Checkpointed after every node.",
  },
  {
    icon: BarChart3,
    step: "05",
    label: "Review Results",
    desc: "Model leaderboard, held-out metrics, SHAP feature importances, and plain-language business recommendations — all in one structured report.",
  },
  {
    icon: Download,
    step: "06",
    label: "Download Model + Report",
    desc: "A production-ready .joblib pipeline and a Markdown report. The model was fitted on all data; evaluation used a clean held-out split.",
  },
]

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
}

const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
}

export function HowItWorksSection() {
  return (
    <section className="py-28 px-4 bg-bg" id="how-it-works">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">How It Works</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            From raw data to business insight
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            Six deterministic steps. Every agent decision is grounded in tool output — no ad-hoc code generation.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {steps.map((s) => (
            <motion.div
              key={s.step}
              variants={item}
              className="group relative rounded-xl border border-border bg-surface p-6 hover:border-accent/40 transition-colors duration-200"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
                  <s.icon className="w-4 h-4" />
                </div>
                <span className="font-mono text-xs text-border">{s.step}</span>
              </div>
              <h3 className="text-sm font-semibold text-text-primary mb-2">{s.label}</h3>
              <p className="text-xs text-text-muted leading-relaxed">{s.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
