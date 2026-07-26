"use client"

import { motion } from "framer-motion"
import { BarChart3, Download } from "lucide-react"
import { LiquidGlassButton } from "@/components/ui/liquid-glass-button"

// Mock SHAP data matching the real explanation_report shape from backend
const topFeatures = [
  { feature: "monthly_charges",    importance: 0.312, direction: "increases" },
  { feature: "tenure",             importance: 0.287, direction: "decreases" },
  { feature: "contract_month",     importance: 0.198, direction: "increases" },
  { feature: "total_charges",      importance: 0.143, direction: "increases" },
  { feature: "tech_support_no",    importance: 0.089, direction: "increases" },
  { feature: "online_security_no", importance: 0.071, direction: "increases" },
  { feature: "payment_electronic", importance: 0.058, direction: "increases" },
]

const recommendations = [
  { title: "Target high monthly-charge customers", confidence: "high",   detail: "Customers paying >$70/month churn at 2.4× the baseline rate. Offer loyalty discounts at the 6-month mark." },
  { title: "Prioritise contract upgrades",          confidence: "high",   detail: "Month-to-month contracts are the single strongest churn predictor. A one-month free incentive to switch to annual reduces churn risk by ~34%." },
  { title: "Expand tech-support coverage",          confidence: "medium", detail: "Absence of tech support is the 5th-ranked driver. Proactive outreach to customers without it reduces early-tenure churn." },
]

const confidenceColor: Record<string, string> = {
  high:   "text-success border-success/30 bg-success/10",
  medium: "text-warning border-warning/30 bg-warning/10",
  low:    "text-text-muted border-border bg-surface",
}

const maxImportance = topFeatures[0].importance

export function ExplainabilitySection() {
  return (
    <section className="py-28 px-4 bg-bg" id="explainability">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">Output Preview</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Explainability &amp; Report
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            Every run produces SHAP feature importances, plain-language business recommendations, and a downloadable model + Markdown report.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* SHAP chart */}
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4 }}
            className="rounded-xl border border-border bg-surface p-6"
          >
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="w-4 h-4 text-accent" />
              <span className="font-mono text-xs text-text-muted uppercase tracking-widest">SHAP Feature Importances</span>
              <span className="ml-auto font-mono text-[10px] text-text-muted border border-border rounded px-1.5 py-0.5">tree_shap</span>
            </div>

            <div className="space-y-3">
              {topFeatures.map((f, i) => (
                <motion.div
                  key={f.feature}
                  initial={{ opacity: 0, x: -8 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: i * 0.05 }}
                  className="flex items-center gap-3"
                >
                  <span className="font-mono text-xs text-text-muted w-36 truncate flex-shrink-0">{f.feature}</span>
                  <div className="flex-1 h-5 rounded bg-bg overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(f.importance / maxImportance) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.6, delay: 0.2 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                      className="h-full rounded"
                      style={{
                        background: f.direction === "increases"
                          ? "rgba(79,107,255,0.7)"
                          : "rgba(34,211,238,0.7)",
                      }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-text-muted w-12 text-right">{f.importance.toFixed(3)}</span>
                  <span className={`font-mono text-[10px] w-16 text-right ${f.direction === "increases" ? "text-accent" : "text-accent-cyan"}`}>
                    {f.direction === "increases" ? "↑" : "↓"}
                  </span>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 flex gap-4 text-[10px] font-mono text-text-muted">
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-accent/70" />increases prediction</span>
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-accent-cyan/70" />decreases prediction</span>
            </div>
          </motion.div>

          {/* Recommendations */}
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4 }}
            className="rounded-xl border border-border bg-surface p-6 flex flex-col"
          >
            <div className="flex items-center gap-2 mb-6">
              <span className="font-mono text-xs text-text-muted uppercase tracking-widest">Business Recommendations</span>
            </div>

            <div className="space-y-4 flex-1">
              {recommendations.map((r, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: i * 0.08 }}
                  className="rounded-lg border border-border bg-bg p-4"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="text-sm font-medium text-text-primary">{r.title}</span>
                    <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded border flex-shrink-0 ${confidenceColor[r.confidence]}`}>
                      {r.confidence}
                    </span>
                  </div>
                  <p className="text-xs text-text-muted leading-relaxed">{r.detail}</p>
                </motion.div>
              ))}
            </div>

            <div className="mt-6 flex gap-3">
              <LiquidGlassButton size="sm" className="flex-1 justify-center">
                <Download className="w-3.5 h-3.5" />
                Download Model
              </LiquidGlassButton>
              <LiquidGlassButton variant="glass" size="sm" className="flex-1 justify-center">
                <Download className="w-3.5 h-3.5" />
                Download Report
              </LiquidGlassButton>
            </div>
          </motion.div>
        </div>

        {/* metrics strip */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4"
        >
          {[
            { label: "accuracy",  value: "0.8214" },
            { label: "precision", value: "0.7891" },
            { label: "recall",    value: "0.6543" },
            { label: "roc_auc",   value: "0.8762" },
          ].map(m => (
            <div key={m.label} className="rounded-lg border border-border bg-surface p-4 text-center">
              <div className="font-mono text-xl font-semibold text-accent">{m.value}</div>
              <div className="mt-1 font-mono text-xs text-text-muted">{m.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
