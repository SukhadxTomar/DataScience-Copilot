"use client"

import { motion } from "framer-motion"
import { Download, RotateCcw, BarChart3, CheckCircle2, FileText } from "lucide-react"
import { LiquidGlassButton } from "@/components/ui/liquid-glass-button"

import { API_BASE } from "@/lib/api"

interface RunResult {
  run_id: string
  dataset_id: string
  problem_text: string
  status: string
  summary?: string
  training_report?: {
    best_model: string
    metric: string
    results: { model: string; cv_mean: number; cv_std: number }[]
  }
  evaluation_report?: { metrics: Record<string, number>; n_test_samples: number }
  explanation_report?: {
    method: string
    top_features: { feature: string; importance: number; direction?: string }[]
  }
  recommendations?: {
    narrative?: string
    recommendations: { title: string; detail: string; expected_impact: string; confidence: string }[]
  }
  error?: string
}

const confidenceColor: Record<string, string> = {
  high:   "text-success border-success/30 bg-success/10",
  medium: "text-warning border-warning/30 bg-warning/10",
  low:    "text-text-muted border-border bg-surface",
}

export function ReportView({ result, onReset }: { result: RunResult; onReset: () => void }) {
  const topFeatures = result.explanation_report?.top_features ?? []
  const maxImp = topFeatures[0]?.importance ?? 1

  function downloadFile(url: string, filename: string) {
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden flex flex-col bg-bg">
      {/* header bar */}
      <div className="flex-shrink-0 border-b border-border bg-surface px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-success" />
          <div>
            <h1 className="text-sm font-semibold text-text-primary">Your Report is Ready</h1>
            <p className="text-xs text-text-muted font-mono mt-0.5 truncate max-w-md">
              {result.problem_text || "No problem description provided"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <LiquidGlassButton
            variant="glass" size="sm"
            onClick={() => downloadFile(`${API_BASE}/api/runs/${result.run_id}/report`, `report_${result.run_id}.md`)}
          >
            <FileText className="w-3.5 h-3.5" />
            Download Report
          </LiquidGlassButton>
          <LiquidGlassButton
            size="sm"
            onClick={() => downloadFile(`${API_BASE}/api/runs/${result.run_id}/model`, `model_${result.run_id}.joblib`)}
          >
            <Download className="w-3.5 h-3.5" />
            Download Model
          </LiquidGlassButton>
          <LiquidGlassButton variant="outline" size="sm" onClick={onReset}>
            <RotateCcw className="w-3.5 h-3.5" />
            New Analysis
          </LiquidGlassButton>
        </div>
      </div>

      {/* scrollable body — only the report content scrolls, not the page */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-5xl mx-auto space-y-5">

          {/* summary */}
          {result.summary && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              className="rounded-xl border border-border bg-surface p-5">
              <p className="text-xs font-mono text-accent uppercase tracking-widest mb-2">Summary</p>
              <pre className="text-sm text-text-muted whitespace-pre-wrap font-mono leading-relaxed">{result.summary}</pre>
            </motion.div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* model leaderboard */}
            {result.training_report && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
                className="rounded-xl border border-border bg-surface p-5">
                <p className="text-xs font-mono text-accent uppercase tracking-widest mb-4">Model Leaderboard</p>
                <div className="space-y-2">
                  {result.training_report.results.map((r, i) => (
                    <div key={r.model} className={`flex items-center justify-between rounded-lg px-3 py-2 ${
                      r.model === result.training_report!.best_model ? "bg-accent/10 border border-accent/20" : "bg-bg"
                    }`}>
                      <div className="flex items-center gap-2">
                        {r.model === result.training_report!.best_model && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-accent flex-shrink-0" />
                        )}
                        <span className="font-mono text-xs text-text-primary">{r.model}</span>
                      </div>
                      <span className="font-mono text-xs text-text-muted">
                        {r.cv_mean.toFixed(4)} ± {r.cv_std.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-xs text-text-muted font-mono">
                  metric: {result.training_report.metric}
                </p>
              </motion.div>
            )}

            {/* evaluation metrics */}
            {result.evaluation_report && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                className="rounded-xl border border-border bg-surface p-5">
                <p className="text-xs font-mono text-accent uppercase tracking-widest mb-4">
                  Held-out Metrics
                  <span className="ml-2 text-text-muted normal-case">({result.evaluation_report.n_test_samples} samples)</span>
                </p>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(result.evaluation_report.metrics).map(([k, v]) => (
                    <div key={k} className="rounded-lg bg-bg border border-border p-3 text-center">
                      <div className="font-mono text-lg font-semibold text-accent">{v.toFixed(4)}</div>
                      <div className="font-mono text-xs text-text-muted mt-0.5">{k}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>

          {/* SHAP */}
          {topFeatures.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
              className="rounded-xl border border-border bg-surface p-5">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-4 h-4 text-accent" />
                <p className="text-xs font-mono text-accent uppercase tracking-widest">
                  Feature Importances
                </p>
                <span className="ml-auto font-mono text-[10px] text-text-muted border border-border rounded px-1.5 py-0.5">
                  {result.explanation_report!.method}
                </span>
              </div>
              <div className="space-y-2.5">
                {topFeatures.slice(0, 10).map((f) => (
                  <div key={f.feature} className="flex items-center gap-3">
                    <span className="font-mono text-xs text-text-muted w-40 truncate flex-shrink-0">{f.feature}</span>
                    <div className="flex-1 h-4 rounded bg-bg overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(f.importance / maxImp) * 100}%` }}
                        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                        className="h-full rounded"
                        style={{ background: f.direction === "decreases" ? "rgba(34,211,238,0.7)" : "rgba(79,107,255,0.7)" }}
                      />
                    </div>
                    <span className="font-mono text-[10px] text-text-muted w-14 text-right">{f.importance.toFixed(4)}</span>
                    {f.direction && (
                      <span className={`font-mono text-[10px] w-4 ${f.direction === "increases" ? "text-accent" : "text-accent-cyan"}`}>
                        {f.direction === "increases" ? "↑" : "↓"}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* recommendations */}
          {result.recommendations && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
              className="rounded-xl border border-border bg-surface p-5">
              <p className="text-xs font-mono text-accent uppercase tracking-widest mb-3">Recommendations</p>
              {result.recommendations.narrative && (
                <p className="text-sm text-text-muted mb-4 leading-relaxed">{result.recommendations.narrative}</p>
              )}
              <div className="space-y-3">
                {result.recommendations.recommendations.map((r, i) => (
                  <div key={i} className="rounded-lg border border-border bg-bg p-4">
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <span className="text-sm font-medium text-text-primary">{r.title}</span>
                      <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded border flex-shrink-0 ${confidenceColor[r.confidence] ?? confidenceColor.low}`}>
                        {r.confidence}
                      </span>
                    </div>
                    <p className="text-xs text-text-muted leading-relaxed">{r.detail}</p>
                    <p className="text-xs text-text-muted mt-1 italic">Impact: {r.expected_impact}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* error fallback */}
          {result.error && (
            <div className="rounded-xl border border-red-400/30 bg-red-400/10 p-5">
              <p className="text-xs font-mono text-red-400 uppercase tracking-widest mb-2">Pipeline Error</p>
              <p className="text-sm text-red-300 font-mono">{result.error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
