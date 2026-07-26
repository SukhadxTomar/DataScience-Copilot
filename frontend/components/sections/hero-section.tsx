"use client"

import { useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, MessagesSquare, Workflow, Database, Download, X, FileText, CheckCircle2, Loader2 } from "lucide-react"
import { WebGLShader } from "@/components/ui/web-gl-shader"
import { LiquidGlassButton } from "@/components/ui/liquid-glass-button"
import { API_BASE, parseJSON } from "@/lib/api"
import type { RunResult } from "@/app/page"

const steps = [
  { icon: Upload,         label: "Upload Dataset" },
  { icon: MessagesSquare, label: "Describe Problem" },
  { icon: Workflow,       label: "Plan & Validate" },
  { icon: Database,       label: "Agents Execute" },
  { icon: Download,       label: "Report + Model" },
]

type UploadState = "idle" | "dragging" | "uploading" | "done" | "error"
type RunState    = "idle" | "running" | "error"

export function HeroSection({ onComplete }: { onComplete: (r: RunResult) => void }) {
  const fileRef = useRef<HTMLInputElement>(null)

  const [showUpload,  setShowUpload]  = useState(false)
  const [showProblem, setShowProblem] = useState(false)

  const [datasetId,   setDatasetId]   = useState<string | null>(null)
  const [fileName,    setFileName]    = useState<string | null>(null)
  const [problem,     setProblem]     = useState("")

  const [uploadState, setUploadState] = useState<UploadState>("idle")
  const [uploadError, setUploadError] = useState("")

  const [runState,    setRunState]    = useState<RunState>("idle")
  const [runError,    setRunError]    = useState("")

  const bothReady = !!datasetId && problem.trim().length > 0

  const doUpload = useCallback(async (file: File) => {
    setUploadState("uploading")
    setUploadError("")
    const form = new FormData()
    form.append("file", file)
    try {
      const { dataset_id } = await parseJSON<{ dataset_id: string }>(
        await fetch(`${API_BASE}/api/datasets/upload`, { method: "POST", body: form })
      )
      setDatasetId(dataset_id)
      setFileName(file.name)
      setUploadState("done")
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : String(e))
      setUploadState("error")
    }
  }, [])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (f) doUpload(f)
  }
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]; if (f) doUpload(f)
  }, [doUpload])

  const handleRun = async () => {
    if (!bothReady) return
    setRunState("running")
    setRunError("")
    try {
      const data = await parseJSON<RunResult>(
        await fetch(`${API_BASE}/api/runs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: datasetId, problem_text: problem }),
        })
      )
      if (data.status === "failed" || data.status === "needs_input") {
        throw new Error(data.error ?? `Run ended with status: ${data.status}`)
      }
      onComplete(data)
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : String(e))
      setRunState("error")
    }
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden flex flex-col items-center justify-center">
      <WebGLShader />
      <div className="absolute inset-0 bg-bg/72 z-[1]" />

      <div className="relative z-[2] flex flex-col items-center text-center px-4 max-w-3xl w-full mx-auto">
        <motion.h1
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight text-text-primary leading-[1.1] mb-5"
        >
          Upload a Dataset.<br />
          <span className="text-accent">Get a Data Scientist.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="text-base sm:text-lg text-text-muted max-w-xl mb-10"
        >
          Upload a dataset, tell us what you want to solve, and receive insights,
          predictions, visualizations, and a complete AI-generated report.
        </motion.p>

        {/* two entry buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-wrap gap-3 justify-center mb-4"
        >
          <LiquidGlassButton size="lg"
            onClick={() => { setShowUpload(true); setShowProblem(false) }}
            className={datasetId ? "border-success/60 shadow-[0_0_20px_rgba(16,185,129,0.2)]" : ""}
          >
            {datasetId
              ? <><CheckCircle2 className="w-4 h-4 text-success" /><span className="text-success">Dataset Ready</span></>
              : <><Upload className="w-4 h-4" />Upload Dataset</>}
          </LiquidGlassButton>

          <LiquidGlassButton variant="glass" size="lg"
            onClick={() => { setShowProblem(true); setShowUpload(false) }}
            className={problem.trim() ? "border-success/60 shadow-[0_0_20px_rgba(16,185,129,0.2)]" : ""}
          >
            {problem.trim()
              ? <><CheckCircle2 className="w-4 h-4 text-success" /><span className="text-success">Problem Set</span></>
              : <><MessagesSquare className="w-4 h-4" />Describe Problem</>}
          </LiquidGlassButton>
        </motion.div>

        {/* generate / status */}
        <AnimatePresence mode="wait">
          {bothReady && runState === "idle" && (
            <motion.div key="run"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }} transition={{ duration: 0.25 }}
            >
              <LiquidGlassButton size="lg" onClick={handleRun}
                className="shadow-[0_0_30px_rgba(79,107,255,0.45)]">
                <Workflow className="w-4 h-4" />Generate Report
              </LiquidGlassButton>
            </motion.div>
          )}
          {runState === "running" && (
            <motion.div key="running" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex items-center gap-2 text-sm text-text-muted font-mono">
              <Loader2 className="w-4 h-4 animate-spin text-accent" />
              Pipeline running — this may take a minute…
            </motion.div>
          )}
          {runState === "error" && (
            <motion.div key="error" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              className="max-w-md rounded-xl border border-red-400/30 bg-red-400/10 px-5 py-3 text-sm font-mono text-red-400">
              {runError}
              <button className="ml-3 underline text-xs" onClick={() => setRunState("idle")}>retry</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* step strip */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-12 flex flex-wrap justify-center gap-x-5 gap-y-2"
        >
          {steps.map((s, i) => (
            <div key={i} className="flex items-center gap-1.5 text-xs text-text-muted font-mono">
              <s.icon className="w-3.5 h-3.5 text-accent" />
              {s.label}
              {i < steps.length - 1 && <span className="ml-2 text-border">→</span>}
            </div>
          ))}
        </motion.div>
      </div>

      {/* Upload modal */}
      <AnimatePresence>
        {showUpload && (
          <Modal onClose={() => setShowUpload(false)} title="Upload Dataset">
            <input ref={fileRef} type="file" accept=".csv,.parquet,.xlsx,.xls"
              className="hidden" onChange={onFileChange} />
            {uploadState !== "done" ? (
              <div
                onDragOver={e => { e.preventDefault(); setUploadState("dragging") }}
                onDragLeave={() => setUploadState(s => s === "dragging" ? "idle" : s)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
                className={`cursor-pointer rounded-xl border-2 border-dashed p-10 flex flex-col items-center gap-3 transition-colors duration-200
                  ${uploadState === "dragging" ? "border-accent bg-accent/10" : "border-border hover:border-accent/50 hover:bg-surface"}`}
              >
                {uploadState === "uploading"
                  ? <Loader2 className="w-8 h-8 text-accent animate-spin" />
                  : <Upload className="w-8 h-8 text-text-muted" />}
                <p className="text-sm text-text-primary font-medium">
                  {uploadState === "uploading" ? "Uploading…" : "Drag & drop or click to browse"}
                </p>
                <p className="text-xs text-text-muted font-mono">CSV · Parquet · Excel — max 200 MB</p>
                {uploadState === "error" && <p className="text-xs text-red-400">{uploadError}</p>}
              </div>
            ) : (
              <div className="rounded-xl border border-success/30 bg-success/10 p-6 flex items-center gap-4">
                <FileText className="w-8 h-8 text-success flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-success">Upload successful</p>
                  <p className="text-xs text-text-muted font-mono mt-0.5">{fileName}</p>
                  <p className="text-xs text-text-muted font-mono">id: {datasetId}</p>
                </div>
              </div>
            )}
            <div className="mt-4 flex justify-end gap-2">
              {uploadState === "done" && (
                <LiquidGlassButton size="sm" onClick={() => setShowUpload(false)}>Done</LiquidGlassButton>
              )}
              {uploadState === "error" && (
                <LiquidGlassButton variant="glass" size="sm"
                  onClick={() => { setUploadState("idle"); setUploadError("") }}>Try again</LiquidGlassButton>
              )}
            </div>
          </Modal>
        )}
      </AnimatePresence>

      {/* Problem modal */}
      <AnimatePresence>
        {showProblem && (
          <Modal onClose={() => setShowProblem(false)} title="Describe the Business Problem">
            <p className="text-xs text-text-muted mb-3">
              Tell the planner what you want to predict or understand. Be specific — the more context, the better the plan.
            </p>
            <textarea
              value={problem}
              onChange={e => setProblem(e.target.value)}
              rows={5}
              placeholder="e.g. Predict which customers will churn next month based on usage and billing data."
              className="w-full rounded-lg border border-border bg-bg px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent resize-none transition-colors"
            />
            <div className="mt-4 flex justify-end gap-2">
              <LiquidGlassButton variant="ghost" size="sm" onClick={() => setShowProblem(false)}>Cancel</LiquidGlassButton>
              <LiquidGlassButton size="sm" onClick={() => setShowProblem(false)}
                disabled={!problem.trim()} className="disabled:opacity-40 disabled:pointer-events-none">
                Confirm
              </LiquidGlassButton>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}

function Modal({ children, onClose, title }: { children: React.ReactNode; onClose: () => void; title: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 12 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-sm font-semibold text-text-primary">{title}</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
        {children}
      </motion.div>
    </motion.div>
  )
}
