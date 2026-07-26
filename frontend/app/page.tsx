"use client"

import { useState } from "react"
import { HeroSection } from "@/components/sections/hero-section"
import { ReportView } from "@/components/sections/report-view"

export interface RunResult {
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

export default function Home() {
  const [result, setResult] = useState<RunResult | null>(null)

  if (result) {
    return <ReportView result={result} onReset={() => setResult(null)} />
  }

  return (
    <div className="h-screen w-screen overflow-hidden">
      <HeroSection onComplete={setResult} />
    </div>
  )
}
