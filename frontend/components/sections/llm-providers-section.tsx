"use client"

import { motion } from "framer-motion"

const providers = [
  { name: "Claude (Anthropic)", tag: "anthropic/[REDACTED].5", default: true },
  { name: "OpenAI GPT-4o",      tag: "openai/gpt-4o",               default: false },
  { name: "Gemini 1.5 Pro",     tag: "google/gemini-pro-1.5",       default: false },
  { name: "Mistral Large",      tag: "mistralai/mistral-large",     default: false },
]

export function LLMProvidersSection() {
  return (
    <section className="py-28 px-4 bg-bg" id="providers">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-16 text-center"
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-3">LLM Abstraction</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight">
            Provider-agnostic by design
          </h2>
          <p className="mt-4 text-text-muted max-w-xl mx-auto">
            Agents receive an <code className="font-mono text-xs text-accent-cyan">LLMClient</code> through dependency injection and never construct one themselves. Swap providers in <code className="font-mono text-xs text-accent-cyan">.env</code> — no agent code changes.
          </p>
        </motion.div>

        <div className="rounded-xl border border-border overflow-hidden">
          <div className="grid grid-cols-[1fr_auto_auto] gap-4 px-5 py-3 bg-surface border-b border-border text-xs font-mono text-text-muted uppercase tracking-widest">
            <span>provider</span>
            <span>model id</span>
            <span>default</span>
          </div>
          {providers.map((p, i) => (
            <motion.div
              key={p.tag}
              initial={{ opacity: 0, x: -8 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: i * 0.06 }}
              className={`grid grid-cols-[1fr_auto_auto] gap-4 px-5 py-4 border-b border-border last:border-0 items-center ${i % 2 === 0 ? "bg-bg" : "bg-surface/30"}`}
            >
              <span className="text-sm text-text-primary font-medium">{p.name}</span>
              <span className="font-mono text-xs text-text-muted">{p.tag}</span>
              {p.default
                ? <span className="font-mono text-[10px] px-2 py-0.5 rounded-full border border-success/30 bg-success/10 text-success">default</span>
                : <span className="font-mono text-[10px] text-border">—</span>
              }
            </motion.div>
          ))}
        </div>

        <div className="mt-6 rounded-lg border border-border bg-surface p-4 font-mono text-xs text-text-muted">
          <span className="text-border"># backend/.env</span><br />
          <span className="text-accent-cyan">APP_LLM_PROVIDER</span>=<span className="text-success">openrouter</span><br />
          <span className="text-accent-cyan">APP_LLM_MODEL</span>=<span className="text-success">anthropic/[REDACTED].5</span><br />
          <span className="text-accent-cyan">APP_OPENROUTER_API_KEY</span>=<span className="text-warning">sk-or-...</span>
        </div>
      </div>
    </section>
  )
}
