"use client"

import { motion } from "framer-motion"
import { Upload, Github } from "lucide-react"
import { LiquidGlassButton } from "@/components/ui/liquid-glass-button"

export function CTASection() {
  return (
    <section className="py-28 px-4 bg-bg" id="cta">
      <div className="max-w-2xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
        >
          <p className="font-mono text-xs text-accent tracking-widest uppercase mb-4">Get Early Access</p>
          <h2 className="text-3xl sm:text-4xl font-semibold text-text-primary tracking-tight mb-4">
            Ready to run your first pipeline?
          </h2>
          <p className="text-text-muted mb-10 max-w-md mx-auto">
            Phase 5 is in progress. Drop your email to be notified when the platform opens, or explore the backend on GitHub.
          </p>

          <form
            className="flex flex-col sm:flex-row gap-3 justify-center mb-8"
            onSubmit={e => e.preventDefault()}
          >
            <input
              type="email"
              placeholder="you@company.com"
              className="flex-1 max-w-xs rounded-full border border-border bg-surface px-5 py-3 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent transition-colors"
            />
            <LiquidGlassButton type="submit">
              <Upload className="w-4 h-4" />
              Get Early Access
            </LiquidGlassButton>
          </form>

          <div className="flex items-center justify-center gap-6 text-sm text-text-muted">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 hover:text-text-primary transition-colors"
            >
              <Github className="w-4 h-4" />
              View on GitHub
            </a>
            <span className="text-border">·</span>
            <a href="/docs" className="hover:text-text-primary transition-colors font-mono text-xs">
              API Docs →
            </a>
          </div>
        </motion.div>
      </div>

      {/* footer */}
      <div className="mt-24 border-t border-border pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 max-w-5xl mx-auto">
        <span className="font-mono text-xs text-text-muted">AI Data Science Platform — Phase 5 in progress</span>
        <div className="flex gap-6 text-xs text-text-muted font-mono">
          <a href="#how-it-works"  className="hover:text-text-primary transition-colors">How It Works</a>
          <a href="#architecture"  className="hover:text-text-primary transition-colors">Architecture</a>
          <a href="#capabilities"  className="hover:text-text-primary transition-colors">Capabilities</a>
          <a href="#roadmap"       className="hover:text-text-primary transition-colors">Roadmap</a>
        </div>
      </div>
    </section>
  )
}
