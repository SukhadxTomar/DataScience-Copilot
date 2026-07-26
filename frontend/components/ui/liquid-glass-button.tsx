"use client"

import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-content gap-2 rounded-full font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-accent text-white hover:bg-accent/90 shadow-[0_0_20px_rgba(79,107,255,0.35)] hover:shadow-[0_0_30px_rgba(79,107,255,0.55)]",
        glass:
          "bg-white/5 border border-white/10 text-text-primary backdrop-blur-md hover:bg-white/10 hover:border-white/20",
        outline:
          "border border-border text-text-primary hover:border-accent hover:text-accent",
        ghost: "text-text-muted hover:text-text-primary hover:bg-surface",
      },
      size: {
        sm: "px-4 py-2 text-sm",
        default: "px-6 py-3 text-sm",
        lg: "px-8 py-4 text-base",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export function LiquidGlassButton({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button"
  return (
    <Comp className={cn(buttonVariants({ variant, size, className }))} {...props} />
  )
}
