import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#08090A",
        surface: "#131518",
        "surface-2": "#17191C",
        border: "#27272A",
        accent: "#4F6BFF",
        "accent-cyan": "#22D3EE",
        success: "#10B981",
        warning: "#F59E0B",
        "text-primary": "#EDEDEF",
        "text-muted": "#8A8F98",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
