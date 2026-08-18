import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import TopNav from "@/components/TopNav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Smart Table · medical-rag",
  description:
    "Life sciences market intelligence — comparative clinical trial grids grounded strictly in the retrieved corpus.",
};

// Runs before React hydrates so the page never paints the wrong theme then
// flips -- reads localStorage synchronously and sets the class TopNav's
// dark: variants (and globals.css's .dark tokens) key off of. Falls back to
// the OS preference on first visit, same as prefers-color-scheme would.
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem("theme");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      // The theme-init script below deliberately mutates this element's
      // classList (adding "dark") before React hydrates, so the live DOM
      // and the server-rendered className React expects to see will
      // legitimately differ on the `dark` token -- this is the documented
      // escape hatch for exactly that pattern (next-themes and shadcn/ui
      // templates use the same script + suppressHydrationWarning combo),
      // not a real mismatch to fix.
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col bg-slate-100 dark:bg-slate-950">
        <TopNav />
        {children}
      </body>
    </html>
  );
}
