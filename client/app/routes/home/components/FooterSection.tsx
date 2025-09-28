import { SUPPORTED_LANGUAGES } from "../constants";
import type { FooterSectionProps } from "../types";

export function FooterSection({ messages, language }: FooterSectionProps) {
  return (
    <footer className="border-t border-slate-200 bg-white/70 px-4 py-6 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-950/90 dark:text-slate-400">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-4">
          <a
            href="/privacy"
            className="hover:text-slate-700 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:hover:text-slate-200"
            aria-label={messages.privacyLinkAria}
          >
            {messages.footerPrivacy}
          </a>
          <a
            href="/legal"
            className="hover:text-slate-700 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:hover:text-slate-200"
          >
            {messages.footerLegal}
          </a>
          <span>
            {messages.footerLanguage}: {SUPPORTED_LANGUAGES.find((lang) => lang.code === language)?.nativeName}
          </span>
        </div>
        <p className="text-xs">{messages.footerCopyright}</p>
      </div>
    </footer>
  );
}
