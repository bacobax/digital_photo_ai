import type { GuidelinesSectionProps } from "../types";

export function GuidelinesSection({ messages }: GuidelinesSectionProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
        {messages.guidelinesHeading}
      </h3>
      <ul className="mt-4 space-y-3">
        {messages.guidelines.map((item) => (
          <li key={item} className="flex items-start gap-3 text-sm text-slate-600 dark:text-slate-300">
            <span className="mt-0.5 inline-flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-sky-500 text-white">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-3 w-3"
              >
                <path d="M5 10l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
      <a
        href="/help"
        target="_blank"
        rel="noreferrer"
        className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-sky-600 transition hover:text-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:text-sky-400 dark:hover:text-sky-300"
        aria-label={messages.helpLinkHint}
      >
        <span>{messages.tipsHeading}</span>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          className="h-4 w-4"
        >
          <path d="M5 5h10v10" strokeLinecap="round" />
          <path d="M5 15 15 5" strokeLinecap="round" />
        </svg>
      </a>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{messages.helpLinkHint}</p>
    </section>
  );
}
