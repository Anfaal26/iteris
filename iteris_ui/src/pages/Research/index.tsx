/**
 * Research showcase (spec §5) — sticky scroll-spy TOC over the nine
 * academic sections. Sections with no evaluation data yet render an
 * honest "pending" state rather than placeholder numbers.
 */
import React, { useEffect, useRef, useState } from 'react';
import { Navbar } from '@/components';
import { ROUTES } from '@/routes';
import { AbstractSection } from './sections/AbstractSection';
import { DatasetsSection } from './sections/DatasetsSection';
import { MethodsSection } from './sections/MethodsSection';
import { ModelsSection } from './sections/ModelsSection';
import { ResultsTableSection } from './sections/ResultsTableSection';
import { ConvergenceSection } from './sections/ConvergenceSection';
import { AblationSection } from './sections/AblationSection';
import { QualitativeGridSection } from './sections/QualitativeGridSection';
import { CitationSection } from './sections/CitationSection';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

interface TocEntry {
  id: string;
  label: string;
}

const TOC: TocEntry[] = [
  { id: 'abstract', label: 'Abstract' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'methods', label: 'Methods' },
  { id: 'models', label: 'Models' },
  { id: 'results', label: 'Results' },
  { id: 'convergence', label: 'Convergence' },
  { id: 'ablations', label: 'Ablations' },
  { id: 'figures', label: 'Figures' },
  { id: 'citation', label: 'Citation' },
];

/** Sticky in-page nav with IntersectionObserver-driven active-section highlight. */
const TableOfContents: React.FC<{ activeId: string }> = ({ activeId }) => (
  <nav
    aria-label="Research page sections"
    className="hidden lg:block sticky self-start w-44 shrink-0"
    style={{ top: 'calc(var(--navbar-height) + 32px)' }}
  >
    <ul className="space-y-1 list-none pl-0 border-l border-border">
      {TOC.map((entry) => (
        <li key={entry.id}>
          <a
            href={`#${entry.id}`}
            aria-current={activeId === entry.id ? 'true' : undefined}
            className={[
              'block pl-4 -ml-px py-1.5 text-xs font-body border-l-2 no-underline',
              'transition-colors duration-panel ease-out',
              activeId === entry.id
                ? 'border-l-accent text-accent font-medium'
                : 'border-l-transparent text-muted hover:text-text',
            ].join(' ')}
          >
            {entry.label}
          </a>
        </li>
      ))}
    </ul>
  </nav>
);

export default function Research() {
  const [activeId, setActiveId] = useState<string>('abstract');
  const tickingRef = useRef(false);

  useEffect(() => {
    // Scroll-listener + getBoundingClientRect rather than IntersectionObserver:
    // whichever section's top is closest to (but above) the "active line" —
    // navbar height plus a small margin — is the active one.
    const ACTIVE_LINE = 96;

    const computeActive = () => {
      tickingRef.current = false;
      let current = TOC[0].id;
      for (const entry of TOC) {
        const el = document.getElementById(entry.id);
        if (el && el.getBoundingClientRect().top <= ACTIVE_LINE) {
          current = entry.id;
        }
      }
      setActiveId(current);
    };

    // setTimeout rather than requestAnimationFrame: rAF is suspended in
    // backgrounded/hidden tabs, which would silently freeze the active
    // section on the last-foreground value.
    const onScroll = () => {
      if (!tickingRef.current) {
        tickingRef.current = true;
        window.setTimeout(computeActive, 100);
      }
    };

    computeActive();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  return (
    <div className="min-h-screen bg-bg font-body text-text">
      <Navbar variant="light" navItems={NAV_ITEMS} />

      <main
        className="mx-auto max-w-6xl px-6 pb-24 flex gap-12"
        style={{ marginTop: 'var(--navbar-height)' }}
      >
        <TableOfContents activeId={activeId} />

        <div className="min-w-0 flex-1 max-w-3xl divide-y divide-border">
          <AbstractSection />
          <DatasetsSection />
          <MethodsSection />
          <ModelsSection />
          <ResultsTableSection />
          <ConvergenceSection />
          <AblationSection />
          <QualitativeGridSection />
          <CitationSection />
        </div>
      </main>
    </div>
  );
}
