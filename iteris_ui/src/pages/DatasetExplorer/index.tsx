/**
 * Dataset Explorer — CAMUS + BRISC curated strips and filterable image grids (spec §9).
 * All data from api.samples() (mock or live). Clicking a tile routes to /workspace.
 */

import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Navbar, SampleImageTile } from '@/components';
import type { SampleImage, Difficulty } from '@/api/contract';
import { api } from '@/api/client';
import { ROUTES } from '@/routes';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

// ── Curated highlight IDs from samples.yaml ─────────────────────────────────

const CAMUS_HIGHLIGHTS: { id: string; label: string; note?: string }[] = [
  { id: 'camus-a2c', label: 'Best absolute Dice', note: 'A2C view — highest overall score' },
  { id: 'camus-a4c', label: 'Best wipe demo', note: 'A4C — most dramatic boundary correction' },
  { id: 'camus-edes', label: 'Largest DRL improvement', note: 'ED/ES frame — best Δ Dice from baseline' },
  { id: 'camus-a4c', label: 'Honest near-failure', note: 'LA boundary hard to refine — DRL ≈ baseline' },
];

const BRISC_HIGHLIGHTS: { id: string; label: string; note?: string }[] = [
  { id: 'brisc-glioma', label: 'Best glioma result', note: 'Highest Dice for diffuse glioma' },
  { id: 'brisc-meningioma', label: 'Meningioma boundary refinement', note: 'Sharp capsule → precise contour' },
  { id: 'brisc-pituitary', label: 'Pituitary — most precise', note: 'Smallest structure, tightest HD' },
  { id: 'brisc-glioma', label: 'Honest glioma challenge', note: 'Diffuse infiltration — inherent uncertainty' },
];

// ── Filter bar ────────────────────────────────────────────────────────────────

type TumorType = 'glioma' | 'meningioma' | 'pituitary' | 'non-tumorous';
type SortMode = 'hardest' | 'best-dice';

interface CamusFilters {
  difficulty: Difficulty | 'all';
  sort: SortMode;
}

interface BriscFilters {
  difficulty: Difficulty | 'all';
  tumorType: TumorType | 'all';
  sort: SortMode;
}

const DIFFICULTY_OPTS: (Difficulty | 'all')[] = ['all', 'easy', 'medium', 'hard'];
const TUMOR_TYPE_OPTS: (TumorType | 'all')[] = ['all', 'glioma', 'meningioma', 'pituitary', 'non-tumorous'];
const SORT_OPTS: { label: string; value: SortMode }[] = [
  { label: 'Hardest first', value: 'hardest' },
  { label: 'Best Dice', value: 'best-dice' },
];

const DIFFICULTY_BADGE: Record<Difficulty, string> = {
  easy: 'bg-success/10 text-success',
  medium: 'bg-warning/10 text-warning',
  hard: 'bg-error/10 text-error',
};

/** Small pill filter button */
function FilterPill<T extends string>({
  value, active, label, onClick,
}: { value: T; active: boolean; label: string; onClick: (v: T) => void }) {
  return (
    <button
      type="button"
      onClick={() => onClick(value)}
      className={[
        'px-3 py-1 rounded-full text-xs font-body border transition-colors duration-panel ease-out',
        active
          ? 'bg-accent text-white border-accent'
          : 'bg-surface text-muted border-border hover:border-accent/60 hover:text-text',
      ].join(' ')}
    >
      {label}
    </button>
  );
}

// ── Curated strip ─────────────────────────────────────────────────────────────

interface CuratedStripProps {
  highlights: { id: string; label: string; note?: string }[];
  samplesById: Record<string, SampleImage>;
  onSelect: (id: string) => void;
}

function CuratedStrip({ highlights, samplesById, onSelect }: CuratedStripProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {highlights.map((h, i) => {
        const sample = samplesById[h.id];
        if (!sample) return null;
        return (
          <div key={`${h.id}-${i}`} className="flex flex-col gap-1">
            <SampleImageTile image={sample} onSelect={onSelect} />
            <p className="text-xs font-heading font-semibold text-text">{h.label}</p>
            {h.note && <p className="text-[11px] font-body text-muted leading-snug">{h.note}</p>}
          </div>
        );
      })}
    </div>
  );
}

// ── Section headings ──────────────────────────────────────────────────────────

function SectionHeading({ dataset, modality, count }: { dataset: string; modality: string; count: number }) {
  return (
    <div className="flex items-baseline gap-3 mb-6">
      <h2 className="font-heading text-2xl font-bold text-text">{dataset}</h2>
      <span className="rounded-full bg-accent/10 px-3 py-0.5 text-xs font-mono text-accent uppercase tracking-wide">
        {modality}
      </span>
      <span className="text-xs font-body text-muted">{count} samples</span>
    </div>
  );
}

// ── Dataset section ───────────────────────────────────────────────────────────

interface DatasetSectionProps {
  id: string;
  title: string;
  modality: string;
  samples: SampleImage[];
  highlights: { id: string; label: string; note?: string }[];
  samplesById: Record<string, SampleImage>;
  filters: React.ReactNode;
  onSelect: (id: string) => void;
}

function DatasetSection({ id, title, modality, samples, highlights, samplesById, filters, onSelect }: DatasetSectionProps) {
  return (
    <section id={id} className="py-12 border-t border-border first:border-t-0">
      <SectionHeading dataset={title} modality={modality} count={samples.length} />

      {/* Curated highlights */}
      <div className="mb-8">
        <p className="text-xs font-mono uppercase tracking-widest text-accent mb-4">Curated examples</p>
        <CuratedStrip highlights={highlights} samplesById={samplesById} onSelect={onSelect} />
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2 mb-6 py-3 sticky top-navbar bg-bg/90 backdrop-blur z-10">
        {filters}
      </div>

      {/* Sample grid */}
      {samples.length > 0 ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {samples.map((s) => (
            <SampleImageTile key={s.id} image={s} onSelect={onSelect} />
          ))}
        </div>
      ) : (
        <p className="py-12 text-center text-sm font-body text-muted">No samples match the current filters.</p>
      )}
    </section>
  );
}

// ── Difficulty badge legend ───────────────────────────────────────────────────

function DifficultyLegend() {
  return (
    <div className="flex items-center gap-3 text-xs font-body text-muted mb-8">
      <span className="font-semibold text-text">Difficulty:</span>
      {(['easy', 'medium', 'hard'] as Difficulty[]).map((d) => (
        <span key={d} className={`rounded-full px-2 py-0.5 capitalize ${DIFFICULTY_BADGE[d]}`}>{d}</span>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

/**
 * DatasetExplorer — two-section page for CAMUS and BRISC datasets.
 * Each section has curated highlights, a filter bar, and a sample grid.
 * Clicking any tile navigates to /workspace with that sample pre-selected.
 */
export default function DatasetExplorer() {
  const navigate = useNavigate();
  const [allSamples, setAllSamples] = useState<SampleImage[]>([]);

  const [camusFilters, setCamusFilters] = useState<CamusFilters>({
    difficulty: 'all',
    sort: 'best-dice',
  });
  const [briscFilters, setBriscFilters] = useState<BriscFilters>({
    difficulty: 'all',
    tumorType: 'all',
    sort: 'best-dice',
  });

  useEffect(() => {
    api.samples().then(setAllSamples).catch(() => {});
  }, []);

  const samplesById = useMemo(
    () => Object.fromEntries(allSamples.map((s) => [s.id, s])),
    [allSamples],
  );

  // Filter + sort helpers
  function applySort(list: SampleImage[], sort: SortMode) {
    return [...list].sort((a, b) =>
      sort === 'best-dice' ? b.bestDice - a.bestDice : a.bestDice - b.bestDice,
    );
  }

  const camusSamples = useMemo(() => {
    let list = allSamples.filter((s): s is SampleImage & { dataset: 'camus' } => s.dataset === 'camus');
    if (camusFilters.difficulty !== 'all') list = list.filter((s) => s.difficulty === camusFilters.difficulty);
    return applySort(list, camusFilters.sort);
  }, [allSamples, camusFilters]);

  const briscSamples = useMemo(() => {
    let list = allSamples.filter((s): s is SampleImage & { dataset: 'brisc' } => s.dataset === 'brisc');
    if (briscFilters.difficulty !== 'all') list = list.filter((s) => s.difficulty === briscFilters.difficulty);
    if (briscFilters.tumorType !== 'all') {
      list = list.filter((s) => s.anatomy.toLowerCase().includes(briscFilters.tumorType as string));
    }
    return applySort(list, briscFilters.sort);
  }, [allSamples, briscFilters]);

  function handleSelect(id: string) {
    const sample = samplesById[id];
    if (!sample) return;
    navigate(`${ROUTES.workspace}?dataset=${sample.dataset}&sampleId=${sample.id}`);
  }

  const camusFilterBar = (
    <>
      <span className="text-xs font-mono text-muted uppercase tracking-wide mr-1">Difficulty</span>
      {DIFFICULTY_OPTS.map((d) => (
        <FilterPill key={d} value={d} active={camusFilters.difficulty === d}
          label={d === 'all' ? 'All' : d.charAt(0).toUpperCase() + d.slice(1)}
          onClick={(v) => setCamusFilters((f) => ({ ...f, difficulty: v }))} />
      ))}
      <span className="text-xs font-mono text-muted uppercase tracking-wide ml-3 mr-1">Sort</span>
      {SORT_OPTS.map((o) => (
        <FilterPill key={o.value} value={o.value} active={camusFilters.sort === o.value}
          label={o.label} onClick={(v) => setCamusFilters((f) => ({ ...f, sort: v }))} />
      ))}
    </>
  );

  const briscFilterBar = (
    <>
      <span className="text-xs font-mono text-muted uppercase tracking-wide mr-1">Difficulty</span>
      {DIFFICULTY_OPTS.map((d) => (
        <FilterPill key={d} value={d} active={briscFilters.difficulty === d}
          label={d === 'all' ? 'All' : d.charAt(0).toUpperCase() + d.slice(1)}
          onClick={(v) => setBriscFilters((f) => ({ ...f, difficulty: v }))} />
      ))}
      <span className="text-xs font-mono text-muted uppercase tracking-wide ml-3 mr-1">Tumor</span>
      {TUMOR_TYPE_OPTS.map((t) => (
        <FilterPill key={t} value={t} active={briscFilters.tumorType === t}
          label={t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
          onClick={(v) => setBriscFilters((f) => ({ ...f, tumorType: v }))} />
      ))}
      <span className="text-xs font-mono text-muted uppercase tracking-wide ml-3 mr-1">Sort</span>
      {SORT_OPTS.map((o) => (
        <FilterPill key={o.value} value={o.value} active={briscFilters.sort === o.value}
          label={o.label} onClick={(v) => setBriscFilters((f) => ({ ...f, sort: v }))} />
      ))}
    </>
  );

  return (
    <div className="min-h-screen bg-bg font-body text-text">
      <Navbar
        variant="light"
        navItems={NAV_ITEMS}
      />

      <main className="mx-auto max-w-7xl px-6 pt-8 pb-16" style={{ marginTop: 'var(--navbar-height)' }}>
        <div className="mb-10">
          <h1 className="font-heading text-3xl font-bold text-text">Dataset Explorer</h1>
          <p className="mt-2 text-sm text-muted max-w-xl">
            Browse curated examples from CAMUS (cardiac ultrasound) and BRISC (brain MRI).
            Click any image to open it in the workstation.
          </p>
          <DifficultyLegend />
        </div>

        <DatasetSection
          id="camus"
          title="CAMUS"
          modality="Cardiac Ultrasound"
          samples={camusSamples}
          highlights={CAMUS_HIGHLIGHTS}
          samplesById={samplesById}
          filters={camusFilterBar}
          onSelect={handleSelect}
        />

        <DatasetSection
          id="brisc"
          title="BRISC"
          modality="Brain MRI"
          samples={briscSamples}
          highlights={BRISC_HIGHLIGHTS}
          samplesById={samplesById}
          filters={briscFilterBar}
          onSelect={handleSelect}
        />
      </main>
    </div>
  );
}
