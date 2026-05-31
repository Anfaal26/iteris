/**
 * Research showcase (spec §5) — placeholder scaffold.
 * Owned by the Research agent: sticky sidebar nav + 9 academic sections.
 */
export default function Research() {
  return (
    <main className="min-h-screen bg-bg font-body text-text">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="font-heading text-3xl font-bold text-accent">
          Research Showcase
        </h1>
        <p className="mt-4 text-muted">
          Scaffold — Abstract, Datasets, Methods, Models, Results, Ablations,
          Transfer Learning, Figures, and Citation sections are built by the
          Research agent.
        </p>
      </div>
    </main>
  );
}
