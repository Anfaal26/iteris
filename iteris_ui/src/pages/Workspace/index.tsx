/**
 * Iteris workstation (spec §6) — placeholder scaffold.
 * Owned by the Workspace agent: three-zone layout, four viewing modes,
 * canvas image viewer, results panel, and the LLM interpretation layer (§7).
 */
export default function Workspace() {
  return (
    <main className="min-h-screen bg-bg font-body text-text">
      <div className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="font-heading text-3xl font-bold text-accent">
          Iteris Workstation
        </h1>
        <p className="mt-4 text-muted">
          Scaffold — control panel, image viewer, viewing modes, and results
          panel are built by the Workspace agent against the API contract.
        </p>
      </div>
    </main>
  );
}
