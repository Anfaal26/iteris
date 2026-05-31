/**
 * Landing page (spec §4) — placeholder scaffold.
 * Owned by the Landing agent: Three.js lattice scene + 8 sections, dark theme.
 */
export default function Landing() {
  return (
    <main
      data-theme="default"
      className="min-h-screen bg-landing-bg text-landing-text font-body"
    >
      <section className="flex min-h-screen flex-col items-center justify-center gap-6 px-6 text-center">
        <span className="rounded-full border border-accent px-4 py-1 font-mono text-xs text-accent">
          Taylor&apos;s University · PRJ63504 Capstone
        </span>
        <h1 className="font-heading text-5xl font-bold sm:text-7xl">
          See How AI
          <br />
          <span className="bg-iteris-gradient bg-clip-text text-transparent">
            Learns to See.
          </span>
        </h1>
        <p className="max-w-xl text-sm text-landing-text/50">
          Landing page scaffold — Three.js lattice and full section sequence are
          built by the Landing agent against the shared design tokens.
        </p>
      </section>
    </main>
  );
}
