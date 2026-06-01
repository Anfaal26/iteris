/**
 * LandingFooter — Section 8: 4-column link grid, logo, tagline, copyright.
 * Uses bg-landing-footer token.
 */

import React from 'react';
import { LogoMark } from '@/components';

interface FooterColumn {
  heading: string;
  links: { label: string; href: string }[];
}

const COLUMNS: FooterColumn[] = [
  {
    heading: 'Product',
    links: [
      { label: 'Workstation', href: '/workspace' },
      { label: 'Model Library', href: '/models' },
      { label: 'Dataset Explorer', href: '/datasets' },
    ],
  },
  {
    heading: 'Research',
    links: [
      { label: 'Research Overview', href: '/research' },
      { label: 'Methodology', href: '/research#methodology' },
      { label: 'Results', href: '/research#results' },
    ],
  },
  {
    heading: 'Resources',
    links: [
      { label: 'Documentation', href: '/docs' },
      { label: 'API Reference', href: '/docs/api' },
      { label: 'GitHub', href: 'https://github.com' },
    ],
  },
  {
    heading: 'Project',
    links: [
      { label: "Taylor's University", href: 'https://taylor.edu.my' },
      { label: 'PRJ63504 Capstone', href: '/research#capstone' },
      { label: 'Team', href: '/research#team' },
    ],
  },
];

/** Section 8 — Landing footer. */
export const LandingFooter: React.FC = () => (
  <footer
    role="contentinfo"
    className="bg-landing-footer border-t border-white/[0.06] px-6 pt-16 pb-8"
  >
    <div className="mx-auto max-w-6xl">
      {/* Top grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-10 pb-12 border-b border-white/[0.06]">
        {/* Logo + tagline column */}
        <div className="col-span-2 sm:col-span-1 flex flex-col gap-4">
          <a href="/" className="flex items-center gap-2 w-fit" aria-label="Iteris home">
            <LogoMark size={24} className="text-grad-a" ariaLabel="" />
            <span className="font-heading font-semibold text-sm text-landing-text">ITERIS</span>
          </a>
          <p className="text-xs text-landing-text/35 leading-relaxed max-w-[160px]">
            Medical AI segmentation research workstation.
          </p>
        </div>

        {/* Link columns */}
        {COLUMNS.map((col) => (
          <div key={col.heading} className="flex flex-col gap-3">
            <h3 className="font-heading font-semibold text-xs text-landing-text/60 uppercase tracking-widest">
              {col.heading}
            </h3>
            <ul className="flex flex-col gap-2 list-none m-0 p-0">
              {col.links.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    className="text-xs text-landing-text/40 hover:text-landing-text transition-colors duration-panel ease-out no-underline"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Bottom bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-8">
        <p className="text-xs text-landing-text/25">
          © 2024 ITERIS · Taylor's University PRJ63504 Capstone
        </p>
        <p className="text-xs text-landing-text/20 max-w-sm text-right leading-relaxed">
          Research prototype only — not approved for clinical use. All metrics are experimental.
        </p>
      </div>
    </div>
  </footer>
);

export default LandingFooter;
