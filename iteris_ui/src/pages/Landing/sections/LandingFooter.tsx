/** LandingFooter — minimal 4-column footer with ice-blue top border. */
import React from 'react';
import { LogoMark } from '@/components';

const COLUMNS = [
  { heading: 'Product',   links: [{ label: 'Workstation',       href: '/workspace' }, { label: 'Model Library',  href: '/models'   }, { label: 'Dataset Explorer', href: '/datasets' }] },
  { heading: 'Research',  links: [{ label: 'Research Overview', href: '/research'  }, { label: 'Methodology',    href: '/research#methodology' }, { label: 'Results', href: '/research#results' }] },
  { heading: 'Resources', links: [{ label: 'Documentation',     href: '/docs'      }, { label: 'API Reference',  href: '/docs/api' }, { label: 'GitHub',  href: 'https://github.com' }] },
  { heading: 'Project',   links: [{ label: "Taylor's University", href: 'https://taylor.edu.my' }, { label: 'PRJ63504 Capstone', href: '/research' }, { label: 'Team', href: '/research#team' }] },
];

export const LandingFooter: React.FC = () => (
  <footer role="contentinfo" className="px-6 lg:px-10 pt-16 pb-8"
    style={{ background: 'var(--color-landing-footer)', borderTop: '1px solid rgba(56,189,248,0.07)' }}>
    <div className="mx-auto max-w-6xl">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-10 pb-12"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        {/* Brand */}
        <div className="col-span-2 sm:col-span-1 flex flex-col gap-4">
          <a href="/" className="flex items-center gap-2 no-underline w-fit" aria-label="Iteris home">
            <LogoMark size={22} className="text-grad-b" ariaLabel="" />
            <span className="font-heading font-semibold text-sm tracking-wide text-landing-text">ITERIS</span>
          </a>
          <p className="text-[12px] leading-relaxed" style={{ color: 'rgba(240,249,255,0.3)', maxWidth: 160 }}>
            Medical AI segmentation research workstation.
          </p>
        </div>
        {/* Link columns */}
        {COLUMNS.map((col) => (
          <div key={col.heading} className="flex flex-col gap-3">
            <h3 className="font-heading font-semibold text-[10px] uppercase tracking-[0.14em]"
              style={{ color: 'rgba(240,249,255,0.45)' }}>{col.heading}</h3>
            <ul className="flex flex-col gap-2 list-none m-0 p-0">
              {col.links.map((link) => (
                <li key={link.href}>
                  <a href={link.href} className="text-[12px] no-underline transition-colors duration-panel ease-out"
                    style={{ color: 'rgba(240,249,255,0.32)' }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(240,249,255,0.85)'; }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(240,249,255,0.32)'; }}>
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-8">
        <p className="text-[11px]" style={{ color: 'rgba(240,249,255,0.2)' }}>© 2024 ITERIS · Taylor's University PRJ63504</p>
        <p className="text-[11px]" style={{ color: 'rgba(240,249,255,0.15)' }}>Research prototype only — not approved for clinical use.</p>
      </div>
    </div>
  </footer>
);
export default LandingFooter;
