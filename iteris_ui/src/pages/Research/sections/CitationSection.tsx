/**
 * CitationSection — BibTeX block with copy button, and links to
 * GitHub, Hugging Face Hub, and dataset sources (spec §5).
 */
import React, { useState } from 'react';

/** Props for CitationSection. */
export interface CitationSectionProps {
  id?: string;
}

const BIBTEX = `@misc{iteris2026,
  title        = {Deep Reinforcement Learning for Adaptive Boundary
                   Refinement in Medical Image Segmentation},
  author       = {Anwar, Ahmad Faaiz and {Capstone Team PRJ63504}},
  year         = {2026},
  institution  = {Taylor's University, School of Computer Science
                   and Digital Technology},
  note         = {Capstone project PRJ63504; code and weights at
                   https://github.com/iteris-ui/iteris},
  howpublished = {\\url{https://github.com/iteris-ui/iteris}},
}`;

interface ExternalLink {
  label: string;
  href: string;
  description: string;
}

const LINKS: ExternalLink[] = [
  {
    label: 'GitHub Repository',
    href: 'https://github.com/iteris-ui/iteris',
    description: 'Source code, training scripts, and model checkpoints.',
  },
  {
    label: 'Hugging Face Hub',
    href: 'https://huggingface.co/iteris',
    description: 'Pre-trained DDPG and Dueling DQN model weights.',
  },
  {
    label: 'CAMUS Dataset',
    href: 'https://www.creatis.insa-lyon.fr/Challenge/camus/',
    description:
      'Official CAMUS echocardiography challenge dataset (450 patients).',
  },
  {
    label: 'BRISC Dataset',
    href: 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
    description: 'Brain tumour MRI dataset used for BRISC experiments.',
  },
];

/**
 * BibTeX citation block with a copy-to-clipboard button and external links.
 */
export const CitationSection: React.FC<CitationSectionProps> = ({ id = 'citation' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      const timer = setTimeout(() => setCopied(false), 2000);
      return () => clearTimeout(timer);
    } catch {
      // clipboard API unavailable in some test environments
    }
  };

  return (
    <section id={id} aria-labelledby="citation-heading" className="py-12 scroll-mt-16">
      <h2 id="citation-heading" className="font-heading text-xl font-bold text-text mb-6">
        Citation
      </h2>

      {/* BibTeX block */}
      <div className="relative rounded-lg overflow-hidden border border-border mb-8">
        <div className="flex items-center justify-between px-4 py-2 bg-landing-bg border-b border-border/30">
          <span className="text-xs font-body text-muted">BibTeX</span>
          <button
            type="button"
            onClick={handleCopy}
            aria-label={copied ? 'Copied to clipboard' : 'Copy BibTeX to clipboard'}
            className={[
              'text-xs font-body px-2 py-1 rounded',
              'transition-colors duration-panel ease-out',
              copied
                ? 'text-success bg-success/10'
                : 'text-muted hover:text-text',
            ].join(' ')}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        <pre
          className="p-4 text-xs font-mono text-landing-text bg-landing-bg overflow-x-auto leading-relaxed whitespace-pre"
          aria-label="BibTeX citation"
        >
          {BIBTEX}
        </pre>
      </div>

      {/* External links */}
      <div>
        <h3 className="font-heading text-sm font-semibold text-text mb-4">
          Resources
        </h3>
        <ul className="space-y-3 list-none pl-0">
          {LINKS.map((link) => (
            <li key={link.href} className="flex items-start gap-3">
              <a
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-body text-accent hover:underline shrink-0"
              >
                {link.label} ↗
              </a>
              <span className="text-sm font-body text-muted">{link.description}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
};

export default CitationSection;
