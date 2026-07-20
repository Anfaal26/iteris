/**
 * DatasetsSection — tabbed view of CAMUS and BRISC datasets with
 * dataset summary tables, preprocessing notes, and EDA findings (spec §5).
 */
import React, { useState } from 'react';

/** Props for DatasetsSection. */
export interface DatasetsSectionProps {
  id?: string;
}

type TabId = 'camus' | 'brisc';

interface DatasetSpec {
  label: string;
  rows: Array<{ field: string; value: string }>;
  preprocessing: string[];
  eda: string[];
}

const DATASETS: Record<TabId, DatasetSpec> = {
  camus: {
    label: 'CAMUS',
    rows: [
      { field: 'Modality', value: 'Echocardiography (2D ultrasound)' },
      { field: 'Patients', value: '450' },
      { field: 'Structures', value: 'LV endocardium, LV epicardium, Left atrium' },
      { field: 'Views', value: '2-chamber, 4-chamber' },
      { field: 'Cardiac phases', value: 'End-diastole, End-systole' },
      { field: 'Total frames', value: '1,800' },
      { field: 'Image resolution', value: '384 × 512 px' },
      { field: 'Annotation type', value: 'Expert-drawn contour polygon' },
      { field: 'Split (train / val / test)', value: '350 / 50 / 50' },
    ],
    preprocessing: [
      'Histogram equalisation applied to correct gain/TGC variation across probes.',
      'Images resized to 256 × 256 with bilinear interpolation; masks nearest-neighbour.',
      'Intensity normalised to [0, 1] using per-image min–max scaling.',
      'Augmentation: random horizontal flip, ±10° rotation, ±10% brightness jitter (train only).',
    ],
    eda: [
      'Mean LV endocardium Dice across annotators: 0.933 — confirms high label quality.',
      'Inter-frame Dice variance ≥ 0.05 in 12% of patients (poor image quality; marked as "hard").',
      'Median structure area fraction: LV endo 6.2%, LV epi 11.4%, LA 8.7%.',
      'Class imbalance ratio (background : foreground): approximately 9:1.',
    ],
  },
  brisc: {
    label: 'BRISC',
    rows: [
      { field: 'Modality', value: 'Multi-parametric MRI (T1ce, T2, FLAIR)' },
      { field: 'Cases', value: '220' },
      { field: 'Tumour classes', value: 'Glioma, Meningioma, Pituitary' },
      { field: 'Slices per volume', value: '20–30 (axial)' },
      { field: 'Total slices', value: '5,500' },
      { field: 'Image resolution', value: '240 × 240 px' },
      { field: 'Annotation type', value: 'Voxel-level label map' },
      { field: 'Split (train / val / test)', value: '160 / 30 / 30' },
    ],
    preprocessing: [
      'Skull-stripped using HD-BET; background region set to zero intensity.',
      'Slices with < 0.5% tumour occupancy excluded to reduce class sparsity.',
      'Images resized to 256 × 256; intensity normalised per-volume z-score.',
      'Augmentation: elastic deformation (σ = 8), random 90° rotation, Gaussian noise (σ = 0.01).',
    ],
    eda: [
      'Class distribution: Glioma 40%, Meningioma 34%, Pituitary 26%.',
      'Mean tumour area fraction: 4.1% — significant background imbalance.',
      'Slice-level difficulty stratification by tumour area tercile (easy > 7%, medium 3–7%, hard < 3%).',
      'T1ce contrast channel contributes most to boundary distinctiveness (mutual information analysis).',
    ],
  },
};

/**
 * Displays CAMUS and BRISC dataset specifications with preprocessing and EDA notes.
 */
export const DatasetsSection: React.FC<DatasetsSectionProps> = ({ id = 'datasets' }) => {
  const [activeTab, setActiveTab] = useState<TabId>('camus');
  const ds = DATASETS[activeTab];

  return (
    <section id={id} aria-labelledby="datasets-heading" className="py-12 scroll-mt-16">
      <h2 id="datasets-heading" className="font-heading text-xl font-bold text-text mb-6">
        Datasets
      </h2>

      {/* Tabs */}
      <div
        role="tablist"
        aria-label="Dataset tabs"
        className="flex gap-1 mb-6 border-b border-border"
      >
        {(['camus', 'brisc'] as TabId[]).map((tab) => (
          <button
            key={tab}
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={`dataset-panel-${tab}`}
            id={`dataset-tab-${tab}`}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={[
              'px-4 py-2 text-sm font-body font-medium border-b-2 -mb-px',
              'transition-colors duration-panel ease-out',
              activeTab === tab
                ? 'border-accent text-accent'
                : 'border-transparent text-muted hover:text-text',
            ].join(' ')}
          >
            {DATASETS[tab].label}
          </button>
        ))}
      </div>

      {/* Tab panel */}
      <div
        role="tabpanel"
        id={`dataset-panel-${activeTab}`}
        aria-labelledby={`dataset-tab-${activeTab}`}
      >
        {/* Spec table */}
        <div className="overflow-x-auto rounded-lg border border-border mb-6">
          <table className="w-full text-sm font-body">
            <thead>
              <tr className="bg-bg">
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border w-1/3">
                  Field
                </th>
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {ds.rows.map((row, i) => (
                <tr key={row.field} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                  <td className="px-4 py-2 text-muted">{row.field}</td>
                  <td className="px-4 py-2 text-text font-mono text-xs">{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Preprocessing notes */}
        <div className="mb-6">
          <h3 className="font-heading text-sm font-semibold text-text mb-2">
            Preprocessing
          </h3>
          <ul className="space-y-1 list-none pl-0">
            {ds.preprocessing.map((note) => (
              <li key={note} className="flex gap-2 text-sm font-body text-muted">
                <span className="text-accent mt-0.5 shrink-0">–</span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* EDA findings */}
        <div>
          <h3 className="font-heading text-sm font-semibold text-text mb-2">
            EDA Findings
          </h3>
          <ul className="space-y-1 list-none pl-0">
            {ds.eda.map((finding) => (
              <li key={finding} className="flex gap-2 text-sm font-body text-muted">
                <span className="text-accent mt-0.5 shrink-0">–</span>
                <span>{finding}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <figure className="mt-8 bg-surface border border-border rounded-lg p-3">
        <img
          src="/research/figures/unet_per_patient_violin.png"
          alt="U-Net per-patient Dice distribution violin plot, both backbones, all classes"
          className="w-full rounded"
        />
        <figcaption className="text-xs font-body text-muted mt-2">
          Per-patient Dice distribution, both backbones — the long lower tails on both are
          why mean Dice alone understates how variable baseline quality is per case.
        </figcaption>
      </figure>
    </section>
  );
};

export default DatasetsSection;
