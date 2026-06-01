/**
 * TransferLearningSection — transfer learning table (zero-shot + few-shot)
 * and a Recharts LineChart showing the label-efficiency curve (spec §5).
 */
import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { colors } from '@/tokens';

/** Props for TransferLearningSection. */
export interface TransferLearningSectionProps {
  id?: string;
}

interface TransferRow {
  regime: string;
  labelFraction: string;
  camusDice: number | null;
  briscDice: number;
  briscRecovery: string;
  note: string;
}

const TRANSFER_ROWS: TransferRow[] = [
  {
    regime: 'Zero-shot (no fine-tuning)',
    labelFraction: '0%',
    camusDice: null,
    briscDice: 0.794,
    briscRecovery: '94.5%',
    note: 'Reasonable cross-modality transfer despite domain gap.',
  },
  {
    regime: 'Few-shot 10%',
    labelFraction: '10%',
    camusDice: null,
    briscDice: 0.821,
    briscRecovery: '97.7%',
    note: '16 BRISC cases sufficient to recover most of the performance.',
  },
  {
    regime: 'Few-shot 20%',
    labelFraction: '20%',
    camusDice: null,
    briscDice: 0.827,
    briscRecovery: '98.3%',
    note: 'Plateau region; marginal gain from 10% → 20%.',
  },
  {
    regime: 'Few-shot 50%',
    labelFraction: '50%',
    camusDice: null,
    briscDice: 0.836,
    briscRecovery: '99.5%',
    note: 'Near-full supervised performance with only half the labels.',
  },
];

// Chart data: label fraction → Dice per agent
const CHART_DATA = [
  { fraction: '10%', DQN: 0.812, DDQN: 0.817, 'Dueling DQN': 0.819, DDPG: 0.821 },
  { fraction: '20%', DQN: 0.818, DDQN: 0.822, 'Dueling DQN': 0.824, DDPG: 0.827 },
  { fraction: '50%', DQN: 0.826, DDQN: 0.831, 'Dueling DQN': 0.833, DDPG: 0.836 },
  { fraction: '100%', DQN: 0.825, DDQN: 0.830, 'Dueling DQN': 0.835, DDPG: 0.840 },
];

const LINE_COLORS = {
  DQN: colors.gradientA,
  DDQN: colors.gradientB,
  'Dueling DQN': colors.gradientC,
  DDPG: colors.accent,
};

/**
 * Transfer learning section with table and label-efficiency line chart.
 */
export const TransferLearningSection: React.FC<TransferLearningSectionProps> = ({
  id = 'transfer-learning',
}) => {
  return (
    <section id={id} aria-labelledby="transfer-heading" className="py-12 scroll-mt-16">
      <h2 id="transfer-heading" className="font-heading text-xl font-bold text-text mb-4">
        Transfer Learning
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        A DDPG agent pre-trained on CAMUS is fine-tuned on increasing fractions of the BRISC
        training set. Recovery is reported relative to the fully-supervised BRISC Dice (0.840).
      </p>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-border mb-8">
        <table className="w-full text-sm font-body" aria-label="Transfer learning results">
          <thead>
            <tr className="bg-bg">
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Regime
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                Labels used
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                BRISC Dice
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                Recovery
              </th>
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Note
              </th>
            </tr>
          </thead>
          <tbody>
            {TRANSFER_ROWS.map((row, i) => (
              <tr key={row.regime} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                <td className="px-4 py-2 text-text font-medium">{row.regime}</td>
                <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                  {row.labelFraction}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-success font-medium">
                  {row.briscDice.toFixed(3)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-accent">
                  {row.briscRecovery}
                </td>
                <td className="px-4 py-2 text-muted text-xs">{row.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Label-efficiency line chart */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <h3 className="font-heading text-sm font-semibold text-text mb-4">
          Label Efficiency Curve — BRISC Dice vs Label Fraction
        </h3>
        <div aria-label="Label efficiency line chart" style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={CHART_DATA} margin={{ top: 4, right: 24, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="fraction"
                tick={{ fontSize: 11, fill: 'var(--color-muted)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
                label={{
                  value: 'Label fraction',
                  position: 'insideBottom',
                  offset: -2,
                  fontSize: 11,
                  fill: 'var(--color-muted)',
                }}
              />
              <YAxis
                domain={[0.80, 0.85]}
                tick={{ fontSize: 11, fill: 'var(--color-muted)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
                tickFormatter={(v: number) => v.toFixed(3)}
                width={52}
                label={{
                  value: 'Dice',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 10,
                  fontSize: 11,
                  fill: 'var(--color-muted)',
                }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                }}
                formatter={(value: number) => value.toFixed(3)}
              />
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-body)' }}
              />
              {(Object.keys(LINE_COLORS) as Array<keyof typeof LINE_COLORS>).map((name) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={LINE_COLORS[name]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
};

export default TransferLearningSection;
