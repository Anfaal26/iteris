/**
 * ConvergenceSection — two side-by-side Recharts LineCharts showing
 * convergence curves for CAMUS (0–50k steps) and BRISC (0–30k steps),
 * five lines each (spec §5).
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

/** Props for ConvergenceSection. */
export interface ConvergenceSectionProps {
  id?: string;
}

// Simulated convergence data – 0..50k (10 points) for CAMUS
const CAMUS_DATA = [
  { step: 0, 'U-Net': 0.890, DQN: 0.640, DDQN: 0.645, 'Dueling DQN': 0.648, DDPG: 0.651 },
  { step: 5000, 'U-Net': 0.890, DQN: 0.742, DDQN: 0.748, 'Dueling DQN': 0.752, DDPG: 0.762 },
  { step: 10000, 'U-Net': 0.890, DQN: 0.821, DDQN: 0.830, 'Dueling DQN': 0.834, DDPG: 0.849 },
  { step: 15000, 'U-Net': 0.890, DQN: 0.861, DDQN: 0.869, 'Dueling DQN': 0.873, DDPG: 0.880 },
  { step: 20000, 'U-Net': 0.890, DQN: 0.879, DDQN: 0.887, 'Dueling DQN': 0.891, DDPG: 0.898 },
  { step: 25000, 'U-Net': 0.890, DQN: 0.889, DDQN: 0.895, 'Dueling DQN': 0.899, DDPG: 0.906 },
  { step: 30000, 'U-Net': 0.890, DQN: 0.896, DDQN: 0.901, 'Dueling DQN': 0.904, DDPG: 0.909 },
  { step: 35000, 'U-Net': 0.890, DQN: 0.899, DDQN: 0.904, 'Dueling DQN': 0.907, DDPG: 0.911 },
  { step: 40000, 'U-Net': 0.890, DQN: 0.900, DDQN: 0.904, 'Dueling DQN': 0.908, DDPG: 0.912 },
  { step: 50000, 'U-Net': 0.890, DQN: 0.901, DDQN: 0.905, 'Dueling DQN': 0.908, DDPG: 0.912 },
];

// Simulated convergence data – 0..30k (8 points) for BRISC
const BRISC_DATA = [
  { step: 0, 'U-Net': 0.810, DQN: 0.580, DDQN: 0.583, 'Dueling DQN': 0.585, DDPG: 0.590 },
  { step: 3000, 'U-Net': 0.810, DQN: 0.671, DDQN: 0.677, 'Dueling DQN': 0.681, DDPG: 0.691 },
  { step: 6000, 'U-Net': 0.810, DQN: 0.745, DDQN: 0.752, 'Dueling DQN': 0.756, DDPG: 0.769 },
  { step: 9000, 'U-Net': 0.810, DQN: 0.786, DDQN: 0.793, 'Dueling DQN': 0.797, DDPG: 0.807 },
  { step: 12000, 'U-Net': 0.810, DQN: 0.805, DDQN: 0.812, 'Dueling DQN': 0.816, DDPG: 0.826 },
  { step: 18000, 'U-Net': 0.810, DQN: 0.817, DDQN: 0.824, 'Dueling DQN': 0.828, DDPG: 0.836 },
  { step: 24000, 'U-Net': 0.810, DQN: 0.823, DDQN: 0.829, 'Dueling DQN': 0.833, DDPG: 0.839 },
  { step: 30000, 'U-Net': 0.810, DQN: 0.825, DDQN: 0.830, 'Dueling DQN': 0.835, DDPG: 0.840 },
];

const LINE_COLORS: Record<string, string> = {
  'U-Net': 'var(--color-muted)',
  DQN: colors.gradientA,
  DDQN: colors.gradientB,
  'Dueling DQN': colors.gradientC,
  DDPG: colors.accent,
};

const AGENTS = ['U-Net', 'DQN', 'DDQN', 'Dueling DQN', 'DDPG'];

interface ChartProps {
  data: typeof CAMUS_DATA;
  title: string;
  maxStep: number;
  yDomain: [number, number];
  baseline: number;
}

const ConvergenceChart: React.FC<ChartProps> = ({ data, title, yDomain }) => (
  <div className="bg-surface border border-border rounded-lg p-4">
    <h3 className="font-heading text-sm font-semibold text-text mb-3">{title}</h3>
    <div aria-label={`Convergence chart: ${title}`} style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
            tickFormatter={(v: number) => v >= 1000 ? `${v / 1000}k` : String(v)}
            label={{
              value: 'Training steps',
              position: 'insideBottom',
              offset: -8,
              fontSize: 10,
              fill: 'var(--color-muted)',
            }}
          />
          <YAxis
            domain={yDomain}
            tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
            tickFormatter={(v: number) => v.toFixed(2)}
            width={44}
            label={{
              value: 'Dice',
              angle: -90,
              position: 'insideLeft',
              offset: 10,
              fontSize: 10,
              fill: 'var(--color-muted)',
            }}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              fontSize: 10,
              fontFamily: 'var(--font-mono)',
            }}
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={(label: number) =>
              label >= 1000 ? `Step ${label / 1000}k` : `Step ${label}`
            }
          />
          <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-body)' }} />
          {AGENTS.map((name) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={LINE_COLORS[name]}
              strokeWidth={name === 'U-Net' ? 1 : 2}
              strokeDasharray={name === 'U-Net' ? '4 3' : undefined}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);

/**
 * Side-by-side convergence curves for CAMUS and BRISC.
 */
export const ConvergenceSection: React.FC<ConvergenceSectionProps> = ({
  id = 'convergence',
}) => {
  return (
    <section
      id={id}
      aria-labelledby="convergence-heading"
      className="py-12 scroll-mt-16"
    >
      <h2
        id="convergence-heading"
        className="font-heading text-xl font-bold text-text mb-4"
      >
        Convergence Curves
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        Mean episode Dice vs training steps. U-Net baseline shown as a dashed reference.
        All DRL agents evaluated against the frozen U-Net outputs as initial masks.
      </p>
      <div className="grid gap-6 lg:grid-cols-2">
        <ConvergenceChart
          data={CAMUS_DATA}
          title="CAMUS (0 – 50k steps)"
          maxStep={50000}
          yDomain={[0.60, 0.93]}
          baseline={0.890}
        />
        <ConvergenceChart
          data={BRISC_DATA}
          title="BRISC (0 – 30k steps)"
          maxStep={30000}
          yDomain={[0.55, 0.86]}
          baseline={0.810}
        />
      </div>
    </section>
  );
};

export default ConvergenceSection;
