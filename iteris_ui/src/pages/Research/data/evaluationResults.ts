/**
 * Real evaluation output — transcribed verbatim from
 * evaluation_outputs/master_comparison.csv (2026-07-20 run).
 *
 * The two U-Net baseline rows per dataset/class were exported with
 * phase="Unknown"; disambiguated by dice (higher = Phase A, paired with
 * the deployed AttentionResUNet; lower = Phase B, paired with LiteUNet) —
 * confirmed by matching against every DRL run's init_dice_mean.
 */

export type Phase = 'Phase A' | 'Phase B';
export type ModelFamily = 'DRL' | 'UNet';

export interface EvalRow {
  dataset: 'CAMUS' | 'BRISC';
  className: string;
  phase: Phase;
  model: string;
  modelFamily: ModelFamily;
  actionSpace: 'discrete' | 'continuous' | 'n/a';
  dice: number;
  hd95: number;
  iou: number | null;
}

export const EVAL_ROWS: EvalRow[] = [
  // --- DRL, Phase A (paired with AttentionResUNet baseline) ---
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase A', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.8672322322119891, hd95: 9.093334420916154, iou: 0.7192558944833565 },
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase A', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.8743027991490442, hd95: 6.946404320391286, iou: 0.7710327818173871 },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase A', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.8937323586924812, hd95: 7.9586191284187215, iou: 0.8057604301215663 },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase A', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.8882873735907229, hd95: 8.307416148344101, iou: 0.7913839433292598 },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase A', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.9338352781416198, hd95: 6.047888823587742, iou: 0.8720025449318152 },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase A', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.9241027556489133, hd95: 6.5456155057490335, iou: 0.8550450469066795 },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase A', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.8612109077049235, hd95: 6.741273010797652, iou: 0.7562040495885834 },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase A', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.842024816021623, hd95: 8.582111512739637, iou: 0.719920616527647 },

  // --- DRL, Phase B (paired with LiteUNet baseline) ---
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase B', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.7526108335052066, hd95: 14.731538363513433, iou: 0.6686978374419226 },
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase B', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.7539957792987068, hd95: 14.559530010438339, iou: 0.6483931651489101 },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase B', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.8517525301322753, hd95: 12.400689804408886, iou: 0.7398531435323841 },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase B', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.8515304785086768, hd95: 11.706153069736002, iou: 0.7257881082103753 },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase B', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.911646184166086, hd95: 8.189137202877596, iou: 0.8393461338092675 },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase B', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.9127856646276189, hd95: 8.273553245416585, iou: 0.8324907245542903 },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase B', model: 'DuelingDDQN', modelFamily: 'DRL', actionSpace: 'discrete', dice: 0.8162694752975729, hd95: 10.063287934006055, iou: 0.6937706935032959 },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase B', model: 'TD3', modelFamily: 'DRL', actionSpace: 'continuous', dice: 0.8022262142787947, hd95: 12.356964757614168, iou: 0.6557683324862359 },

  // --- U-Net baselines (phase disambiguated from "Unknown" by dice rank) ---
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase A', model: 'AttentionResUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.8695, hd95: 7.67, iou: null },
  { dataset: 'BRISC', className: 'tumor', phase: 'Phase B', model: 'LiteUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.8191, hd95: 12.76, iou: null },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase A', model: 'AttentionResUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.8935, hd95: 7.79, iou: null },
  { dataset: 'CAMUS', className: 'LA', phase: 'Phase B', model: 'LiteUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.8214, hd95: 19.86, iou: null },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase A', model: 'AttentionResUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.936, hd95: 5.58, iou: null },
  { dataset: 'CAMUS', className: 'LV_endo', phase: 'Phase B', model: 'LiteUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.9105, hd95: 11.82, iou: null },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase A', model: 'AttentionResUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.869, hd95: 6.13, iou: null },
  { dataset: 'CAMUS', className: 'LV_epi', phase: 'Phase B', model: 'LiteUNet', modelFamily: 'UNet', actionSpace: 'n/a', dice: 0.809, hd95: 14.16, iou: null },
];

/** BRISC + all 3 CAMUS classes for a given model/phase, dice-sorted ascending is not assumed by callers. */
export function rowsFor(model: string, phase: Phase): EvalRow[] {
  return EVAL_ROWS.filter((r) => r.model === model && r.phase === phase);
}

export function baselineFor(dataset: EvalRow['dataset'], className: string, phase: Phase): EvalRow | undefined {
  return EVAL_ROWS.find(
    (r) => r.dataset === dataset && r.className === className && r.phase === phase && r.modelFamily === 'UNet',
  );
}

/** Mean dice for a model within a phase, across all classes of a dataset (BRISC has one class). */
export function meanDice(model: string, phase: Phase, dataset?: EvalRow['dataset']): number {
  const rows = EVAL_ROWS.filter(
    (r) => r.model === model && r.phase === phase && (dataset ? r.dataset === dataset : true),
  );
  return rows.reduce((sum, r) => sum + r.dice, 0) / rows.length;
}
