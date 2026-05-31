import { describe, expect, it } from 'vitest';
import { api } from './client';

/**
 * Foundation smoke tests: with mocks enabled (default), the client returns
 * spec-shaped data. These guard the contract the page agents build against.
 */
describe('mock api', () => {
  it('returns the six deployed models', async () => {
    const models = await api.models();
    expect(models).toHaveLength(5); // U-Net baseline + DQN, DDQN, Dueling DQN, DDPG
    expect(models.find((m) => m.id === 'ddpg')?.diceCamus).toBe(0.912);
    expect(models.find((m) => m.id === 'unet-baseline')?.selectable).toBe(false);
  });

  it('predicts with per-structure metrics for CAMUS', async () => {
    const res = await api.predict({
      imageB64: '',
      modelId: 'ddpg',
      dataset: 'camus',
      mode: 'single',
    });
    expect(res.masks).toHaveLength(3);
    expect(res.metrics.structures.map((s) => s.structure)).toEqual([
      'lv_endo',
      'lv_epi',
      'la',
    ]);
  });

  it('produces a 20-step sequence in playback mode', async () => {
    const res = await api.predict({
      imageB64: '',
      modelId: 'ddpg',
      dataset: 'brisc',
      mode: 'single',
      playback: true,
    });
    expect(res.stepSequence).toHaveLength(20);
  });

  it('streams interpretation sections', async () => {
    let text = '';
    for await (const chunk of api.interpret({
      modelId: 'ddpg',
      structures: ['glioma'],
      metrics: (await api.predict({ imageB64: '', modelId: 'ddpg', dataset: 'brisc', mode: 'single' })).metrics,
      dataset: 'brisc',
      modality: 'mri',
    })) {
      text += chunk;
    }
    expect(text).toContain('Segmentation Summary');
    expect(text).toContain('Literature References');
  });
});
