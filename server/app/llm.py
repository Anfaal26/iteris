"""
Streams a Claude-generated interpretation of a segmentation result, in the
exact `[section-id]` + text format iteris_ui's LLMInterpretationPanel expects
(see src/components/LLMInterpretationPanel/LLMInterpretationPanel.tsx — it
regexes each chunk for a leading `[section-id]` marker to switch sections).

One Claude call per section rather than asking the model to self-format —
the backend emits the markers, so the contract can't be broken by the model
drifting from the expected syntax.
"""

import os

import anthropic

from .schemas import InterpretRequest

MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')

# Foundational papers only — given to the model explicitly so it cites real,
# verifiable work instead of inventing references for the literature section.
KNOWN_REFERENCES = (
    'Leclerc et al., "Deep Learning for Segmentation Using an Open '
    'Large-Scale Dataset in 2D Echocardiography," IEEE TMI 2019 (CAMUS dataset); '
    'Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," '
    '2018 (attention-gate architecture used by this model); '
    'Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections '
    'with expert segmentation labels," Scientific Data 2017 (brain tumour '
    'segmentation benchmarking).'
)

SECTION_ORDER = [
    'segmentation-summary',
    'clinical-significance',
    'metric-interpretation',
    'performance-analysis',
    'literature-references',
]


def _context_block(req: InterpretRequest) -> str:
    structures = ', '.join(req.structures) if req.structures else 'none detected'
    lines = [
        f'Model: {req.modelId}',
        f'Dataset: {req.dataset} ({req.modality})',
        f'Structures segmented: {structures}',
        f'Overall Dice: {req.metrics.dice} (baseline Dice: {req.metrics.baselineDice})',
        f'Overall IoU: {req.metrics.iou}, HD: {req.metrics.hd}, HD95: {req.metrics.hd95}',
    ]
    if req.difficulty:
        lines.append(f'Case difficulty: {req.difficulty}')
    if req.tumorType:
        lines.append(f'Tumor type: {req.tumorType}')
    if req.metrics.structures:
        per_structure = '; '.join(
            f'{s.label}: Dice {s.dice}, IoU {s.iou}' for s in req.metrics.structures
        )
        lines.append(f'Per-structure: {per_structure}')
    return '\n'.join(lines)


SECTION_PROMPTS = {
    'segmentation-summary': (
        'In 2-3 sentences, summarise what was segmented and the overall result quality, '
        'in plain technical language for an ML researcher audience.'
    ),
    'clinical-significance': (
        'In 2-3 sentences, explain what this level of segmentation accuracy would mean '
        'for downstream clinical/research use (e.g. volumetric measurement reliability). '
        'Be measured — this is a research prototype, not a diagnostic tool, and must not '
        'be presented as one.'
    ),
    'metric-interpretation': (
        'In 2-3 sentences, interpret the Dice/IoU/HD95 numbers given — what do they mean '
        'in practical terms (e.g. typical boundary error in mm/pixels for structures this size)?'
    ),
    'performance-analysis': (
        'In 2-3 sentences, comment on how this model likely achieved this result relative to '
        'the baseline Dice given, in terms of what the architecture (attention gates, residual '
        'encoder/decoder) is doing.'
    ),
    'literature-references': (
        'In 2-3 sentences, relate this result to the relevant literature below. ONLY cite '
        'these exact papers — do not invent or reference any other paper:\n' + KNOWN_REFERENCES
    ),
}


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY is not set on this Space.')
    return anthropic.Anthropic(api_key=api_key)


def stream_interpretation(req: InterpretRequest):
    """Sync generator — yields `[section-id]` markers and text chunks in
    iteris_ui's expected streaming format. Used directly as a FastAPI
    StreamingResponse body.
    """
    client = _client()
    context = _context_block(req)

    for section_id in SECTION_ORDER:
        yield f'[{section_id}]'
        prompt = (
            f'{SECTION_PROMPTS[section_id]}\n\nCase data:\n{context}\n\n'
            'Respond with only the requested sentences, no headers, no preamble.'
        )
        with client.messages.stream(
            model=MODEL,
            max_tokens=220,
            messages=[{'role': 'user', 'content': prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
        yield '\n\n'
