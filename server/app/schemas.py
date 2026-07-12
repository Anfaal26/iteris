"""
Pydantic mirror of iteris_ui/src/api/contract.ts. Field names are camelCase
to match the TS contract exactly — the UI does no case conversion.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel

DatasetId = Literal['camus', 'brisc']
ModelFamily = Literal['baseline', 'discrete-drl', 'continuous-drl']
ModelId = Literal['unet-baseline', 'lite-unet', 'dqn', 'ddqn', 'dueling-dqn', 'ddpg', 'td3']
Regime = Literal['low', 'high']
ViewMode = Literal['single', 'wipe', 'side-by-side']
Difficulty = Literal['easy', 'medium', 'hard']
StructureId = Literal['lv_endo', 'lv_epi', 'la', 'glioma', 'meningioma', 'pituitary']
Modality = Literal['ultrasound', 'mri']
TumorType = Literal['glioma', 'meningioma', 'pituitary']


class StructureMetrics(BaseModel):
    structure: StructureId
    label: str
    dice: float
    iou: float
    hd: float
    hd95: float


class Metrics(BaseModel):
    dice: float
    iou: float
    hd: float
    hd95: float
    structures: List[StructureMetrics]
    baselineDice: float


class MaskLayer(BaseModel):
    structure: StructureId
    label: str
    imageB64: str
    color: str


class IterationStep(BaseModel):
    step: int
    masks: List[MaskLayer]
    deltaDice: float
    annotation: str


class PredictRequest(BaseModel):
    imageB64: str
    modelId: ModelId
    dataset: DatasetId
    regime: Optional[Regime] = None
    mode: ViewMode
    playback: Optional[bool] = False
    gtMaskB64: Optional[str] = None


class PredictResponse(BaseModel):
    sessionId: str
    modelId: ModelId
    dataset: DatasetId
    masks: List[MaskLayer]
    metrics: Metrics
    stepSequence: Optional[List[IterationStep]] = None
    preprocessingMs: float
    inferenceMs: float
    imageWidth: int
    imageHeight: int


class ModelRecord(BaseModel):
    id: ModelId
    name: str
    family: ModelFamily
    description: str
    diceCamus: Optional[float] = None
    diceBrisc: Optional[float] = None
    iou: Optional[float] = None
    hd: Optional[float] = None
    deployed: bool
    selectable: bool


class CompareRequest(BaseModel):
    imageB64: str
    modelIds: List[ModelId]
    dataset: DatasetId
    regime: Optional[Regime] = None


ModelFamilyArg = Literal['baseline', 'drl']
AlgoId = Literal['duelingddqn', 'td3', 'dqn', 'ddqn', 'ddpg']


class InferRequest(BaseModel):
    """POST /infer — the generic inference entry point keyed by
    (dataset, model_family, algo, regime). See app/drl.py REGISTRY."""
    imageB64: str
    dataset: DatasetId
    modelFamily: ModelFamilyArg
    algo: AlgoId
    regime: Regime
    gtMaskB64: Optional[str] = None


class InferResponse(BaseModel):
    sessionId: str
    dataset: DatasetId
    algo: AlgoId
    regime: Regime
    masks: List[MaskLayer]
    metrics: Metrics
    # DRL-only: refinement episode length. None for a single U-Net pass.
    refinementSteps: Optional[int] = None
    inferenceMs: float
    imageWidth: int
    imageHeight: int


class CompareResult(BaseModel):
    modelId: ModelId
    masks: List[MaskLayer]
    metrics: Metrics


class CompareResponse(BaseModel):
    results: List[CompareResult]


class SampleImage(BaseModel):
    id: str
    thumbnailB64: str
    modality: Literal['ultrasound', 'mri']
    anatomy: str
    difficulty: Difficulty
    bestDice: float
    dataset: DatasetId


class InterpretRequest(BaseModel):
    modelId: ModelId
    structures: List[StructureId]
    metrics: Metrics
    dataset: DatasetId
    modality: Modality
    difficulty: Optional[Difficulty] = None
    tumorType: Optional[TumorType] = None


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded', 'down']
    modelsLoaded: int
    gpuAvailable: bool
    datasetsAvailable: List[DatasetId]
