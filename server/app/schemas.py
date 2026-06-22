"""
Pydantic mirror of iteris_ui/src/api/contract.ts. Field names are camelCase
to match the TS contract exactly — the UI does no case conversion.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel

DatasetId = Literal['camus', 'brisc']
ModelFamily = Literal['baseline', 'discrete-drl', 'continuous-drl']
ModelId = Literal['unet-baseline', 'dqn', 'ddqn', 'dueling-dqn', 'ddpg']
ViewMode = Literal['single', 'wipe', 'side-by-side']
Difficulty = Literal['easy', 'medium', 'hard']
StructureId = Literal['lv_endo', 'lv_epi', 'la', 'glioma', 'meningioma', 'pituitary']


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


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded', 'down']
    modelsLoaded: int
    gpuAvailable: bool
    datasetsAvailable: List[DatasetId]
