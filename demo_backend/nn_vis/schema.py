from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class InputSpec(BaseModel):
    timesteps: int
    channels: int
    channel_names: List[str]


class LabelSpec(BaseModel):
    action: Dict[str, str]
    finger: Dict[str, str]


class ManifestNode(BaseModel):
    id: str
    title: str
    kind: str
    shape: Optional[str] = None
    shape_in: Optional[str] = None
    shape_out: Optional[str] = None
    params: int = 0
    macs: int = 0


class ManifestEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str

    class Config:
        populate_by_name = True


class ManifestTotals(BaseModel):
    params: int
    macs_per_window: int


class TimelineSpec(BaseModel):
    available: bool
    manifest_url: Optional[str] = None


class Manifest(BaseModel):
    model_name: str
    input: InputSpec
    labels: LabelSpec
    nodes: List[ManifestNode]
    edges: List[ManifestEdge]
    totals: ManifestTotals
    timeline: TimelineSpec


class PackedEncoding(BaseModel):
    encoding: str
    shape: List[int]
    data: str


PackedArray = Union[
    List[float],
    List[List[float]],
    List[List[List[float]]],
    PackedEncoding,
]


class ConvWeights(BaseModel):
    id: str
    weight_shape: List[int]
    weights: PackedArray
    bias: Optional[PackedArray]


class LinearWeights(BaseModel):
    id: str
    weight_shape: List[int]
    weights: PackedArray
    bias: PackedArray


class TopKEdge(BaseModel):
    matrix: str
    i: int
    j: int
    v: float


class LSTMWeights(BaseModel):
    id: str
    weight_ih_l0_shape: List[int]
    weight_hh_l0_shape: List[int]
    bias_ih_l0_shape: List[int]
    bias_hh_l0_shape: List[int]
    weight_ih_l0: PackedArray
    weight_hh_l0: PackedArray
    bias_ih_l0: PackedArray
    bias_hh_l0: PackedArray
    topk: Dict[str, Union[int, List[TopKEdge]]]


class WeightsPayload(BaseModel):
    version: int
    conv: List[ConvWeights]
    linear: List[LinearWeights]
    lstm: LSTMWeights


class ActivationTensor(BaseModel):
    shape: List[int]
    values: PackedArray


class ProbSpec(BaseModel):
    values: List[float]
    pred_id: int
    pred_name: str


class Probabilities(BaseModel):
    finger: ProbSpec
    action: ProbSpec


class UncertaintySpec(BaseModel):
    present: bool
    finger_std_mean: Optional[float] = None
    action_std_mean: Optional[float] = None
    finger_entropy: Optional[float] = None
    action_entropy: Optional[float] = None
    finger_mi: Optional[float] = None
    action_mi: Optional[float] = None


class SampleSpec(BaseModel):
    source: str
    index: Optional[int]
    time_s: Optional[float]


class ActivationsPayload(BaseModel):
    sample: SampleSpec
    input: ActivationTensor
    conv1: ActivationTensor
    conv2: ActivationTensor
    lstm_out: ActivationTensor
    last_features: ActivationTensor
    probs: Probabilities
    uncertainty: UncertaintySpec
