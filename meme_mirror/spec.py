"""ExpressionSpec: the motion description an LLM writes (the 'own' generator path).

Units the LLM sees (same conventions as binhpham's recipes, so prompts transfer):
    pitch  deg, + = look down          roll  deg, + = tilt toward robot's left
    yaw    deg, + = turn to robot's left   z    mm, + = head up
    ear_r / ear_l  deg of droop: 0 = straight up, + = swung outward/down (sad ears), - = inward
    body_yaw deg, + = body turns to robot's left
Neutral is pitch=roll=yaw=z=body_yaw=0, ears 10.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

Ease = Literal["smooth", "linear", "snap"]
Channel = Literal["pitch", "roll", "yaw", "z", "ear_r", "ear_l", "ears", "body_yaw"]


class Keyframe(BaseModel):
    dt: float = Field(gt=0, le=10, description="seconds to reach this pose from the previous one")
    pitch: float | None = None
    roll: float | None = None
    yaw: float | None = None
    z: float | None = None
    ear_r: float | None = None
    ear_l: float | None = None
    ears: float | None = Field(None, description="sets both ears")
    body_yaw: float | None = None
    ease: Ease = "smooth"


class KeyframeSegment(BaseModel):
    kind: Literal["keyframes"] = "keyframes"
    intent: str = Field("", description="what this part is meant to convey")
    keyframes: list[Keyframe] = Field(min_length=1)


class LibrarySegment(BaseModel):
    kind: Literal["library"] = "library"
    intent: str = ""
    move_id: str = Field(description="catalog id, e.g. 'pollen/sad1'")
    time_scale: float = Field(1.0, ge=0.3, le=3.0, description=">1 = slower")
    amplitude: float = Field(1.0, ge=0.2, le=1.6, description="scales motion away from neutral")
    start_s: float = Field(0.0, ge=0)
    end_s: float | None = Field(None, description="trim; None = to the end")


class HoldSegment(BaseModel):
    kind: Literal["hold"] = "hold"
    intent: str = ""
    seconds: float = Field(gt=0, le=10)


Segment = Annotated[Union[KeyframeSegment, LibrarySegment, HoldSegment], Field(discriminator="kind")]


class Overlay(BaseModel):
    """Periodic or jittery motion added on top of the timeline (trembling, bobbing, wiggling ears)."""
    channel: Channel
    shape: Literal["sine", "tremble"] = "sine"
    amplitude: float = Field(gt=0, le=30, description="deg (mm for z)")
    freq_hz: float = Field(gt=0, le=8)
    start_s: float = Field(0, ge=0)
    end_s: float | None = None
    antiphase_ears: bool = Field(False, description="for channel 'ears': ears move in opposite directions")


class ExpressionSpec(BaseModel):
    title: str
    segments: list[Segment] = Field(min_length=1)
    overlays: list[Overlay] = []
    sound: str | None = Field(None, description="catalog move_id whose sound to play, or None")


NEUTRAL = dict(pitch=0.0, roll=0.0, yaw=0.0, z=0.0, ear_r=10.0, ear_l=10.0, body_yaw=0.0, x=0.0, y=0.0)
