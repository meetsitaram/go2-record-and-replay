"""
Algorithmic dance moves for the Go2.

Each move is a generator function `frame(t, ctx) -> dict` that, given a
time `t` (seconds since the move started) and a context `ctx`, returns a
command dict like:

    {"type": "euler",       "params": {"x": 0.1, "y": 0.0, "z": 0.0}}
    {"type": "body_height", "params": {"data": 0.05}}
    {"type": "move",        "params": {"x": 0.0, "y": 0.3, "z": 0.0}}
    {"type": "compound",    "params": [<sub-command>, <sub-command>, ...]}

`ctx` is a dict providing tempo info. In Phase 1 these default to:
    - beat_hz:    1.0   (treats absolute frequencies as "Hz")
    - beat_phase: 0.0

In Phase 3 we will populate ctx["beat_hz"] from librosa BPM detection so
moves automatically adjust to the song tempo. `beats_per_cycle` on each
move declares how many beats one full cycle of the move should span.

Notes:
    - Euler / BodyHeight moves require the robot to be in BalanceStand.
    - Move (vx/vy/vyaw) requires walking gait active.
    - `category`: A=pure-Euler, B=pure-BodyHeight, C=pure-Move,
       D=Euler+BodyHeight, E=full-compound, F=stinger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

TAU = 2.0 * math.pi


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Hard safety clamps applied by the sender after every frame.
# Note: teleop recordings show the Pro firmware accepts up to ±0.6 rad on
# roll and ±0.45 rad on pitch when in pose mode, so we raise the clamps.
EULER_MAX = 0.60   # rad
HEIGHT_MAX = 0.15  # meters
MOVE_VX_MAX = 0.5  # m/s
MOVE_VY_MAX = 0.5  # m/s
MOVE_VYAW_MAX = 1.8  # rad/s


FrameFn = Callable[[float, dict], dict]


@dataclass
class Move:
    name: str
    description: str
    category: str           # 'A' / 'B' / 'C' / 'D' / 'E' / 'F'
    duration: float         # seconds
    beats_per_cycle: float  # for Phase 3 tempo sync
    requires_walking: bool  # True if this move uses `Move` (vx/vy/vyaw)
    frame: FrameFn = field(repr=False)


# ─── A: Pure Euler ─────────────────────────────────────────────────────

def _shoulder_roll(t, ctx):
    f = 1.2 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.35 * math.sin(TAU * f * t), "y": 0.0, "z": 0.0}}


def _head_nod(t, ctx):
    f = 1.5 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.20 * math.sin(TAU * f * t), "z": 0.0}}


def _head_swivel(t, ctx):
    f = 1.0 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.0, "z": 0.30 * math.sin(TAU * f * t)}}


def _head_swivel_nod(t, ctx):
    """Polyrhythmic combo: yaw at 1.0 Hz + pitch at 1.5 Hz.

    Independent rhythms -- the head shakes "no" while nodding "yes" at
    different rates. Feels curious / playful.
    """
    bh = ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {
        "x": 0.0,
        "y": 0.20 * math.sin(TAU * 1.5 * bh * t),   # nod (same as head_nod)
        "z": 0.30 * math.sin(TAU * 1.0 * bh * t),   # swivel (same as head_swivel)
    }}


def _head_swivel_nod_sync(t, ctx):
    """Two-step nod: swivel right + nod down, recover, swivel left + nod down.

    The head visits four discrete poses in sequence:
        1. yaw right + pitch down       (nod toward the right)
        2. center                        (recovery)
        3. yaw left + pitch down         (nod toward the left)
        4. center                        (recovery)

    Yaw uses a standard sine; pitch uses |sin(yaw_freq)| so it dips down at
    BOTH yaw extremes (not asymmetric like a 90° phase offset which would
    only dip on one side). The pitch oscillates at 2x the yaw frequency
    naturally because |sin(x)| has period pi instead of 2*pi.

    Note: on this Pro firmware +pitch = nose-DOWN (inverse of the documented
    convention), so we use +pitch for the "nod down" direction.
    """
    bh = ctx.get("beat_hz", 1.0)
    f = 0.5 * bh   # one full yaw-right -> yaw-left cycle takes 2 seconds at bh=1
    yaw = 0.30 * math.sin(TAU * f * t)
    pitch = +0.20 * abs(math.sin(TAU * f * t))   # down at both yaw extremes
    return {"type": "euler", "params": {"x": 0.0, "y": pitch, "z": yaw}}


def _head_swivel_dip(t, ctx):
    """Yaw swivel with a pitch DOWN dip at both yaw extremes.

    yaw = 0.30 * sin(2*pi*f*t)             -> swings -0.30 .. +0.30
    pitch = +0.15 * (1 - cos(2*pi*2f*t))/2  -> 0 at center, +0.15 at extremes

    Note: on the Pro (MCF firmware) +pitch is nose-DOWN. (Inverse of the
    documented convention -- verified empirically.)

    Net effect: head bows down whenever it reaches the left-most or right-most
    swing, then lifts back up as it crosses center. Two pitch dips per yaw cycle.
    """
    f = 1.0 * ctx.get("beat_hz", 1.0)
    yaw = 0.30 * math.sin(TAU * f * t)
    # cos(2 * yaw_phase) is +1 at center, -1 at extremes; map to [0, +0.15].
    pitch = +0.075 * (1.0 - math.cos(TAU * 2.0 * f * t))
    return {"type": "euler", "params": {"x": 0.0, "y": pitch, "z": yaw}}


def _figure8_head(t, ctx):
    f = 1.0 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {
        "x": 0.25 * math.sin(TAU * f * t),
        "y": 0.15 * math.sin(TAU * 2 * f * t),
        "z": 0.0,
    }}


def _headbang(t, ctx):
    f = 2.5 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.30 * math.sin(TAU * f * t), "z": 0.0}}


# ─── B: Pure BodyHeight ────────────────────────────────────────────────
#
# NOTE: BodyHeight (api_id 1013) is rejected on Pro firmware (MCF mode) with
# code 3203 (unknown api_id). MCF doesn't expose BodyHeight separately.
# These moves are KEPT as-is in case a future firmware re-enables them, but
# the Euler-pitch variants below (slow_bob, deep_bob, pitch_pop) achieve a
# similar "bobbing" feel on MCF Pro using only Euler.

def _breathing(t, ctx):
    f = 0.4 * ctx.get("beat_hz", 1.0)
    return {"type": "body_height", "params": {"data": 0.08 * math.sin(TAU * f * t)}}


def _bounce_eighth(t, ctx):
    f = 2.0 * ctx.get("beat_hz", 1.0)
    return {"type": "body_height", "params": {"data": 0.05 * math.sin(TAU * f * t)}}


def _pop_drop(t, ctx):
    # Square-ish pop: hold high 0.5s, snap low 0.5s. Period = 1.0s.
    period = 1.0
    phase = (t % period) / period
    delta = +0.08 if phase < 0.5 else -0.10
    return {"type": "body_height", "params": {"data": delta}}


# ─── B': Pitch-based "bounces" that work on MCF (Pro) ──────────────────
# These approximate a bob by oscillating Euler pitch. The body visually
# leans forward/back, which gives a similar "bouncing to the beat" feel.

def _slow_bob(t, ctx):
    f = 0.5 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.18 * math.sin(TAU * f * t), "z": 0.0}}


def _deep_bob(t, ctx):
    f = 1.0 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.25 * math.sin(TAU * f * t), "z": 0.0}}


def _pitch_pop(t, ctx):
    # Square wave on pitch: 0.5s up, 0.5s down. Same vibe as pop_drop.
    period = 1.0
    phase = (t % period) / period
    pitch = +0.20 if phase < 0.5 else -0.20
    return {"type": "euler", "params": {"x": 0.0, "y": pitch, "z": 0.0}}


# ─── C: Pure Move (needs walking gait active) ──────────────────────────

def _hip_shake(t, ctx):
    # Sideways strafe -- the locomotion controller smooths this so the actual
    # motion is gentler than the commanded sine. Higher amplitude + lower freq
    # gives a more visible shimmy than tight high-freq oscillation.
    f = 1.0 * ctx.get("beat_hz", 1.0)
    return {"type": "move", "params": {"x": 0.0, "y": 0.6 * math.sin(TAU * f * t), "z": 0.0}}


def _hip_shake_yaw(t, ctx):
    """Hip shake via Euler yaw only (no walking gait needed)."""
    f = 1.5 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.0, "z": 0.30 * math.sin(TAU * f * t)}}


def _circle_walk(t, ctx):
    # Need higher angular velocity to actually complete a circle in 8s.
    # 8s * 1.2 rad/s = 9.6 rad (~1.5 turns). Robot's actual yaw rate is lower.
    return {"type": "move", "params": {"x": 0.30, "y": 0.0, "z": 1.2}}


# ─── D: Combined multi-axis Euler (no BodyHeight) ──────────────────────
# Originally these used BodyHeight, but the Pro/MCF firmware rejects it.
# We fold the "bounce" into the Euler vector itself: pitch oscillation
# stands in for vertical bounce, giving a similar musical feel.

def _body_wave(t, ctx):
    f = 1.2 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {
        "x": 0.25 * math.sin(TAU * f * t),
        "y": 0.18 * math.cos(TAU * f * t),
        "z": 0.10 * math.sin(TAU * 0.5 * f * t),
    }}


def _reggae_sway(t, ctx):
    f = 0.8 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {
        "x": 0.30 * math.sin(TAU * f * t),
        "y": 0.10 * math.sin(TAU * f * t),
        "z": 0.0,
    }}


def _disco_bounce(t, ctx):
    # Yaw alternates ±0.30 every 0.5s. Pitch bounces at 2 Hz.
    yaw = 0.30 if int(t / 0.5) % 2 == 0 else -0.30
    pitch = 0.15 * math.sin(TAU * 2.0 * ctx.get("beat_hz", 1.0) * t)
    return {"type": "euler", "params": {"x": 0.0, "y": pitch, "z": yaw}}


def _booty_dance(t, ctx):
    """Diagonal lean: right-front corner drops, left-rear corner rises, swap.

    Why diagonal: Euler only commands rigid body tilt, so we can't shorten
    a single leg. But combining roll + pitch shifts the COM toward one
    front corner, which makes that corner's leg shorten the most while
    the opposite rear lengthens the most.

    Right-front low:  roll right (+x) + pitch forward (+y)
    Left-rear high:   automatically follows from the rigid tilt
    Then swap on the next half-cycle.
    """
    f = 0.8 * ctx.get("beat_hz", 1.0)
    s = math.sin(TAU * f * t)
    return {"type": "euler", "params": {
        "x": 0.30 * s,    # roll right when s>0, left when s<0
        "y": 0.25 * s,    # pitch forward when s>0, back when s<0  (key change!)
        "z": 0.15 * s,    # small yaw twist to emphasize the lean
    }}


# ─── E: Full compound (Euler-only, since Move+Euler conflict) ──────────
# NOTE: The firmware treats `Move` (walking) and `Euler` (balance-stand)
# as mutually exclusive states. Combining them silently drops one side.
# So we make the "full body" moves pure Euler with multi-axis oscillation.

def _shakira_combo(t, ctx):
    bh = ctx.get("beat_hz", 1.0)
    f = 1.0 * bh
    # Yaw + roll out of phase = "hip swivel" feel.
    # Pitch at 2x = "bounce" on top.
    # Big yaw amplitude for the hip-shake effect.
    return {"type": "euler", "params": {
        "x": 0.20 * math.sin(TAU * f * t),                    # roll: shoulders
        "y": 0.12 * math.sin(TAU * 2 * f * t + math.pi/2),    # pitch: bob (cos)
        "z": 0.30 * math.sin(TAU * f * t + math.pi/2),        # yaw: hips (cos)
    }}


# ─── G: Deep moves derived from teleop recordings ──────────────────────
# These amplitudes/frequencies were extracted from analyze_teleop_moves.py
# runs on actual teleop sessions where the Pro firmware accepted them.

def _deep_shoulder_roll(t, ctx):
    """Big shoulder roll like teleop seg #35: ±0.57rad @ 0.38Hz."""
    f = 0.38 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.55 * math.sin(TAU * f * t), "y": 0.0, "z": 0.0}}


def _twist_sway(t, ctx):
    """Roll + yaw locked in phase (seg #22): big body twist."""
    f = 0.37 * ctx.get("beat_hz", 1.0)
    s = math.sin(TAU * f * t)
    return {"type": "euler", "params": {"x": 0.55 * s, "y": 0.0, "z": 0.35 * s}}


def _fast_twist(t, ctx):
    """Faster version of twist_sway (seg #20)."""
    f = 0.55 * ctx.get("beat_hz", 1.0)
    s = math.sin(TAU * f * t)
    return {"type": "euler", "params": {"x": 0.55 * s, "y": 0.0, "z": 0.40 * s}}


def _deep_wave(t, ctx):
    """Big roll + small pitch (seg #34, 22 sec sustained)."""
    f = 0.49 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {
        "x": 0.55 * math.sin(TAU * f * t),
        "y": 0.20 * math.sin(TAU * 1.2 * f * t),
        "z": 0.0,
    }}


def _slow_pitch_bob(t, ctx):
    """Slow deep pitch bob (seg #67): ±0.22rad @ 0.16Hz."""
    f = 0.16 * ctx.get("beat_hz", 1.0)
    return {"type": "euler", "params": {"x": 0.0, "y": 0.25 * math.sin(TAU * f * t), "z": 0.0}}


def _roll_with_double_headbang(t, ctx):
    """Big shoulder roll with two full headbangs per roll cycle.

    Roll uses the deep_shoulder_roll pattern (±0.55 rad @ 0.38 Hz, period ~2.6s).
    A "headbang" is a full down-up swing of the pitch axis, so to fit TWO
    complete bangs into one roll cycle the pitch must oscillate at 4x the
    roll frequency (each bang = half of a 2x pitch cycle... no, that's
    only the down half. We need 2 full up-down-up cycles inside one roll
    period, i.e. pitch_freq = 2 * roll_freq, AND a symmetric sine, not |sin|).

    Wait -- a SINE at 2x roll freq gives exactly 2 full bangs per roll cycle:
        - 1st bang: down at t=T/8, back up by t=3T/8
        - 2nd bang: down at t=5T/8, back up by t=7T/8

    The previous version used abs(sin) which is unidirectional and reads as
    "bowing", not "banging".

    On Pro firmware, +pitch = nose-DOWN, so a standard sine swings the head
    from down to up and back, which is what makes it look like a headbang.
    """
    bh = ctx.get("beat_hz", 1.0)
    f_roll = 0.38 * bh
    roll = 0.55 * math.sin(TAU * f_roll * t)
    # Two full pitch oscillations per roll cycle = pitch_freq = 2 * roll_freq.
    # Use -cos so pitch starts at -0.35 (head UP) when t=0 and the roll is
    # also at zero, then snaps down -> up -> down -> up over one roll cycle.
    pitch = -0.35 * math.cos(TAU * (2.0 * f_roll) * t)
    return {"type": "euler", "params": {"x": roll, "y": pitch, "z": 0.0}}


# ─── F: Punctuation stinger ────────────────────────────────────────────

def _head_pop(t, ctx):
    # 0.4s total: pitch down to -0.25 by t=0.2, return to 0 by t=0.4.
    if t < 0.2:
        pitch = -0.25 * (t / 0.2)
    elif t < 0.4:
        pitch = -0.25 * (1.0 - (t - 0.2) / 0.2)
    else:
        pitch = 0.0
    return {"type": "euler", "params": {"x": 0.0, "y": pitch, "z": 0.0}}


# ─── Catalogue ─────────────────────────────────────────────────────────

CATALOGUE: list[Move] = [
    Move("shoulder_roll", "Shoulder roll: side-to-side body roll", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_shoulder_roll),
    Move("head_nod", "Head nod: yes-style pitch bob", "A",
         duration=8.0, beats_per_cycle=0.67, requires_walking=False, frame=_head_nod),
    Move("head_swivel", "Head swivel: no-style yaw turn", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_head_swivel),
    Move("head_swivel_dip", "Yaw swivel + head bows down at each side", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_head_swivel_dip),
    Move("head_swivel_nod", "Swivel + nod at independent rhythms (1.0 + 1.5 Hz)", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_head_swivel_nod),
    Move("head_swivel_nod_sync", "Swivel + nod locked at same freq (diagonal sweep)", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_head_swivel_nod_sync),
    Move("figure8_head", "Figure-8 head: roll+pitch trace an 8", "A",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_figure8_head),
    Move("headbang", "Fast aggressive nod (metal style)", "A",
         duration=6.0, beats_per_cycle=0.4, requires_walking=False, frame=_headbang),

    # B: pitch-based bob alternatives (work on MCF/Pro; BodyHeight is rejected).
    Move("slow_bob", "Slow pitch bob -- body leans fwd/back", "B",
         duration=8.0, beats_per_cycle=2.0, requires_walking=False, frame=_slow_bob),
    Move("deep_bob", "Deeper, quicker pitch bob", "B",
         duration=6.0, beats_per_cycle=1.0, requires_walking=False, frame=_deep_bob),
    Move("pitch_pop", "Square-wave pitch pop", "B",
         duration=6.0, beats_per_cycle=1.0, requires_walking=False, frame=_pitch_pop),

    Move("hip_shake", "Sideways hip shimmy via Move strafe (needs walking)", "C",
         duration=8.0, beats_per_cycle=1.0, requires_walking=True, frame=_hip_shake),
    Move("hip_shake_yaw", "Hip swivel via Euler yaw (no walking needed)", "A",
         duration=8.0, beats_per_cycle=0.67, requires_walking=False, frame=_hip_shake_yaw),
    Move("circle_walk", "Walks in a tight circle", "C",
         duration=10.0, beats_per_cycle=4.0, requires_walking=True, frame=_circle_walk),

    Move("body_wave", "Roll + pitch + yaw multi-axis wave", "D",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_body_wave),
    Move("reggae_sway", "Slow lazy roll + small pitch sway", "D",
         duration=10.0, beats_per_cycle=1.25, requires_walking=False, frame=_reggae_sway),
    Move("disco_bounce", "Punchy yaw hits + pitch bob", "D",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_disco_bounce),
    Move("booty_dance", "Asymmetric lean-twist: right front low + left hip up, alternates", "D",
         duration=10.0, beats_per_cycle=1.25, requires_walking=False, frame=_booty_dance),

    Move("shakira_combo", "Multi-axis Euler: hip yaw + shoulder roll + pitch bob", "E",
         duration=8.0, beats_per_cycle=1.0, requires_walking=False, frame=_shakira_combo),

    Move("head_pop", "Single sharp pitch flick (stinger)", "F",
         duration=0.5, beats_per_cycle=0.25, requires_walking=False, frame=_head_pop),

    # G: derived from teleop recordings (much larger amplitudes than A-E)
    Move("deep_shoulder_roll", "Full-amplitude shoulder roll (teleop-derived)", "G",
         duration=8.0, beats_per_cycle=2.6, requires_walking=False, frame=_deep_shoulder_roll),
    Move("twist_sway", "Big roll + yaw locked: full body twist", "G",
         duration=8.0, beats_per_cycle=2.7, requires_walking=False, frame=_twist_sway),
    Move("fast_twist", "Quicker twist_sway (0.55 Hz)", "G",
         duration=6.0, beats_per_cycle=1.8, requires_walking=False, frame=_fast_twist),
    Move("deep_wave", "Big roll + small pitch wave (long sustain)", "G",
         duration=12.0, beats_per_cycle=2.0, requires_walking=False, frame=_deep_wave),
    Move("slow_pitch_bob", "Very slow deep pitch bob (breathing)", "G",
         duration=12.0, beats_per_cycle=6.3, requires_walking=False, frame=_slow_pitch_bob),
    Move("roll_double_headbang",
         "Deep shoulder roll with two headbang dips per roll cycle", "G",
         duration=10.5, beats_per_cycle=2.6, requires_walking=False,
         frame=_roll_with_double_headbang),
]


CATALOGUE_BY_NAME = {m.name: m for m in CATALOGUE}


def clamp_params(cmd: dict) -> dict:
    """Apply safety clamps to a command dict (in-place safe copy)."""
    t = cmd.get("type")
    if t == "euler":
        p = cmd["params"]
        cmd = {"type": "euler", "params": {
            "x": _clamp(p.get("x", 0.0), -EULER_MAX, EULER_MAX),
            "y": _clamp(p.get("y", 0.0), -EULER_MAX, EULER_MAX),
            "z": _clamp(p.get("z", 0.0), -EULER_MAX, EULER_MAX),
        }}
    elif t == "body_height":
        d = cmd["params"].get("data", 0.0)
        cmd = {"type": "body_height", "params": {"data": _clamp(d, -HEIGHT_MAX, HEIGHT_MAX)}}
    elif t == "move":
        p = cmd["params"]
        cmd = {"type": "move", "params": {
            "x": _clamp(p.get("x", 0.0), -MOVE_VX_MAX, MOVE_VX_MAX),
            "y": _clamp(p.get("y", 0.0), -MOVE_VY_MAX, MOVE_VY_MAX),
            "z": _clamp(p.get("z", 0.0), -MOVE_VYAW_MAX, MOVE_VYAW_MAX),
        }}
    elif t == "compound":
        cmd = {"type": "compound", "params": [clamp_params(sub) for sub in cmd["params"]]}
    return cmd
