"""
Xbox gamepad input via evdev.

Reads joystick axes and buttons, maps them to the Unitree controller protocol,
and applies safety rules (blocked combos, emergency stop, vibration feedback).
"""

import glob
import grp
import json
import os
import sys
import time
import threading

import evdev
import numpy as np

from .constants import (
    KEY_R1, KEY_L1, KEY_START, KEY_SELECT, KEY_R2, KEY_L2,
    KEY_F1, KEY_F2, KEY_A, KEY_B, KEY_X, KEY_Y,
    KEY_UP, KEY_RIGHT, KEY_DOWN, KEY_LEFT,
    ALL_SHOULDERS, ANY_FACE,
    BUTTON_MAP, BLOCKED_COMBOS, BUTTON_ACTIONS,
    COUNTDOWN_SECS, PULSE_TIMES, PULSE_MS,
    STICK_CENTER, STICK_DEADZONE, STICK_RANGE,
    TRIGGER_THRESHOLD,
    SEND_RATE,
)


# ── Controller state ─────────────────────────────────────────────────────────


class ControllerState:
    """Thread-safe container for the latest gamepad values."""

    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.keys: int = 0
        self.lock = threading.Lock()

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "lx": self.lx,
                "ly": self.ly,
                "rx": self.rx,
                "ry": self.ry,
                "keys": self.keys,
            }

    def set_button(self, mask: int, pressed: bool):
        with self.lock:
            if pressed:
                self.keys |= mask
            else:
                self.keys &= ~mask

    def set_axis(self, axis: str, value: float):
        with self.lock:
            setattr(self, axis, max(-1.0, min(1.0, value)))

    def action_array(self) -> np.ndarray:
        """Return [lx, ly, rx, ry] as a float32 numpy array."""
        with self.lock:
            return np.array([self.lx, self.ly, self.rx, self.ry], dtype=np.float32)

    def buttons_array(self) -> np.ndarray:
        """Return [keys_bitmask] as an int32 numpy array."""
        with self.lock:
            return np.array([self.keys], dtype=np.int32)


# ── Stick normalisation ──────────────────────────────────────────────────────


def normalize_stick(raw: int) -> float:
    """Convert raw stick value (0-65535, center 32768) to +/-1.0 with deadzone."""
    centered = raw - STICK_CENTER
    if abs(centered) < STICK_DEADZONE:
        return 0.0
    sign = 1.0 if centered > 0 else -1.0
    magnitude = (abs(centered) - STICK_DEADZONE) / (STICK_RANGE - STICK_DEADZONE)
    return sign * min(1.0, magnitude)


# ── Rumble / haptic feedback ─────────────────────────────────────────────────


class RumbleHelper:
    """Manages a single FF_RUMBLE effect for haptic feedback."""

    def __init__(self, device: evdev.InputDevice):
        self.device = device
        self.effect_id = None
        try:
            caps = device.capabilities()
            if (evdev.ecodes.EV_FF, 21) not in caps and evdev.ecodes.EV_FF not in caps:
                return
            effect = evdev.ff.Effect(
                evdev.ecodes.FF_RUMBLE, -1, 0,
                evdev.ff.Trigger(0, 0),
                evdev.ff.Replay(PULSE_MS, 0),
                evdev.ff.EffectType(
                    ff_rumble_effect=evdev.ff.Rumble(
                        strong_magnitude=0xFFFF, weak_magnitude=0xFFFF,
                    )
                ),
            )
            self.effect_id = device.upload_effect(effect)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self.effect_id is not None

    def pulse(self):
        if self.effect_id is not None:
            try:
                self.device.write(evdev.ecodes.EV_FF, self.effect_id, 1)
            except Exception:
                pass

    def stop(self):
        if self.effect_id is not None:
            try:
                self.device.write(evdev.ecodes.EV_FF, self.effect_id, 0)
            except Exception:
                pass

    def cleanup(self):
        self.stop()
        if self.effect_id is not None:
            try:
                self.device.erase_effect(self.effect_id)
            except Exception:
                pass


# ── Device discovery ─────────────────────────────────────────────────────────


def check_device_permissions() -> dict | None:
    """Check if any /dev/input/event* devices exist but are unreadable."""
    all_events = sorted(glob.glob("/dev/input/event*"))
    if not all_events:
        return None
    unreadable = [p for p in all_events if not os.access(p, os.R_OK)]
    if not unreadable:
        return None

    user = os.environ.get("USER", "unknown")
    try:
        input_gid = grp.getgrnam("input").gr_gid
        in_input_group = input_gid in os.getgroups()
    except KeyError:
        in_input_group = False

    return {
        "total": len(all_events),
        "unreadable": len(unreadable),
        "in_input_group": in_input_group,
        "user": user,
    }


def find_gamepad() -> evdev.InputDevice | None:
    """Find the first gamepad/joystick among evdev input devices."""
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        caps = device.capabilities()
        has_abs = evdev.ecodes.EV_ABS in caps
        has_key = evdev.ecodes.EV_KEY in caps
        if has_abs and has_key:
            key_codes = [k if isinstance(k, int) else k[0] for k in caps[evdev.ecodes.EV_KEY]]
            if 304 in key_codes:
                return device
    return None


def validate_gamepad(device: evdev.InputDevice) -> list[str]:
    """Check that the gamepad has the expected Xbox-style buttons and axes."""
    caps = device.capabilities()
    warnings = []

    key_codes = set()
    if evdev.ecodes.EV_KEY in caps:
        for k in caps[evdev.ecodes.EV_KEY]:
            key_codes.add(k if isinstance(k, int) else k[0])

    expected_buttons = {
        304: "A (BTN_SOUTH)", 305: "B (BTN_EAST)",
        307: "X (BTN_NORTH)", 308: "Y (BTN_WEST)",
        310: "LB (BTN_TL)", 311: "RB (BTN_TR)",
        314: "Back (BTN_SELECT)", 315: "Start (BTN_START)",
        317: "L-stick click (BTN_THUMBL)", 318: "R-stick click (BTN_THUMBR)",
    }
    missing = {code: name for code, name in expected_buttons.items() if code not in key_codes}
    if missing:
        warnings.append(f"Missing buttons: {', '.join(missing.values())}")

    abs_codes = set()
    if evdev.ecodes.EV_ABS in caps:
        for item in caps[evdev.ecodes.EV_ABS]:
            abs_codes.add(item[0] if isinstance(item, tuple) else item)

    if evdev.ecodes.ABS_X not in abs_codes or evdev.ecodes.ABS_Y not in abs_codes:
        warnings.append("Missing left stick axes (ABS_X / ABS_Y)")

    has_zrz = evdev.ecodes.ABS_Z in abs_codes and evdev.ecodes.ABS_RZ in abs_codes
    has_rxry = evdev.ecodes.ABS_RX in abs_codes and evdev.ecodes.ABS_RY in abs_codes
    if not has_zrz and not has_rxry:
        warnings.append("Missing right stick axes (expected ABS_Z/ABS_RZ or ABS_RX/ABS_RY)")

    if evdev.ecodes.ABS_HAT0X not in abs_codes or evdev.ecodes.ABS_HAT0Y not in abs_codes:
        warnings.append("Missing D-pad axes (ABS_HAT0X / ABS_HAT0Y)")

    name_lower = device.name.lower()
    if not any(kn in name_lower for kn in ["xbox", "microsoft", "x-box"]):
        warnings.append(
            f"Controller '{device.name}' may not be Xbox-compatible — "
            f"button/axis mapping could be wrong"
        )

    return warnings


# ── Gamepad event loop ────────────────────────────────────────────────────────


def gamepad_loop(device: evdev.InputDevice, state: ControllerState, stop_event: threading.Event):
    """Read evdev gamepad events and update controller state. Blocks until stop_event or Ctrl+C."""
    EV_ABS = evdev.ecodes.EV_ABS
    EV_KEY = evdev.ecodes.EV_KEY

    for event in device.read_loop():
        if stop_event.is_set():
            break

        if event.type == EV_KEY and event.code in BUTTON_MAP:
            state.set_button(BUTTON_MAP[event.code], event.value == 1)

        elif event.type == EV_ABS:
            code, val = event.code, event.value

            if code == evdev.ecodes.ABS_X:
                state.set_axis("lx", normalize_stick(val))
            elif code == evdev.ecodes.ABS_Y:
                state.set_axis("ly", -normalize_stick(val))
            elif code == evdev.ecodes.ABS_Z:
                state.set_axis("rx", normalize_stick(val))
            elif code == evdev.ecodes.ABS_RZ:
                state.set_axis("ry", -normalize_stick(val))
            elif code == evdev.ecodes.ABS_RX:
                state.set_axis("rx", normalize_stick(val))
            elif code == evdev.ecodes.ABS_RY:
                state.set_axis("ry", -normalize_stick(val))
            elif code == evdev.ecodes.ABS_BRAKE:
                state.set_button(KEY_L2, val > TRIGGER_THRESHOLD)
            elif code == evdev.ecodes.ABS_GAS:
                state.set_button(KEY_R2, val > TRIGGER_THRESHOLD)
            elif code == evdev.ecodes.ABS_HAT0Y:
                state.set_button(KEY_UP,   val == -1)
                state.set_button(KEY_DOWN, val == 1)
            elif code == evdev.ecodes.ABS_HAT0X:
                state.set_button(KEY_LEFT,  val == -1)
                state.set_button(KEY_RIGHT, val == 1)

        _print_state(state)


# ── Status display ────────────────────────────────────────────────────────────


def _lookup_action(keys: int) -> str:
    for combo_mask, action in BUTTON_ACTIONS:
        if (keys & combo_mask) == combo_mask:
            return action
    return ""


def _print_state(state: ControllerState):
    s = state.to_dict()
    pressed = []
    for name, mask in [
        ("L1", KEY_L1), ("L2", KEY_L2), ("R1", KEY_R1), ("R2", KEY_R2),
        ("A", KEY_A), ("B", KEY_B), ("X", KEY_X), ("Y", KEY_Y),
        ("F1", KEY_F1), ("F2", KEY_F2),
        ("Start", KEY_START), ("Select", KEY_SELECT),
        ("Up", KEY_UP), ("Down", KEY_DOWN), ("Left", KEY_LEFT), ("Right", KEY_RIGHT),
    ]:
        if s["keys"] & mask:
            pressed.append(name)

    action = _lookup_action(s["keys"])
    action_str = f" -> {action}" if action else ""

    line = (
        f"  LX:{s['lx']:+.2f} LY:{s['ly']:+.2f} | "
        f"RX:{s['rx']:+.2f} RY:{s['ry']:+.2f} | "
        f"Buttons: {', '.join(pressed) if pressed else 'none'}{action_str}"
    )
    sys.stdout.write(f"\r{line}\033[K")
    sys.stdout.flush()


# ── Safety filter ─────────────────────────────────────────────────────────────


class SafetyFilter:
    """Applies blocked-combo rules, emergency stop, and speed limiting to controller output."""

    def __init__(self, allow_all: bool, speed_limit: float, rumble: RumbleHelper | None,
                 conn=None, loop=None, dry_run: bool = False):
        self.allow_all = allow_all
        self.speed_limit = speed_limit
        self.rumble = rumble
        self.conn = conn
        self.loop = loop
        self.dry_run = dry_run

        self._blocked_warned: set[str] = set()
        self._estop_sent = False
        self._estop_cd: dict | None = None
        self._countdowns: dict[str, dict] = {}
        self._armed_combos: set[str] = set()
        self._was_start = False
        self._was_select = False
        self._is_walking = False

    def apply(self, state_dict: dict) -> dict:
        """Apply safety rules to a controller state dict. Returns the (possibly modified) dict."""
        s = dict(state_dict)
        if self.speed_limit < 1.0 and self._is_walking:
            lim = self.speed_limit
            s["lx"] = max(-lim, min(lim, s["lx"]))
            s["ly"] = max(-lim, min(lim, s["ly"]))
            s["rx"] = max(-lim, min(lim, s["rx"]))
            s["ry"] = max(-lim, min(lim, s["ry"]))

        keys = s["keys"]

        # Emergency stop: all shoulders + any face button
        if (keys & ALL_SHOULDERS) == ALL_SHOULDERS and (keys & ANY_FACE):
            keys, s = self._handle_estop(keys, s)
        else:
            self._estop_sent = False
            self._estop_cd = None
            keys = self._handle_combos(keys)

        s["keys"] = keys

        # Walk/Stand mode tracking
        start_pressed = bool(s["keys"] & KEY_START)
        select_pressed = bool(s["keys"] & KEY_SELECT)
        if start_pressed and not self._was_start:
            self._is_walking = True
            limit_note = f"  (speed limit {self.speed_limit:.0%})" if self.speed_limit < 1.0 else ""
            sys.stdout.write(f"\n  [WALK mode]{limit_note}\n")
            sys.stdout.flush()
            if self.rumble:
                self.rumble.pulse()
        if select_pressed and not self._was_select:
            self._is_walking = False
            sys.stdout.write("\n  [STAND mode]\n")
            sys.stdout.flush()
            if self.rumble:
                self.rumble.pulse()
        self._was_start = start_pressed
        self._was_select = select_pressed

        return s

    def _handle_estop(self, keys: int, s: dict) -> tuple[int, dict]:
        if not self._estop_sent:
            if self._estop_cd is None:
                self._estop_cd = {"start": time.monotonic(), "pulses_fired": 0}
                sys.stdout.write("\n")
                print("  E-STOP: hold for 3...2...1... (release to cancel)")

            elapsed = time.monotonic() - self._estop_cd["start"]
            for i, pt in enumerate(PULSE_TIMES):
                if self._estop_cd["pulses_fired"] <= i and elapsed >= pt:
                    if self.rumble:
                        self.rumble.pulse()
                    self._estop_cd["pulses_fired"] = i + 1

            if elapsed >= COUNTDOWN_SECS:
                sys.stdout.write("\n")
                print("  EMERGENCY STOP -- sending Damp (all motors off)")
                if not self.dry_run and self.conn and self.loop:
                    self._send_damp()
                self._estop_sent = True

        s["keys"] = 0
        s["lx"] = s["ly"] = s["rx"] = s["ry"] = 0.0
        self._countdowns.clear()
        self._armed_combos.clear()
        return 0, s

    def _send_damp(self):
        import asyncio
        from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
        try:
            coro = self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD.get("Damp", 1001)},
            )
            asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=2)
        except Exception:
            pass

    def _handle_combos(self, keys: int) -> int:
        if self.allow_all:
            active_descs: set[str] = set()
            for combo_mask, strip_mask, desc in BLOCKED_COMBOS:
                if (keys & combo_mask) != combo_mask:
                    continue
                active_descs.add(desc)
                if desc in self._armed_combos:
                    continue

                if desc not in self._countdowns:
                    self._countdowns[desc] = {"start": time.monotonic(), "pulses_fired": 0}
                    sys.stdout.write("\n")
                    print(f"  ARMED: {desc} -- hold for 3...2...1...")

                cd = self._countdowns[desc]
                elapsed = time.monotonic() - cd["start"]
                for i, pt in enumerate(PULSE_TIMES):
                    if cd["pulses_fired"] <= i and elapsed >= pt:
                        if self.rumble:
                            self.rumble.pulse()
                        cd["pulses_fired"] = i + 1

                if elapsed < COUNTDOWN_SECS:
                    keys &= ~strip_mask
                else:
                    self._armed_combos.add(desc)
                    sys.stdout.write("\n")
                    print(f"  SENT: {desc}")

            released = [d for d in list(self._countdowns) if d not in active_descs]
            for d in released:
                del self._countdowns[d]
                self._armed_combos.discard(d)
        else:
            for combo_mask, strip_mask, desc in BLOCKED_COMBOS:
                if (keys & combo_mask) == combo_mask:
                    keys &= ~strip_mask
                    if desc not in self._blocked_warned:
                        self._blocked_warned.add(desc)
                        sys.stdout.write("\n")
                        print(f"  BLOCKED: {desc}  (use --allow-all to override)")
                        if self.rumble:
                            self.rumble.pulse()

        return keys
