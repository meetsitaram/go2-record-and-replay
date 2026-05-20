"""Recording-specific safety overrides.

Importing this module extends ``go2_driver.constants.BLOCKED_COMBOS`` with
actions that would corrupt recording sessions by triggering autonomous motions
(tricks, dances, posture switches) that make the data unusable for imitation
learning.

The driver's ``SafetyFilter`` reads BLOCKED_COMBOS at runtime, so entries
appended here are automatically enforced.
"""

from go2_driver.constants import (
    BLOCKED_COMBOS,
    KEY_A, KEY_B, KEY_X, KEY_Y,
    KEY_L1, KEY_L2, KEY_R1, KEY_R2,
    KEY_LEFT, KEY_RIGHT,
    KEY_SELECT, KEY_START,
)

RECORDING_EXTRA_BLOCKED = [
    # Tricks triggered by shoulder + face button
    (KEY_L2 | KEY_X,      KEY_X,      "Stand up from fall (LT+X)"),
    (KEY_L2 | KEY_SELECT, KEY_SELECT, "Searchlight toggle (LT+Select)"),
    (KEY_R2 | KEY_A,      KEY_A,      "Stretch (RT+A)"),
    (KEY_R2 | KEY_B,      KEY_B,      "Shake hands (RT+B)"),
    (KEY_R2 | KEY_Y,      KEY_Y,      "Love (RT+Y)"),
    (KEY_R1 | KEY_B,      KEY_B,      "Sit down (RB+B)"),
    (KEY_L1 | KEY_A,      KEY_A,      "Greet (LB+A)"),
    (KEY_L1 | KEY_B,      KEY_B,      "Dance (LB+B)"),
    (KEY_L1 | KEY_SELECT, KEY_SELECT, "Endurance mode (LB+Select)"),
    # Stair/mode combos that change gait unexpectedly
    (KEY_RIGHT | KEY_START,  KEY_START,  "Stair mode 1 (D-Right+Start)"),
    (KEY_LEFT  | KEY_SELECT, KEY_SELECT, "Stair mode 2 (D-Left+Select)"),
]


def install_recording_blocks() -> None:
    """Append recording-specific entries to BLOCKED_COMBOS (idempotent)."""
    existing = {desc for _, _, desc in BLOCKED_COMBOS}
    for entry in RECORDING_EXTRA_BLOCKED:
        if entry[2] not in existing:
            BLOCKED_COMBOS.append(entry)


install_recording_blocks()
