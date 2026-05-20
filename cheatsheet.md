# Unitree Go2 Xbox Controller Cheat Sheet

## Movement

| Button | Action |
|--------|--------|
| Left Stick | Walk (forward/back/strafe) |
| Right Stick | Turn (yaw) / Look (pitch) |
| Start | Walking mode |
| Select | Standing mode (stop walking) |

## Tricks (Air + Pro)

| Combo | Action |
|-------|--------|
| LT + A | Lock posture (stand/crouch toggle) |
| LT + X | Stand up from fall |
| LT + B | ⚠️ Damp (motors OFF — robot collapses) |
| LT + Select | Searchlight toggle |
| RT + A | Stretch |
| RT + B | Shake hands |
| RT + Y | Love (heart gesture) |
| LB + A | Greet (wave hello) |
| LB + B | Dance 1 |
| LB + X | Dance 2 |
| RB + A | Jump forward |
| RB + B | Sit down |
| RB + X | Pounce |

## Pro Only

| Combo | Action |
|-------|--------|
| R1 (Double Click) | Handstand (stand on front legs) |
| R2 (Double Click) | Erect (stand on hind legs) |
| X (Double Click) | Bound / Jump |
| Y (Double Click) | Bound / Jump |
| B (Double Click) | Cross Step (1 front + 1 back leg, avoidance OFF) |
| A (Double Click) | Free Avoid (back to 4 legs, avoidance ON) |

## Modes

| Combo | Action |
|-------|--------|
| Start | Unlock / Default Gait |
| Select | Standing pose |
| L2 (hold) + Start | Running mode (higher speed) |
| L1 (hold) + Start | Normal Gait |
| Right (hold) + Start | Classic Gait |
| Left (hold) + Start | Free Walk |
| LB + Select | Endurance mode |

## Recording (record.py)

| Button | Action |
|--------|--------|
| F1 (L-stick click) | Toggle recording on/off |
| Ctrl+C | Stop and save |

## Safety (--allow-all flag)

- **With** `--allow-all`: Jump, Pounce, Damp require ~2s hold (rumble countdown)
- **Without** `--allow-all`: Jump, Pounce, Damp are fully BLOCKED

## Air vs Pro Differences

- Air cannot do Handstand (RT+X), Cross-step, or Free-avoid
- Pro has faster max speed (~3.5 m/s vs ~2.5 m/s)
- All other tricks work on both
