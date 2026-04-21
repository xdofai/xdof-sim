# Data Collection Randomization

This document summarizes the randomization currently used for the 10 explicit
data-collection task families in `xdof-sim`.

The collection families are defined in `xdof_sim/collection_tasks.py`. Most of
them use a registry randomizer from `xdof_sim/randomization.py`. Two important
exceptions are:

- `sweep`: now uses the normal registry path, including pose randomization
  with the same rejection-sampling structure as the other scenes
- `random_object_handover`: uses the special `inhand_transfer` model-swapping
  path instead of the normal `TASK_RANDOMIZERS` registry

## Overview

| Collection family | `env_task` | Sim task specs / prompts | Randomization mode |
| --- | --- | --- | --- |
| Chess | `chess` | `set_up_chess_pieces_on_board` | Registry |
| Spelling With Blocks | `blocks` | `spell_cat`, `spell_dog`, `spell_fish`, `spell_bair`, `spell_xdof`, `spell_abc`, `spell_yam`, `spell_agi` | Registry |
| Marker In Drawer | `drawer` | `put_markers_in_top_drawer`, `put_markers_in_middle_drawer`, `put_markers_in_bottom_drawer` | Registry |
| Dish Rack | `dishrack` | `load_plates_into_dish_rack` | Registry |
| Sweep | `sweep` | `sweep_away_paper_scraps_from_table` | Registry |
| Place Bottle In Bin | `bottles` | `throw_plastic_bottles_in_bin` | Registry |
| Flip Mug | `mug_flip` | `turn_mug_right_side_up` | Registry |
| Mug On Tree | `mug_tree` | `hang_mug_on_mug_rack` | Registry |
| Pouring Beads | `pour` | `pouring` | Registry |
| Random Object Handover | `inhand_transfer` | no separate `SimTaskSpec`; collection family targets the special handover task path | Special |

## Shared Sampling Logic

All registry randomizers inherit from `SceneRandomizer` and use the same base
algorithm:

1. Start from the nominal XML pose after `mj_resetData`.
2. Sample deltas in `x`, `y`, `z`, and yaw for each randomized object or fixed
   body.
3. Clamp the sampled XY positions to the default table workspace:
   `x in [0.36, 0.82]`, `y in [-0.55, 0.55]`.
4. Reject samples that fail the task's pairwise XY clearance threshold.
5. Apply the candidate state and run a MuJoCo contact check.
6. Retry up to `200` times.

If no collision-free placement is found after `200` tries, the code logs a
warning and keeps the last sampled placement anyway.

### Important defaults

- If a `PerturbRange` does not specify `delta_yaw`, it gets the default
  `(-pi, pi)`, so that object receives full yaw randomization.
- `fixed_body=True` means the randomizer writes directly to `model.body_pos`
  / `model.body_quat` instead of writing through a free joint.
- Some tasks add custom post-processing on top of the base sampler, such as
  shifting dependent objects, mirroring the scene, or randomizing materials.

### Shared size randomization

- All 10 collection families now apply a small multiplicative size perturbation
  to movable free-jointed task objects.
- The scale factor is sampled independently per movable object:
  `scale ~ U(0.95, 1.05)`.
- This is implemented by regenerating the scene XML on reset, duplicating the
  referenced mesh assets for the affected object, and scaling both mesh assets
  and primitive geom sizes.
- Fixed scene furniture that is only moved as a fixed body, such as the chess
  board, drawer frame, dish rack, tray, and mug tree, is not size-randomized.
- For `random_object_handover`, the imported handover object also gets its own
  `U(0.95, 1.05)` scale factor during XML construction.

## Per-Task Breakdown

### 1. Chess

- Underlying randomizer: `ChessRandomizer`
- Pairwise clearance: `0.056 m`
- Randomized objects:
  - `chessboard` as a fixed body
  - all 32 chess pieces as free-jointed objects

#### Chessboard

- `delta_x = [-0.05, 0.05]`
- `delta_y = [-0.10, 0.10]`
- `delta_yaw = [-0.15, 0.15]`

#### Pieces

- Each piece starts with very large `delta_x = [-1.0, 1.0]`,
  `delta_y = [-1.0, 1.0]`
- In practice this means "anywhere on the table bounds" after clamping
- Each piece also gets full default yaw randomization: `delta_yaw = [-pi, pi]`

#### Custom logic

- The board is sampled first
- Each piece is then sampled with per-piece rejection until it is:
  - outside the board's new footprint
  - clear of all previously placed pieces
- The board exclusion uses a board half-extent of `0.224 m` plus a
  `0.05 m` margin
- The per-piece off-board sampler allows up to `50` tries per piece before
  falling back to the last sampled pose

### 2. Spelling With Blocks

- Underlying randomizer: `BlocksRandomizer`
- Pairwise clearance: `0.01 m`
- Words currently used:
  - `cat`
  - `dog`
  - `fish`
  - `bair`
  - `xdof`
  - `abc`
  - `yam`
  - `agi`

#### Blocks

- All 26 letter blocks `A-Z` are randomized every episode
- This is true even for spelling tasks that only care about one target word
- For each block:
  - `delta_x = [-0.08, 0.08]`
  - `delta_y = [-0.45, 0.45]`
  - `delta_yaw = [-0.25, 0.25]`

#### Notes

- The spelling task family is just a prompt/spec layer on top of the normal
  `blocks` scene
- There is no separate spelling scene randomizer anymore

### 3. Marker In Drawer

- Underlying randomizer: `DrawerRandomizer`
- Pairwise clearance: `0.04 m`
- Applies to all three prompt variants:
  - top drawer
  - middle drawer
  - bottom drawer

#### Drawer frame

- Randomized as fixed body `drawer_body`
- `delta_x = [-0.08, 0.0]`
- `delta_y = [-0.10, 0.10]`
- `delta_yaw = [-0.3, 0.3]`

#### Markers

- Randomizes five markers:
  - `marker_1_joint`
  - `marker_2_joint`
  - `marker_3_joint`
  - `marker_4_joint`
  - `marker_5_joint`
- For each marker:
  - `delta_x = [-0.08, 0.08]`
  - `delta_y = [-0.10, 0.10]`
  - `delta_yaw = [-pi, pi]` by default

#### Custom logic

- After sampling the drawer frame, all markers are shifted by the same drawer
  `(dx, dy)` so they remain near the drawer after randomization

### 4. Dish Rack

- Underlying randomizer: `DishRackRandomizer`
- Pairwise clearance: `0.10 m`

#### Dish rack

- Randomized as fixed body `dishrack`
- `delta_x = [-0.10, 0.09]`
- `delta_y = [-0.15, 0.15]`
- `delta_yaw = [-0.25, 0.25]`

#### Plate

- Randomized as free-jointed `plate_joint`
- `delta_x = [-0.35, 0.25]`
- `delta_y = [-0.35, 0.35]`
- `delta_yaw = [-pi, pi]` by default

#### Custom logic

- 50% chance of mirroring the scene about the XZ plane:
  - negate all Y coordinates
  - rotate the rack by `180 deg` around Z
- The plate material `plate_mat` is randomized every episode from an
  8-color tint palette:
  - white
  - warm cream
  - off-white / linen
  - powder blue
  - sage green
  - blush pink
  - lavender
  - slate grey-blue

### 5. Sweep

- Underlying randomizer: `SweepRandomizer`
- Pairwise clearance:
  - trash-trash: `0.018 m`
  - tool-tool (`brush`, `bin`, `dustpan`): `0.08 m`
  - tool-trash: `0.04 m`

#### Main tools

- `brush_jnt`
  - `delta_x = [-0.08, 0.10]`
  - `delta_y = [-0.14, 0.12]`
  - `delta_yaw = [-0.6, 0.6]`
- `bin_joint`
  - `delta_x = [-0.08, 0.05]`
  - `delta_y = [-0.20, 0.20]`
  - `delta_yaw = [-0.5, 0.5]`
- `dustpan_jnt`
  - `delta_x = [-0.08, 0.04]`
  - `delta_y = [-0.12, 0.16]`
  - `delta_yaw = [-0.6, 0.6]`

#### Trash

- `trash_1_jnt` through `trash_6_jnt` are randomized as a loose cluster
- First, the whole cluster is shifted together by:
  - `cluster_dx = [-0.03, 0.07]`
  - `cluster_dy = [-0.18, 0.18]`
- Then each trash piece gets an additional local jitter:
  - `delta_x = [-0.02, 0.02]`
  - `delta_y = [-0.02, 0.02]`
  - default full yaw: `delta_yaw = [-pi, pi]`

#### Notes

- This keeps the debris task-like instead of scattering each piece
  independently across the whole table
- The same Stage 2 MuJoCo contact rejection is used as in the other scenes
- Sweep also continues to receive the shared per-object size perturbation
  `scale ~ U(0.95, 1.05)` for the brush, bin, dustpan, and all six trash pieces

### 6. Place Bottle In Bin

- Underlying randomizer: `BottlesRandomizer`
- Pairwise clearance: `0.08 m`

#### Bottles

- Randomizes four bottle joints:
  - `bottle_1_joint`
  - `bottle_2_joint`
  - `bottle_3_joint`
  - `bottle_4_joint`
- Each bottle uses:
  - `delta_x = [-1.0, 1.0]`
  - `delta_y = [-1.0, 1.0]`
  - full default yaw randomization: `delta_yaw = [-pi, pi]`
- In practice, this means bottles can land anywhere on the allowed table
  workspace after clamping

#### Bin

- Randomized as free-jointed `bin_joint`
- `delta_x = [-0.05, 0.03]`
- `delta_y = [-0.20, 0.20]`
- `delta_yaw = [-0.5, 0.5]`

### 7. Flip Mug

- Underlying randomizer: `MugFlipRandomizer`
- Pairwise clearance: `0.03 m`

#### Tray

- Randomized as fixed body `tray`
- `delta_x = [-0.10, 0.05]`
- `delta_y = [-0.30, 0.30]`
- `delta_yaw = [-0.25, 0.25]`

#### Mugs

- `mug_1_jnt`
  - relative `delta_x = [-0.035, 0.000]`
  - relative `delta_y = [-0.005, 0.025]`
  - full default yaw randomization: `delta_yaw = [-pi, pi]`
- `mug_2_jnt`
  - relative `delta_x = [-0.000, 0.035]`
  - relative `delta_y = [-0.025, 0.005]`
  - full default yaw randomization: `delta_yaw = [-pi, pi]`

#### Custom logic

- The tray is sampled first
- Each mug is then placed relative to the tray's new pose, using:
  - the mug's original tray-relative offset
  - plus a small extra mug-local delta
- This keeps the mugs on the tray after randomization
- Mug colors are randomized with probability `1.0` from the shared mug color
  palette:
  - original green
  - red-orange
  - amber
  - purple
  - sky blue
  - dark red
  - blue
  - near-white
  - near-black

### 8. Mug On Tree

- Underlying randomizer: `MugTreeRandomizer`
- Pairwise clearance: `0.12 m`

#### Mug rack

- Randomized as fixed body `mug_tree`
- `delta_x = [-0.10, 0.10]`
- `delta_y = [-0.20, 0.20]`
- `delta_yaw = [-0.25, 0.25]`

#### Mugs

- `mug_1_jnt`
  - `delta_x = [-0.10, 0.10]`
  - `delta_y = [-0.30, 0.30]`
  - `delta_yaw = [-pi, pi]` by default
- `mug_2_jnt`
  - `delta_x = [-0.10, 0.10]`
  - `delta_y = [-0.30, 0.30]`
  - `delta_yaw = [-pi, pi]` by default

#### Custom logic

- Mug materials `mug_1_color` and `mug_2_color` are randomized with
  probability `1.0` from the shared mug palette

### 9. Pouring Beads

- Underlying randomizer: `PourRandomizer`
- Pairwise clearance: `0.08 m`

#### Mug and cup

- `mug_1_jnt`
  - `delta_x = [-0.12, 0.12]`
  - `delta_y = [-0.15, 0.15]`
  - `delta_yaw = [-pi, pi]` by default
- `cup_1_jnt`
  - `delta_x = [-0.10, 0.10]`
  - `delta_y = [-0.20, 0.20]`
  - `delta_yaw = [-pi, pi]` by default

#### Beads

- Ten bead joints are tracked:
  - `bead_1_jnt` through `bead_10_jnt`
- Beads are not independently sampled
- Instead, every bead is translated by the same mug `(dx, dy)` so the beads
  stay inside the mug after reset
- Bead quaternions are kept at their nominal values

#### Custom logic

- Only mug and cup count as the main randomized bodies for pairwise/contact
  checks
- Mug and cup body colors are randomized with probability `1.0` from the
  shared mug palette using materials:
  - `mug_1_color`
  - `cup_body_color`

### 10. Random Object Handover

- Randomization mode: special
- Underlying implementation: `InHandTransferRandomizer`
- This task does not use `PerturbRange` or the normal registry randomizer path
- Instead, it rebuilds the full MuJoCo XML on every reset

#### Object categories

Current approved categories:

- `dish_brush`
- `whisk`
- `salt_and_pepper_shaker`
- `cream_cheese_stick`
- `cheese_grater`
- `pizza_cutter`
- `rolling_pin`
- `water_bottle`
- `can`
- `ladle`

#### Current local asset counts

After downloading the RoboCasa packs in this checkout, the currently available
variant counts are:

| Category | Variants |
| --- | ---: |
| `dish_brush` | 6 |
| `whisk` | 6 |
| `salt_and_pepper_shaker` | 24 |
| `cream_cheese_stick` | 12 |
| `cheese_grater` | 12 |
| `pizza_cutter` | 6 |
| `rolling_pin` | 6 |
| `water_bottle` | 21 |
| `can` | 14 |
| `ladle` | 5 |

Total currently available variants: `112`

#### Sampling behavior

- Sample one category uniformly from the 10 categories
- Sample one variant directory uniformly from the available variants in that category
- Sample transfer side:
  - left with probability `0.5`
  - right with probability `0.5`
- Sample object pose:
  - `x ~ U(0.42, 0.78)`
  - if left side: `y ~ U(0.10, 0.40)`
  - if right side: `y ~ U(-0.40, -0.10)`
  - `yaw ~ U(-pi, pi)`
  - fixed object height `z = 0.82`
- Sample object size:
  - `scale ~ U(0.95, 1.05)`

#### XML build behavior

- Start from `yam_inhand_transfer_base.xml`
- Parse the chosen object's `model.xml`
- Copy its meshes, textures, and materials into the scene
- Insert a single free-jointed `task_object` body
- Assign collision properties to every imported collision geom:
  - `density = 30.0`
  - `friction = "3.0 0.3 0.05"`
  - `solimp = "0.95 0.99 0.001"`
  - `solref = "0.004 1"`

#### Important difference from the other 9 families

- The other collection families randomize poses inside a fixed scene model
- Handover swaps the actual MuJoCo model on every reset
- Scene transforms such as `clean`, `mocap`, and `flexible_gripper` are now
  reapplied to each regenerated handover XML before the model reload

## Quick Practical Notes

- All collection scenes now include size randomization for movable objects.
- The default size factor is `U(0.95, 1.05)` per object.
- The collection spelling tasks do not reduce the block set to the target word.
  They randomize the full 26-letter block scene.
- The three drawer prompt variants share exactly the same randomizer.
- `random_object_handover` is the only collection family that currently rebuilds
  the entire scene model on reset instead of perturbing a fixed set of joints.
