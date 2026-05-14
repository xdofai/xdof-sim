Chess tin box assets
====================

Source GLBs:

- `tin_0`: `vintage_japanese_fujiya_chocolate_tin_box.glb`
- `tin_1`: `yellow_tin_box-freepoly.org.glb`
- `tin_2`: `/Users/adamrasb/Documents/blender/tin_box/tin.glb`

Variants `tin_0` and `tin_1` are split into separate `body/` and `lid/` MJCF
assets. Variant `tin_2` is body-only. The source GLB meshes were converted to
MuJoCo Z-up coordinates, centered in XY with the bottom at Z=0, then exported
as OBJ and processed with `obj2mjcf` using CoACD convex decomposition. The
Fujiya scan was split by height; the yellow tin source already had the body and
lid side-by-side, so it was split by connected component clusters.

The generated MJCF files are standalone assets:

- `tin_*/body/body.xml`
- `tin_*/lid/lid.xml`

`tin_2` intentionally only has `tin_2/body/body.xml`.

Collision geoms are in group 3 with transparent RGBA and task-asset contact
parameters. Visual geoms are in group 2 with contacts disabled.
