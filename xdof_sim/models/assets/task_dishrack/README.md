# Task-Local Dishrack Assets

This directory is a task-scoped asset namespace for the `dishrack` scene.

Layout:

- `plate/current/`
  Current baked plate asset used by `yam_dishwasher_scene.xml`.
- `plate/plate_*/`
  Robocasa plate variants copied from `assets_robocasa/objaverse/objaverse/plate/`.
- `dish_rack/current/`
  Current baked dish rack asset used by `yam_dishwasher_scene.xml`.
- `dish_rack/DishRack*/`
  Robocasa dish-rack variants copied from
  `assets_robocasa/objects_lightwheel/lightwheel/dish_rack/`.

The `current/` directories are intended to act as the explicit default variants
when the dishrack task moves to a placeholder-based base scene.
