#!/usr/bin/env python3
"""Regenerate dishrack plate collision meshes from combined visual geometry.

This keeps each variant's existing visual assets and materials, but replaces
the imported collision decomposition with a fresh CoACD pass over the combined
plate mesh. The result is a cleaner, capped set of convex collision hulls that
matches the current plate asset more closely.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import trimesh
from obj2mjcf.cli import CoacdArgs, decompose_convex


REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_ROOT = REPO_ROOT / "xdof_sim" / "models" / "assets" / "task_dishrack" / "plate"


def _mesh_local_name(mesh_elem: ET.Element) -> str:
    name = mesh_elem.get("name")
    if name:
        return name
    file_attr = mesh_elem.get("file")
    if not file_attr:
        raise ValueError("Mesh asset is missing both name and file attributes")
    return Path(file_attr).stem


def _is_collision_mesh_file(file_attr: str) -> bool:
    file_path = Path(file_attr)
    return "collision" in file_path.parts or "_collision_" in file_path.stem


def _collision_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected collision mesh filename: {path.name}") from exc


def _find_object_body(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Variant model.xml is missing <worldbody>")

    for candidate_name in ("object", "model"):
        body = worldbody.find(f".//body[@name='{candidate_name}']")
        if body is not None:
            return body

    for body in worldbody.iter("body"):
        if any(child.tag == "geom" for child in body.iter()):
            return body
    raise ValueError("Variant model.xml does not contain a body with geoms")


def _remove_collision_geoms(body: ET.Element, collision_mesh_names: set[str]) -> list[ET.Element]:
    removed: list[ET.Element] = []
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            if child.get("mesh") not in collision_mesh_names:
                continue
            removed.append(child)
            parent.remove(child)
    return removed


def _visual_mesh_files(asset_root: ET.Element, variant_dir: Path) -> list[Path]:
    visual_files: list[Path] = []
    for mesh in asset_root.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr or _is_collision_mesh_file(file_attr):
            continue
        visual_files.append((variant_dir / file_attr).resolve())
    if not visual_files:
        raise ValueError(f"No visual mesh files found under {variant_dir}")
    return visual_files


def _combine_visual_meshes(mesh_files: list[Path], output_path: Path) -> None:
    meshes = [trimesh.load(mesh_file, force="mesh", process=False) for mesh_file in mesh_files]
    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)


def _regenerate_variant(
    variant_dir: Path,
    *,
    threshold: float,
    max_convex_hulls: int,
    seed: int,
) -> int:
    model_xml_path = variant_dir / "model.xml"
    root = ET.parse(model_xml_path).getroot()
    asset_root = root.find("asset")
    if asset_root is None:
        raise ValueError(f"{model_xml_path} is missing <asset>")
    object_body = _find_object_body(root)

    old_collision_mesh_elems = [
        mesh for mesh in asset_root.findall("mesh") if _is_collision_mesh_file(mesh.get("file", ""))
    ]
    if not old_collision_mesh_elems:
        raise ValueError(f"{model_xml_path} does not define collision mesh assets")
    old_collision_mesh_names = {_mesh_local_name(mesh) for mesh in old_collision_mesh_elems}
    old_collision_geoms = _remove_collision_geoms(object_body, old_collision_mesh_names)
    if not old_collision_geoms:
        raise ValueError(f"{model_xml_path} does not define collision geoms")

    scale = old_collision_mesh_elems[0].get("scale") or "1 1 1"
    refquat = old_collision_mesh_elems[0].get("refquat")
    collision_geom_template = {
        key: value
        for key, value in old_collision_geoms[0].attrib.items()
        if key not in {"mesh", "name"}
    }

    for mesh in old_collision_mesh_elems:
        asset_root.remove(mesh)

    visual_mesh_files = _visual_mesh_files(asset_root, variant_dir)
    collision_dir = variant_dir / "collision"
    shutil.rmtree(collision_dir, ignore_errors=True)
    collision_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{variant_dir.name}_coacd_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        combined_obj = temp_dir / "model.obj"
        _combine_visual_meshes(visual_mesh_files, combined_obj)
        decompose_convex(
            combined_obj,
            temp_dir,
            CoacdArgs(
                threshold=threshold,
                max_convex_hull=max_convex_hulls,
                seed=seed,
            ),
        )
        generated_collision_meshes = sorted(
            temp_dir.glob("model_collision_*.obj"),
            key=_collision_index,
        )
        if not generated_collision_meshes:
            raise RuntimeError(f"CoACD did not generate collision meshes for {variant_dir.name}")

        for generated_mesh in generated_collision_meshes:
            shutil.copy2(generated_mesh, collision_dir / generated_mesh.name)

    for collision_mesh in sorted(collision_dir.glob("model_collision_*.obj"), key=_collision_index):
        mesh_attrs = {
            "file": f"collision/{collision_mesh.name}",
            "name": collision_mesh.stem,
            "scale": scale,
        }
        if refquat:
            mesh_attrs["refquat"] = refquat
        asset_root.append(ET.Element("mesh", mesh_attrs))
        object_body.append(
            ET.Element(
                "geom",
                {
                    **collision_geom_template,
                    "mesh": collision_mesh.stem,
                },
            )
        )

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    model_xml_path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    mujoco.MjModel.from_xml_path(str(model_xml_path))
    return len(list(collision_dir.glob("model_collision_*.obj")))


def _iter_variant_dirs(include_current: bool, requested_variants: list[str] | None) -> list[Path]:
    if requested_variants:
        return [PLATE_ROOT / variant_name for variant_name in requested_variants]

    variant_dirs = [path for path in sorted(PLATE_ROOT.iterdir()) if path.is_dir()]
    if include_current:
        return variant_dirs
    return [path for path in variant_dirs if path.name != "current"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-current", action="store_true", help="also regenerate the current plate asset")
    parser.add_argument("--variant", action="append", dest="variants", help="specific plate variant(s) to regenerate")
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold")
    parser.add_argument("--max-convex-hulls", type=int, default=16, help="maximum convex hulls per plate")
    parser.add_argument("--seed", type=int, default=0, help="CoACD random seed")
    args = parser.parse_args()

    variant_dirs = _iter_variant_dirs(args.include_current, args.variants)
    if not variant_dirs:
        raise ValueError("No plate variants selected")

    for variant_dir in variant_dirs:
        collision_count = _regenerate_variant(
            variant_dir,
            threshold=args.threshold,
            max_convex_hulls=args.max_convex_hulls,
            seed=args.seed,
        )
        print(f"{variant_dir.name}: regenerated {collision_count} collision hulls")


if __name__ == "__main__":
    main()
