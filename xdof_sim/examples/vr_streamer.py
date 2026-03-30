"""Lightweight VR teleop: server-side MuJoCo + Three.js WebXR streaming.

Runs MuJoCo physics on the server, exports body meshes as GLB at startup,
and streams only body transforms (position + quaternion) to a minimal
Three.js VR app running on the headset. No WASM, no heavy client-side
computation.

Usage:
    # Terminal 1: Start the VR streaming server
    uv run python -m xdof_sim.examples.vr_streamer --task blocks

    # Terminal 2: Start GELLO leader(s)
    uv run python -m xdof_sim.teleop.gello_leader --name right ...

    # Open in VR headset browser: http://<your-ip>:8012
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import threading
import time
from pathlib import Path

os.environ.pop("MUJOCO_GL", None)

import mujoco
import numpy as np
import trimesh
import trimesh.visual

from aiohttp import web

try:
    import zmq
except ImportError:
    raise SystemExit("Missing zmq. Install with: uv add pyzmq")

from xdof_sim.teleop import communication as comms


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def _get_geom_rgba(model, geom_id):
    matid = model.geom_matid[geom_id]
    if matid >= 0:
        return model.mat_rgba[matid].copy()
    rgba = model.geom_rgba[geom_id].copy()
    if rgba.sum() == 0:
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


def _create_primitive_mesh(model, geom_id):
    """Convert a MuJoCo primitive geom to trimesh with texture support."""
    from PIL import Image

    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    rgba = _get_geom_rgba(model, geom_id)
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
        plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
        mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
    else:
        return None

    # Check if this geom has a textured material
    matid = model.geom_matid[geom_id]
    has_texture = False
    if matid >= 0 and matid < model.nmat:
        texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        if texid < 0:
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
        if texid >= 0 and texid < model.ntex:
            has_texture = True
            mat_rgba = model.mat_rgba[matid]
            tex_w = model.tex_width[texid]
            tex_h = model.tex_height[texid]
            tex_nc = model.tex_nchannel[texid]
            tex_adr = model.tex_adr[texid]
            tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]
            texrepeat = model.mat_texrepeat[matid]

            if tex_nc == 3:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 3).astype(np.uint8)), mode="RGB")
            elif tex_nc == 4:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 4).astype(np.uint8)), mode="RGBA")
            elif tex_nc == 1:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w).astype(np.uint8)), mode="L")
            else:
                has_texture = False

    if has_texture:
        verts = mesh.vertices
        uv = np.zeros((len(verts), 2))
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_range = bb_max - bb_min
        bb_range[bb_range == 0] = 1
        uv[:, 0] = (verts[:, 0] - bb_min[0]) / bb_range[0] * texrepeat[0]
        uv[:, 1] = (verts[:, 1] - bb_min[1]) / bb_range[1] * texrepeat[1]
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mat_rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    else:
        vertex_colors = np.tile(rgba_uint8, (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

    return mesh


def _mujoco_mesh_to_trimesh(model, geom_id):
    """Convert a MuJoCo mesh geom to trimesh with texture support."""
    from PIL import Image

    mesh_id = model.geom_dataid[geom_id]
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        texcoords = model.mesh_texcoord[texcoord_adr:texcoord_adr + texcoord_num]
        face_texcoord_idx = model.mesh_facetexcoord[face_start:face_start + face_count]

        # Expand vertices/UVs per-face for independent texcoord indexing
        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_id]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            if texid < 0:
                texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])

            if texid >= 0 and texid < model.ntex:
                tex_w = model.tex_width[texid]
                tex_h = model.tex_height[texid]
                tex_nc = model.tex_nchannel[texid]
                tex_adr = model.tex_adr[texid]
                tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]

                image = None
                if tex_nc == 1:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w)).astype(np.uint8), mode="L")
                elif tex_nc == 3:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 3)).astype(np.uint8), mode="RGB")
                elif tex_nc == 4:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 4)).astype(np.uint8), mode="RGBA")

                if image is not None:
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba,
                        baseColorTexture=image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
                else:
                    rgba_255 = (rgba * 255).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(
                        vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
            else:
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
        else:
            rgba = _get_geom_rgba(model, geom_id)
            rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_id)
        rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1)))

    return mesh


def _patch_glb_add_material(glb_path: str):
    """Patch GLB: add default PBR material for primitives that lack one.

    trimesh exports COLOR_0 vertex colors but no material, which causes
    Three.js GLTFLoader to ignore the vertex colors. Textured meshes
    already have materials and are left untouched.
    """
    import json as _json

    with open(glb_path, "rb") as f:
        header = f.read(12)
        chunk0_len = struct.unpack("<I", f.read(4))[0]
        chunk0_type = f.read(4)
        json_bytes = f.read(chunk0_len)
        rest = f.read()  # binary chunk

    gltf = _json.loads(json_bytes)

    # Check if any primitives need a default material
    needs_patch = False
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                needs_patch = True
                break

    if not needs_patch:
        return

    # Add a default material at the end of the materials list
    if "materials" not in gltf:
        gltf["materials"] = []
    default_idx = len(gltf["materials"])
    gltf["materials"].append({
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.7,
        }
    })

    # Assign default material only to primitives that lack one
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                prim["material"] = default_idx

    # Re-encode
    new_json = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "

    with open(glb_path, "wb") as f:
        total = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)


def export_body_glbs(model, output_dir: Path):
    """Export each body's merged mesh as a GLB file. Returns body info dict."""
    from scipy.spatial.transform import Rotation

    output_dir.mkdir(parents=True, exist_ok=True)
    visible_groups = {0, 1, 2}
    body_geoms: dict[int, list[int]] = {}

    for i in range(model.ngeom):
        if int(model.geom_group[i]) not in visible_groups:
            continue
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        body_geoms.setdefault(model.geom_bodyid[i], []).append(i)

    bodies = {}
    for body_id, geom_ids in body_geoms.items():
        meshes = []
        for gid in geom_ids:
            geom_type = model.geom_type[gid]
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = _mujoco_mesh_to_trimesh(model, gid)
            else:
                mesh = _create_primitive_mesh(model, gid)
            if mesh is None:
                continue

            # Geom-local transform
            qw = model.geom_quat[gid]
            rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = model.geom_pos[gid]
            mesh.apply_transform(T)

            # Convert Z-up to Y-up for Three.js: (x,y,z) -> (x,z,-y)
            verts = np.array(mesh.vertices)
            new_verts = np.empty_like(verts)
            new_verts[:, 0] = verts[:, 0]
            new_verts[:, 1] = verts[:, 2]
            new_verts[:, 2] = -verts[:, 1]
            mesh.vertices = new_verts

            meshes.append(mesh)

        if not meshes:
            continue

        is_fixed = (model.body_weldid[body_id] == 0
                    and model.body_mocapid[model.body_rootid[body_id]] < 0)

        glb_path = output_dir / f"body_{body_id}.glb"

        # Export as Scene to preserve multiple materials (textured + colored)
        has_texture = any(isinstance(m.visual, trimesh.visual.TextureVisuals) for m in meshes)
        if has_texture or len(meshes) > 1:
            scene = trimesh.Scene(meshes)
            scene.export(str(glb_path), file_type="glb")
        else:
            meshes[0].export(str(glb_path), file_type="glb")

        # Patch GLB: add material for vertex-colored meshes (trimesh omits it)
        _patch_glb_add_material(str(glb_path))

        bodies[body_id] = {
            "file": f"body_{body_id}.glb",
            "is_fixed": is_fixed,
        }

    return bodies


# ---------------------------------------------------------------------------
# HTML app
# ---------------------------------------------------------------------------

VR_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XDoF VR Teleop</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; }
  #info { position: absolute; top: 10px; width: 100%; text-align: center;
          color: #fff; font-family: monospace; font-size: 14px; z-index: 1; }
  #vr-btn { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            padding: 12px 24px; font-size: 18px; cursor: pointer; z-index: 1;
            background: #4CAF50; color: white; border: none; border-radius: 8px; }
</style>
</head>
<body>
<div id="info">Connecting...</div>
<button id="vr-btn" style="display:none">Enter VR</button>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.35);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(2, 4, 3);
scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
dir2.position.set(-2, 3, -1);
scene.add(dir2);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(10, 10);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3e, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.001;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(4, 20, 0x444466, 0x333355);
scene.add(grid);

// Camera
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.5, 1.5, 1.5);
camera.lookAt(0, 0.8, 0);

// VR rig — positions the user in the scene when entering VR
const vrPos = window.__VR_POS__ || [0, 1.5, 1.0];
const vrTarget = window.__VR_TARGET__ || [0, 0.8, 0];
const vrRig = new THREE.Group();
vrRig.position.set(vrPos[0], vrPos[1], vrPos[2]);
// Rotate rig to face the target (only yaw — lookAt then zero out pitch)
const targetVec = new THREE.Vector3(vrTarget[0], vrPos[1], vrTarget[2]);
vrRig.lookAt(targetVec);
scene.add(vrRig);
vrRig.add(camera);

// Renderer with WebXR
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.xr.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

// VR button
const vrBtn = VRButton.createButton(renderer);
document.body.appendChild(vrBtn);

// Orbit controls for non-VR (use vr target as orbit target too)
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(vrTarget[0], vrTarget[1], vrTarget[2]);
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Scene rotation: rotate so table faces VR user
const sceneRoot = new THREE.Group();
sceneRoot.rotation.y = Math.PI / 2;
scene.add(sceneRoot);

// Load body info and meshes via GLTFLoader
const info = document.getElementById('info');
const loader = new GLTFLoader();
const bodyMeshes = {};
let bodyInfo = null;

async function loadScene() {
  info.textContent = 'Loading scene...';

  const resp = await fetch('/api/bodies');
  bodyInfo = await resp.json();
  // Cache bust: append timestamp to mesh URLs
  const cacheBust = '?t=' + Date.now();

  const total = Object.keys(bodyInfo).length;
  let loaded = 0;

  for (const [bid, bdata] of Object.entries(bodyInfo)) {
    try {
      const gltf = await new Promise((resolve, reject) => {
        loader.load('/meshes/' + bdata.file + cacheBust, resolve, undefined, reject);
      });

      const obj = gltf.scene;
      obj.traverse((child) => {
        if (child.isMesh) {
          // Recompute normals — GLB normals are stale after Z-up to Y-up vertex swap
          child.geometry.computeVertexNormals();

          if (child.material.map) {
            // Has a texture map (from GLB) — keep the material as-is
            child.material.side = THREE.DoubleSide;
          } else if (child.geometry.attributes.color) {
            // Has vertex colors but no texture — replace material entirely
            // (setting vertexColors on GLTFLoader material doesn't recompile shader)
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
              roughness: 0.5,
              metalness: 0.15,
            });
          } else {
            child.material.side = THREE.DoubleSide;
          }
        }
      });

      sceneRoot.add(obj);
      bodyMeshes[bid] = obj;
      loaded++;
    } catch (e) {
      console.error(`Failed to load body ${bid}:`, e);
    }
    info.textContent = `Loading meshes: ${loaded}/${total}`;
  }

  info.textContent = `Loaded ${loaded}/${total} bodies. Connecting...`;
  connectWS();
}

// WebSocket for transform streaming
let ws = null;
let frameCount = 0;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    info.textContent = 'Connected! Waiting for GELLO...';
  };

  ws.onmessage = (event) => {
    // Binary format: [n_bodies (uint16), body_id (uint16), px, py, pz, qx, qy, qz, qw (float32) × n]
    const buf = new Float32Array(event.data);
    const n = buf.length / 8;  // 8 floats per body: id(as float), px, py, pz, qx, qy, qz, qw

    for (let i = 0; i < n; i++) {
      const offset = i * 8;
      const bid = Math.round(buf[offset]).toString();
      const px = buf[offset + 1];
      const py = buf[offset + 2];
      const pz = buf[offset + 3];
      const qx = buf[offset + 4];
      const qy = buf[offset + 5];
      const qz = buf[offset + 6];
      const qw = buf[offset + 7];

      const obj = bodyMeshes[bid];
      if (obj) {
        obj.position.set(px, py, pz);
        obj.quaternion.set(qx, qy, qz, qw);
      }
    }

    frameCount++;
    if (frameCount === 1 && !window.__DEBUG__) {
      info.textContent = '';
    }
    if (window.__DEBUG__ && frameCount % 30 === 0) {
      info.textContent = `Frame: ${frameCount}`;
    }
  };

  ws.onclose = () => {
    info.textContent = 'Disconnected. Reconnecting...';
    setTimeout(connectWS, 1000);
  };
}

// Render loop
renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});

loadScene();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lightweight VR streaming teleop")
    parser.add_argument("--task", type=str, default="blocks",
                        choices=["bottles", "marker", "ball_sorting", "empty",
                                 "dishrack", "chess", "chess2", "blocks",
                                 "mug_tree", "mug_flip"])
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--left-leader", type=str, default="left")
    parser.add_argument("--right-leader", type=str, default="right")
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument("--stream-rate", type=float, default=60.0,
                        help="Rate to stream transforms to VR client")
    parser.add_argument("--debug", action="store_true",
                        help="Show frame counter overlay in VR client")
    parser.add_argument("--vr-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="VR user floor position in Y-up coords (default: 0 0 -1)")
    parser.add_argument("--vr-target", type=float, nargs=3, default=[0.0, 0.75, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Point the VR user looks at in Y-up coords (default: 0 0.75 0)")
    args = parser.parse_args()

    import xdof_sim
    env = xdof_sim.make_env(scene="hybrid", task=args.task, render_cameras=False)
    model = env.model
    data = env.data

    print(f"Task: {args.task}")
    print(f"Environment: nbody={model.nbody}, ngeom={model.ngeom}")

    # Export meshes (clean dir to avoid stale files from other tasks)
    import shutil
    mesh_dir = Path("/tmp/xdof_vr_meshes")
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    body_info = export_body_glbs(model, mesh_dir)
    fixed_ids = [bid for bid, info in body_info.items() if info["is_fixed"]]
    dynamic_ids = [bid for bid, info in body_info.items() if not info["is_fixed"]]
    all_ids = list(body_info.keys())
    print(f"Exported {len(body_info)} body meshes ({len(fixed_ids)} fixed, {len(dynamic_ids)} dynamic)")

    # Reset
    obs, _ = env.reset()
    state = obs["state"].copy()

    # ZMQ
    zmq_context = zmq.Context()
    left_sub = comms.create_subscriber(zmq_context, f"{args.left_leader}_actions", conflate=1)
    right_sub = comms.create_subscriber(zmq_context, f"{args.right_leader}_actions", conflate=1)
    left_sub.setsockopt(zmq.RCVTIMEO, 0)
    right_sub.setsockopt(zmq.RCVTIMEO, 0)

    def poll_leader(sub):
        try:
            msg, _ = comms.subscribe(sub)
            return msg
        except (zmq.Again, Exception):
            return None

    # Z-up to Y-up conversion for positions and quaternions
    from scipy.spatial.transform import Rotation

    R_conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    def build_frame_buffer():
        """Pack all body transforms into a flat float32 buffer (Y-up)."""
        buf = np.zeros(len(all_ids) * 8, dtype=np.float32)
        for i, bid in enumerate(all_ids):
            xpos = data.xpos[bid]
            xmat = data.xmat[bid].reshape(3, 3)
            pos_yup = R_conv @ xpos
            det = np.linalg.det(xmat)
            if abs(det) > 1e-6:
                mat_yup = R_conv @ xmat @ R_conv.T
                q = Rotation.from_matrix(mat_yup).as_quat()  # xyzw
            else:
                q = np.array([0, 0, 0, 1.0])
            offset = i * 8
            buf[offset] = float(bid)
            buf[offset+1:offset+4] = pos_yup
            buf[offset+4:offset+8] = q
        return buf.tobytes()

    # Shared state for physics thread
    GRIP_SCALE = 0.0475
    lock = threading.Lock()
    connected = [False]
    step_count = [0]
    frame_bytes = [build_frame_buffer()]

    def physics_loop():
        nonlocal state
        dt = 1.0 / args.control_rate

        while True:
            t0 = time.time()

            left_pos = poll_leader(left_sub)
            right_pos = poll_leader(right_sub)

            if left_pos is not None or right_pos is not None:
                if not connected[0]:
                    connected[0] = True
                    print("GELLO connected!")

                action = state.copy()
                if left_pos is not None and len(left_pos) >= 7:
                    action[:7] = left_pos[:7]
                if right_pos is not None and len(right_pos) >= 7:
                    action[7:14] = right_pos[:7]

                with lock:
                    env._step_single(action)
                    state = env.get_obs()["state"].copy()
                    frame_bytes[0] = build_frame_buffer()
                step_count[0] += 1

            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # Start physics in background thread
    physics_thread = threading.Thread(target=physics_loop, daemon=True)
    physics_thread.start()

    # --- aiohttp web server ---
    app_web = web.Application()
    ws_clients: list[web.WebSocketResponse] = []

    async def handle_index(request):
        config_script = '<script>'
        if args.debug:
            config_script += 'window.__DEBUG__ = true;'
        config_script += f'window.__VR_POS__ = {args.vr_pos};'
        config_script += f'window.__VR_TARGET__ = {args.vr_target};'
        config_script += '</script>'
        html = VR_HTML.replace('</head>', config_script + '</head>')
        return web.Response(text=html, content_type="text/html")

    async def handle_bodies(request):
        return web.json_response({
            str(k): {"file": v["file"], "is_fixed": bool(v["is_fixed"])}
            for k, v in body_info.items()
        })

    async def handle_mesh(request):
        filename = request.match_info["filename"]
        path = mesh_dir / filename
        if not path.exists():
            return web.Response(status=404)
        return web.FileResponse(path, headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Content-Type": "model/gltf-binary",
        })

    async def handle_ws(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.append(ws)
        print(f"VR client connected ({len(ws_clients)} total)")

        try:
            async for msg in ws:
                pass  # We only send, never receive
        finally:
            ws_clients.remove(ws)
            print(f"VR client disconnected ({len(ws_clients)} remaining)")

        return ws

    async def stream_loop():
        """Broadcast transforms to all connected WebSocket clients."""
        dt = 1.0 / args.stream_rate
        while True:
            if ws_clients:
                with lock:
                    data_bytes = frame_bytes[0]
                for ws in list(ws_clients):
                    try:
                        await ws.send_bytes(data_bytes)
                    except Exception:
                        pass
            await asyncio.sleep(dt)

    async def on_startup(app):
        asyncio.create_task(stream_loop())

    app_web.router.add_get("/", handle_index)
    app_web.router.add_get("/api/bodies", handle_bodies)
    app_web.router.add_get("/meshes/{filename}", handle_mesh)
    app_web.router.add_get("/ws", handle_ws)
    app_web.on_startup.append(on_startup)

    print(f"\nVR Streaming Server: http://0.0.0.0:{args.port}")
    print(f"VR position: {args.vr_pos}, looking at: {args.vr_target}")
    print(f"Open in VR headset browser to connect")
    print(f"Waiting for GELLO leaders on '{args.left_leader}' and '{args.right_leader}'...")

    web.run_app(app_web, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
