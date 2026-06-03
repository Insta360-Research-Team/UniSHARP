from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unisharp.cli.unified_trainer import UnifiedTrainer  # noqa: E402
from unisharp.models.unisharp_feature import UnisharpFeatureConfig, UnisharpFeatureModel  # noqa: E402
from unisharp.utils.camera_utils import transform_gaussians_to_world  # noqa: E402
from unisharp.utils.color_space import linearRGB2sRGB  # noqa: E402
from unisharp.utils.fisheye_geer import render_gaussians_fisheye624  # noqa: E402
from unisharp.utils.gaussians import save_ply  # noqa: E402
from unisharp.utils.gsplat import GSplatRenderer  # noqa: E402
from unisharp.utils.camera_projection import build_extrinsics_w2c  # noqa: E402
from unisharp.utils.rayfit_camera import fit_fisheye624_params_from_rays, fit_pinhole_intrinsics_from_rays  # noqa: E402


LOGGER = logging.getLogger("infer_unisharp")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
CameraKind = Literal["perspective", "fisheye", "panorama"]
FACE_NAMES = ["up", "back", "left", "front", "right", "down"]


def _configure_torchhub_cache() -> None:
    torchhub_dir = REPO_ROOT / "checkpoints" / "torchhub"
    torchhub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torchhub_dir)
    torch.hub.set_dir(str(torchhub_dir))


def _feature_config_from_checkpoint(checkpoint_path: Path, ckpt: dict[str, Any]) -> UnisharpFeatureConfig:
    cfg = UnisharpFeatureConfig()
    merged: dict[str, Any] = {}
    cfg_payload = ckpt.get("config", {})
    if isinstance(cfg_payload, dict):
        merged.update(cfg_payload)
    for key in cfg.__dict__.keys():
        if key in ckpt:
            merged[key] = ckpt[key]
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        try:
            sidecar = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            sidecar = None
        if isinstance(sidecar, dict):
            merged.update({k: v for k, v in sidecar.items() if k in cfg.__dict__})
    for key in cfg.__dict__.keys():
        if key in merged:
            setattr(cfg, key, merged[key])
    return cfg


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[UnisharpFeatureModel, int]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(ckpt)} from {checkpoint_path}")
    cfg = _feature_config_from_checkpoint(checkpoint_path, ckpt)
    model = UnisharpFeatureModel(cfg).to(device)
    missing, unexpected = model.load_from_checkpoint(str(checkpoint_path), strict=False)
    if missing or unexpected:
        LOGGER.warning("Loaded checkpoint with missing=%s unexpected=%s", missing[:20], unexpected[:20])
    model.eval()
    return model, int(ckpt.get("step", 0))


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.image is not None:
        paths.append(Path(args.image))
    if args.image_list is not None:
        for raw in Path(args.image_list).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    if args.image_dir is not None:
        root = Path(args.image_dir)
        paths.extend(sorted(p for p in root.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES))
    if not paths:
        raise ValueError("Provide --image, --image-list, or --image-dir.")
    return paths[: int(args.max_images)] if int(args.max_images) > 0 else paths


def _load_rgb_u8(image_path: Path, max_long_edge: int) -> torch.Tensor:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if int(max_long_edge) > 0:
        w, h = image.size
        scale = min(1.0, float(max_long_edge) / float(max(h, w)))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                resample=Image.BILINEAR,
            )
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_u8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    if img_chw.dtype == torch.uint8:
        return img_chw.permute(1, 2, 0).detach().cpu().numpy()
    x = img_chw.detach().to(torch.float32).clamp(0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _crop_border_u8(frame: np.ndarray, fraction: float) -> np.ndarray:
    if float(fraction) <= 0.0:
        return frame
    if frame.ndim < 2:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    crop_y = int(round(float(h) * float(fraction)))
    crop_x = int(round(float(w) * float(fraction)))
    if crop_y <= 0 and crop_x <= 0:
        return frame
    if crop_y * 2 >= h or crop_x * 2 >= w:
        return frame
    return frame[crop_y : h - crop_y, crop_x : w - crop_x].copy()


def _save_gif(frames: list[np.ndarray], out_file: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
    )


def _slug_from_path(image_path: Path) -> str:
    raw = f"{image_path.parent.name}_{image_path.stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _normalize_rays(rays: torch.Tensor) -> torch.Tensor:
    rays_f = rays.detach().to(torch.float32)
    return rays_f / torch.linalg.vector_norm(rays_f, dim=1, keepdim=True).clamp(min=1e-6)


def _angular_span_deg(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(np.degrees(np.nanpercentile(a, 99.0) - np.nanpercentile(a, 1.0)))


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1.0, 1.0))))


def _ray_fov_stats(rays_b3hw: torch.Tensor) -> dict[str, float]:
    rays = _normalize_rays(rays_b3hw)[0].detach().cpu().numpy()
    _, h, w = rays.shape
    rows = [max(0, min(h - 1, int(round(h * q)))) for q in (0.25, 0.5, 0.75)]
    cols = [max(0, min(w - 1, int(round(w * q)))) for q in (0.25, 0.5, 0.75)]
    h_spans = []
    for row in rows:
        lon = np.unwrap(np.arctan2(rays[0, row], rays[2, row]))
        h_spans.append(_angular_span_deg(lon))
    v_spans = []
    for col in cols:
        x = rays[0, :, col]
        y = rays[1, :, col]
        z = rays[2, :, col]
        lat = np.arctan2(y, np.sqrt(x * x + z * z))
        v_spans.append(_angular_span_deg(lat))
    corners = [rays[:, 0, 0], rays[:, 0, w - 1], rays[:, h - 1, 0], rays[:, h - 1, w - 1]]
    diag = max(_angle_between_deg(corners[i], corners[j]) for i in range(4) for j in range(i + 1, 4))
    return {
        "horizontal_fov_deg": float(np.median(h_spans)),
        "vertical_fov_deg": float(np.median(v_spans)),
        "diagonal_fov_deg": float(diag),
        "aspect": float(w) / float(max(h, 1)),
    }


def _classify_camera(stats: dict[str, float], args: argparse.Namespace) -> CameraKind:
    forced = str(args.camera).strip().lower()
    if forced != "auto":
        return {"pinhole": "perspective", "erp": "panorama"}.get(forced, forced)  # type: ignore[return-value]
    aspect = float(stats["aspect"])
    h_fov = float(stats["horizontal_fov_deg"])
    v_fov = float(stats["vertical_fov_deg"])
    diag_fov = float(stats["diagonal_fov_deg"])
    if (
        float(args.panorama_aspect_min) <= aspect <= float(args.panorama_aspect_max)
        and h_fov >= float(args.panorama_hfov_threshold_deg)
        and v_fov >= float(args.panorama_vfov_threshold_deg)
    ):
        return "panorama"
    fishlike_aspect = aspect <= float(args.fisheye_max_aspect)
    fishlike_fov = (
        max(h_fov, v_fov) >= float(args.fisheye_fov_threshold_deg)
        or (diag_fov >= float(args.fisheye_diag_threshold_deg) and v_fov >= float(args.fisheye_vfov_min_deg))
    )
    if fishlike_aspect and fishlike_fov:
        return "fisheye"
    return "perspective"


def _empty_ray_stats() -> dict[str, float]:
    return {
        "horizontal_fov_deg": float("nan"),
        "vertical_fov_deg": float("nan"),
        "diagonal_fov_deg": float("nan"),
        "aspect": float("nan"),
    }


def _pinhole_intrinsics_from_values(values: list[float] | None, *, device: torch.device) -> torch.Tensor | None:
    if values is None:
        return None
    vals = [float(v) for v in values]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        k = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
    elif len(vals) == 9:
        k = torch.tensor(vals, dtype=torch.float32, device=device).reshape(3, 3)
    else:
        raise ValueError("--camera-intrinsics expects 4 values (fx fy cx cy) or 9 row-major K values.")
    return k.unsqueeze(0)


def _fisheye624_params_from_values(values: list[float] | None, *, device: torch.device) -> torch.Tensor | None:
    if values is None:
        return None
    vals = [float(v) for v in values]
    if len(vals) == 8:
        vals = vals + [0.0] * 8
    if len(vals) != 16:
        raise ValueError("--camera-params expects 8 or 16 Fisheye624 values.")
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(1, 16)


def _load_camera_json(path: Path | None) -> Any:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--camera-json must point to a JSON object.")
    return payload


def _camera_json_for_image(payload: Any, image_path: Path) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    images = payload.get("images", None)
    if isinstance(images, dict):
        keys = [
            str(image_path),
            image_path.as_posix(),
            image_path.name,
            image_path.stem,
        ]
        for key in keys:
            value = images.get(key, None)
            if isinstance(value, dict):
                base = payload.get("default", {})
                merged = dict(base) if isinstance(base, dict) else {}
                merged.update(value)
                return merged
    if isinstance(payload.get("default", None), dict):
        return dict(payload["default"])
    return dict(payload)


def _values_from_camera_json(entry: dict[str, Any] | None, *names: str) -> list[float] | None:
    if not isinstance(entry, dict):
        return None
    for name in names:
        value = entry.get(name, None)
        if value is None:
            continue
        if isinstance(value, dict):
            if all(k in value for k in ("fx", "fy", "cx", "cy")):
                return [float(value["fx"]), float(value["fy"]), float(value["cx"]), float(value["cy"])]
            if "K" in value:
                value = value["K"]
            else:
                continue
        if isinstance(value, (list, tuple)):
            if len(value) == 3 and all(isinstance(row, (list, tuple)) for row in value):
                flat = [float(x) for row in value for x in row]
            else:
                flat = [float(x) for x in value]
            return flat
    return None


def _camera_name_from_json(entry: dict[str, Any] | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("camera", entry.get("camera_model", entry.get("type", None)))
    return str(value).strip().lower() if value is not None and str(value).strip() else None


@torch.no_grad()
def _predict_unik3d_rays(
    model: UnisharpFeatureModel,
    image_u8: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    model.feature_extractor.forward(
        rgb_u8=image_u8,
        target_h=int(image_h),
        target_w=int(image_w),
        use_predicted_rays=True,
    )
    output = model.feature_extractor._unisharp_last_unik3d_output
    if not isinstance(output, dict) or not torch.is_tensor(output.get("rays", None)):
        raise RuntimeError("UniK3D did not return predicted rays for camera classification.")
    return output["rays"]


def _build_forward_poses(num_views: int, distance_m: float, device: torch.device) -> list[torch.Tensor]:
    poses = []
    r_c2w = torch.eye(3, dtype=torch.float32, device=device)
    views = max(1, int(num_views))
    for idx in range(views):
        alpha = float(idx + 1) / float(views)
        eye = torch.tensor([0.0, 0.0, float(distance_m) * alpha], dtype=torch.float32, device=device)
        poses.append(build_extrinsics_w2c(r_c2w, eye, "c2w"))
    return poses


def _build_rotate_poses(num_views: int, radius_m: float, device: torch.device) -> list[torch.Tensor]:
    poses = []
    src_r_c2w = torch.eye(3, dtype=torch.float32, device=device)
    views = max(1, int(num_views))
    for idx in range(views):
        theta = -2.0 * math.pi * float(idx) / float(views)
        eye = torch.tensor(
            [
                float(radius_m) * math.sin(theta),
                float(radius_m) * math.cos(theta),
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        poses.append(build_extrinsics_w2c(src_r_c2w, eye, "c2w"))
    return poses


def _render_pinhole_frame(
    renderer: GSplatRenderer,
    gaussians: Any,
    *,
    extr_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    out = renderer(
        gaussians,
        extrinsics=extr_w2c[None],
        intrinsics=intrinsics[None],
        image_width=int(image_w),
        image_height=int(image_h),
    )
    alpha = out.alpha.detach().to(torch.float32).clamp(0.0, 1.0)
    rgb = linearRGB2sRGB((out.color / alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return _to_u8_hwc(rgb[0])


def _render_fisheye_frame(
    gaussians: Any,
    *,
    extr_w2c: torch.Tensor,
    camera_params: torch.Tensor,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    out = render_gaussians_fisheye624(
        gaussians,
        extrinsics_w2c=extr_w2c[None],
        camera_params=camera_params,
        image_h=int(image_h),
        image_w=int(image_w),
        valid_mask=None,
    )
    alpha = out["alpha"].detach().to(torch.float32).clamp(0.0, 1.0)
    rgb = linearRGB2sRGB((out["color"] / alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return _to_u8_hwc(rgb[0])


def _render_panorama_frame_and_faces(
    trainer: UnifiedTrainer,
    gaussians: Any,
    *,
    extr_w2c: torch.Tensor,
    equ_h: int,
    equ_w: int,
    face_w: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cube_color, _, cube_alpha = trainer._render_cubemap(gaussians, extr_w2c, face_w=int(face_w))
    erp_color = trainer._cube_to_erp(cube_color, equ_h=int(equ_h), equ_w=int(equ_w), face_w=int(face_w))
    erp_alpha = trainer._cube_to_erp(cube_alpha, equ_h=int(equ_h), equ_w=int(equ_w), face_w=int(face_w))
    erp = linearRGB2sRGB((erp_color / erp_alpha.clamp(min=1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    face_views: dict[str, np.ndarray] = {}
    for face_idx, face_name in enumerate(FACE_NAMES):
        face = linearRGB2sRGB(
            (cube_color[face_idx : face_idx + 1] / cube_alpha[face_idx : face_idx + 1].clamp(min=1e-4)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        face_views[face_name] = _to_u8_hwc(face[0])
    return _to_u8_hwc(erp[0]), face_views


@torch.no_grad()
def _run_model_pinhole(
    model: UnisharpFeatureModel,
    image: torch.Tensor,
    image_u8: torch.Tensor,
    *,
    intrinsics: torch.Tensor,
    distance_init_cap_m: float,
) -> dict[str, Any]:
    return model(
        image=image,
        image_u8=image_u8,
        camera_intrinsics=intrinsics,
        camera_params=None,
        camera_model="pinhole",
        depth_gt=None,
        distance_init_cap_m=(float(distance_init_cap_m) if float(distance_init_cap_m) > 0.0 else None),
        return_aux=True,
    )


@torch.no_grad()
def _run_model_fisheye(
    model: UnisharpFeatureModel,
    image: torch.Tensor,
    image_u8: torch.Tensor,
    *,
    camera_params: torch.Tensor,
    distance_init_cap_m: float,
) -> dict[str, Any]:
    return model(
        image=image,
        image_u8=image_u8,
        camera_intrinsics=None,
        camera_params=camera_params,
        camera_model="fisheye624",
        depth_gt=None,
        distance_init_cap_m=(float(distance_init_cap_m) if float(distance_init_cap_m) > 0.0 else None),
        return_aux=True,
    )


@torch.no_grad()
def _run_model_panorama(
    model: UnisharpFeatureModel,
    image: torch.Tensor,
    image_u8: torch.Tensor,
    *,
    distance_init_cap_m: float,
) -> dict[str, Any]:
    return model(
        image=image,
        image_u8=image_u8,
        camera_intrinsics=None,
        camera_params=None,
        camera_model="spherical",
        depth_gt=None,
        distance_init_cap_m=(float(distance_init_cap_m) if float(distance_init_cap_m) > 0.0 else None),
        return_aux=True,
    )


def _save_ply_if_requested(gaussians: Any, path: Path, f_px: float, image_h: int, image_w: int, enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, f_px=float(f_px), image_shape=(int(image_h), int(image_w)), path=path)


@torch.no_grad()
def _process_one(
    *,
    model: UnisharpFeatureModel,
    renderer: GSplatRenderer,
    train_renderer: UnifiedTrainer,
    image_path: Path,
    out_root: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    rgb_u8 = _load_rgb_u8(image_path, max_long_edge=int(args.max_long_edge))
    _, h, w = rgb_u8.shape
    if h < 4 or w < 4:
        raise ValueError(f"Invalid image size for {image_path}: {tuple(rgb_u8.shape)}")

    device = next(model.parameters()).device
    image_u8 = rgb_u8.unsqueeze(0).to(device=device)
    image = image_u8.to(torch.float32) / 255.0

    camera_json_entry = _camera_json_for_image(getattr(args, "_camera_json_data", None), image_path)
    json_camera_name = _camera_name_from_json(camera_json_entry)
    json_intrinsics = _values_from_camera_json(camera_json_entry, "intrinsics", "camera_intrinsics", "K")
    json_camera_params = _values_from_camera_json(camera_json_entry, "camera_params", "fisheye624_params", "params")
    explicit_intrinsics = _pinhole_intrinsics_from_values(json_intrinsics or args.camera_intrinsics, device=device)
    explicit_camera_params = _fisheye624_params_from_values(json_camera_params or args.camera_params, device=device)
    if explicit_intrinsics is not None and explicit_camera_params is not None:
        raise ValueError("Use only one of --camera-intrinsics or --camera-params.")

    rays: torch.Tensor | None
    render_intrinsics: torch.Tensor | None = None
    render_camera_params: torch.Tensor | None = None
    if explicit_intrinsics is not None:
        camera_kind: CameraKind = "panorama" if json_camera_name in {"panorama", "erp", "spherical"} else "perspective"
        render_intrinsics = explicit_intrinsics
        if camera_kind == "panorama":
            out = _run_model_panorama(model, image, image_u8, distance_init_cap_m=float(args.distance_init_cap_m))
        else:
            out = _run_model_pinhole(
                model,
                image,
                image_u8,
                intrinsics=explicit_intrinsics,
                distance_init_cap_m=float(args.distance_init_cap_m),
            )
        rays = out.get("geometry_rays", out.get("unik3d_gt_rays", out.get("unik3d_rays", None)))
        stats = _ray_fov_stats(rays) if torch.is_tensor(rays) else _empty_ray_stats()
    elif explicit_camera_params is not None:
        camera_kind = "fisheye"
        render_camera_params = explicit_camera_params
        out = _run_model_fisheye(
            model,
            image,
            image_u8,
            camera_params=explicit_camera_params,
            distance_init_cap_m=float(args.distance_init_cap_m),
        )
        rays = out.get("geometry_rays", out.get("unik3d_gt_rays", out.get("unik3d_rays", None)))
        stats = _ray_fov_stats(rays) if torch.is_tensor(rays) else _empty_ray_stats()
    else:
        rays = _predict_unik3d_rays(model, image_u8, image_h=h, image_w=w)
        stats = _ray_fov_stats(rays)
        if json_camera_name in {"panorama", "erp", "spherical"}:
            camera_kind = "panorama"
        elif json_camera_name in {"fisheye", "fisheye624", "opencv_fisheye"}:
            camera_kind = "fisheye"
        elif json_camera_name in {"perspective", "pinhole"}:
            camera_kind = "perspective"
        else:
            camera_kind = _classify_camera(stats, args)
        if camera_kind == "panorama":
            out = _run_model_panorama(model, image, image_u8, distance_init_cap_m=float(args.distance_init_cap_m))
        elif camera_kind == "fisheye":
            render_camera_params = fit_fisheye624_params_from_rays(rays).detach().to(device=device, dtype=torch.float32)
            out = _run_model_fisheye(
                model,
                image,
                image_u8,
                camera_params=render_camera_params,
                distance_init_cap_m=float(args.distance_init_cap_m),
            )
        else:
            render_intrinsics = fit_pinhole_intrinsics_from_rays(rays).detach().to(device=device, dtype=torch.float32)
            out = _run_model_pinhole(
                model,
                image,
                image_u8,
                intrinsics=render_intrinsics,
                distance_init_cap_m=float(args.distance_init_cap_m),
            )

    LOGGER.info(
        "%s -> %s | hfov=%.1f vfov=%.1f diag=%.1f aspect=%.3f",
        image_path,
        camera_kind,
        stats["horizontal_fov_deg"],
        stats["vertical_fov_deg"],
        stats["diagonal_fov_deg"],
        stats["aspect"],
    )

    src_w2c = torch.eye(4, dtype=torch.float32, device=device)
    gaussians_world = transform_gaussians_to_world(out["gaussians"], src_w2c)
    forward_poses = _build_forward_poses(
        num_views=int(args.forward_views),
        distance_m=float(args.forward_distance_m),
        device=device,
    )
    rotate_poses = _build_rotate_poses(
        num_views=int(args.rotate_views),
        radius_m=float(args.rotate_radius_m),
        device=device,
    )

    sample_dir = out_root / _slug_from_path(image_path)
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_crop_border_fraction = 0.0 if camera_kind == "panorama" else 0.05
    Image.fromarray(_crop_border_u8(_to_u8_hwc(rgb_u8), output_crop_border_fraction)).save(sample_dir / "input.png")

    forward_frames: list[np.ndarray] = []
    rotate_frames: list[np.ndarray] = []

    if camera_kind == "panorama":
        face_w = int(args.face_w) if int(args.face_w) > 0 else max(16, int(min(h, w // 4)))
        forward_dir = sample_dir / "forward_erp"
        rotate_dir = sample_dir / "rotate_erp"
        rotate_faces_dir = sample_dir / "rotate_cubemap_faces"
        forward_dir.mkdir(parents=True, exist_ok=True)
        rotate_dir.mkdir(parents=True, exist_ok=True)
        for face_name in FACE_NAMES:
            (rotate_faces_dir / face_name).mkdir(parents=True, exist_ok=True)
        for pose in forward_poses:
            erp_u8, _ = _render_panorama_frame_and_faces(
                train_renderer,
                gaussians_world,
                extr_w2c=pose,
                equ_h=h,
                equ_w=w,
                face_w=face_w,
            )
            forward_dir.joinpath(f"forward_{len(forward_frames):02d}.png").parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(erp_u8).save(forward_dir / f"forward_{len(forward_frames):02d}.png")
            forward_frames.append(erp_u8)
        for pose in rotate_poses:
            erp_u8, face_views = _render_panorama_frame_and_faces(
                train_renderer,
                gaussians_world,
                extr_w2c=pose,
                equ_h=h,
                equ_w=w,
                face_w=face_w,
            )
            frame_idx = len(rotate_frames)
            Image.fromarray(erp_u8).save(rotate_dir / f"rotate_{frame_idx:02d}.png")
            for face_name, face_u8 in face_views.items():
                Image.fromarray(face_u8).save(rotate_faces_dir / face_name / f"rotate_{frame_idx:02d}_{face_name}.png")
            rotate_frames.append(erp_u8)
        f_px = float(w) / (2.0 * math.pi)
    elif camera_kind == "fisheye":
        if render_camera_params is None:
            if not torch.is_tensor(rays):
                raise RuntimeError("Fisheye ray fitting requires model rays.")
            render_camera_params = fit_fisheye624_params_from_rays(rays)
        params = render_camera_params
        params = params.detach().to(device=device, dtype=torch.float32)
        for pose in forward_poses:
            forward_frames.append(_render_fisheye_frame(gaussians_world, extr_w2c=pose, camera_params=params, image_h=h, image_w=w))
        for pose in rotate_poses:
            rotate_frames.append(_render_fisheye_frame(gaussians_world, extr_w2c=pose, camera_params=params, image_h=h, image_w=w))
        f_px = float(0.5 * (float(params[0, 0].detach().cpu()) + float(params[0, 1].detach().cpu())))
    else:
        if render_intrinsics is None:
            if not torch.is_tensor(rays):
                raise RuntimeError("Pinhole ray fitting requires model rays.")
            render_intrinsics = fit_pinhole_intrinsics_from_rays(rays)
        intrinsics = render_intrinsics
        k3 = intrinsics.detach().to(device=device, dtype=torch.float32)[0]
        for pose in forward_poses:
            forward_frames.append(_render_pinhole_frame(renderer, gaussians_world, extr_w2c=pose, intrinsics=k3, image_h=h, image_w=w))
        for pose in rotate_poses:
            rotate_frames.append(_render_pinhole_frame(renderer, gaussians_world, extr_w2c=pose, intrinsics=k3, image_h=h, image_w=w))
        f_px = float(0.5 * (float(k3[0, 0].detach().cpu()) + float(k3[1, 1].detach().cpu())))

    if output_crop_border_fraction > 0.0:
        forward_frames = [_crop_border_u8(frame, output_crop_border_fraction) for frame in forward_frames]
        rotate_frames = [_crop_border_u8(frame, output_crop_border_fraction) for frame in rotate_frames]

    _save_gif(forward_frames, sample_dir / "forward_0p2m.gif", duration_ms=int(args.gif_duration_ms))
    _save_gif(rotate_frames, sample_dir / "rotate_0p1m.gif", duration_ms=int(args.gif_duration_ms))
    _save_ply_if_requested(gaussians_world, sample_dir / "gaussians.ply", f_px=f_px, image_h=h, image_w=w, enabled=bool(args.save_ply))

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(step),
        "image": str(image_path),
        "camera_kind": camera_kind,
        "ray_stats": stats,
        "camera_json": str(args.camera_json) if args.camera_json is not None else None,
        "camera_json_entry": camera_json_entry,
        "explicit_camera_intrinsics": args.camera_intrinsics,
        "explicit_camera_params": args.camera_params,
        "forward_distance_m": float(args.forward_distance_m),
        "rotate_radius_m": float(args.rotate_radius_m),
        "rotate_path": "clockwise_camera_xy_orbit_fixed_source_orientation",
        "panorama_renderer": "unisharp.cli.unified_trainer.UnifiedTrainer._render_cubemap/_cube_to_erp",
        "low_pass_filter_eps": float(args.low_pass_filter_eps),
        "output_crop_border_fraction": float(output_crop_border_fraction),
        "height": int(h),
        "width": int(w),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved outputs -> %s", sample_dir)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UniSharp single-image inference with automatic camera-type detection.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--image-list", type=Path, default=None)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "inference")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--max-long-edge", type=int, default=768)
    p.add_argument("--forward-views", type=int, default=10)
    p.add_argument("--forward-distance-m", type=float, default=0.2)
    p.add_argument("--rotate-views", type=int, default=10)
    p.add_argument("--rotate-radius-m", type=float, default=0.1)
    p.add_argument("--gif-duration-ms", type=int, default=300)
    p.add_argument("--face-w", type=int, default=0, help="Panorama cubemap face width. 0 uses min(H, W/4).")
    p.add_argument("--distance-init-cap-m", type=float, default=0.0)
    p.add_argument("--save-ply", action="store_true")
    p.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="JSON file with calibrated camera parameters. Supports a global object or an images mapping keyed by path/name/stem.",
    )
    p.add_argument(
        "--camera-intrinsics",
        type=float,
        nargs="+",
        default=None,
        help="Explicit pinhole intrinsics. Pass fx fy cx cy or 9 row-major K values. If omitted, intrinsics are fitted from rays.",
    )
    p.add_argument(
        "--camera-params",
        type=float,
        nargs="+",
        default=None,
        help="Explicit Fisheye624 parameters. Pass 8 values (fx fy cx cy k1 k2 k3 k4) or all 16 values. If omitted, parameters are fitted from rays.",
    )
    p.add_argument(
        "--camera",
        type=str,
        default="auto",
        choices=["auto", "perspective", "pinhole", "fisheye", "panorama", "erp"],
        help="Override automatic ray-range camera classification.",
    )
    p.add_argument("--fisheye-fov-threshold-deg", type=float, default=95.0)
    p.add_argument("--fisheye-diag-threshold-deg", type=float, default=130.0)
    p.add_argument("--fisheye-vfov-min-deg", type=float, default=70.0)
    p.add_argument("--fisheye-max-aspect", type=float, default=1.65)
    p.add_argument("--panorama-hfov-threshold-deg", type=float, default=260.0)
    p.add_argument("--panorama-vfov-threshold-deg", type=float, default=120.0)
    p.add_argument("--panorama-aspect-min", type=float, default=1.75)
    p.add_argument("--panorama-aspect-max", type=float, default=2.25)
    p.add_argument("--low-pass-filter-eps", type=float, default=0.0)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _configure_torchhub_cache()
    args = _build_argparser().parse_args()
    args._camera_json_data = _load_camera_json(args.camera_json)
    device = torch.device(str(args.device))
    model, step = _load_model(Path(args.checkpoint), device=device)
    renderer = GSplatRenderer(
        color_space="sRGB",
        background_color="black",
        low_pass_filter_eps=float(args.low_pass_filter_eps),
    ).to(device)
    train_renderer = UnifiedTrainer(
        model=model,
        renderer=renderer,
        loss_fn=None,
        device=device,
    )
    image_paths = _collect_image_paths(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info("Rendering %d image(s) to %s", len(image_paths), args.out_dir)
    for image_path in image_paths:
        _process_one(
            model=model,
            renderer=renderer,
            train_renderer=train_renderer,
            image_path=Path(image_path),
            out_root=Path(args.out_dir),
            step=int(step),
            args=args,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
