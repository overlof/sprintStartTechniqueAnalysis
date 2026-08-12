import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except Exception as e:
    raise RuntimeError(
        "Could not import MediaPipe Tasks API. Install/update mediapipe: python.exe -m pip install --upgrade mediapipe opencv-python"
    ) from e


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def load_config(path: Path | None):
    cfg = {
        "sequence_length": 72,
        "landmark_count": 33,
        "values_per_landmark": 4,
        "min_detection_confidence": 0.3,
        "min_presence_confidence": 0.3,
        "min_tracking_confidence": 0.3,
        "use_first_person_only": True,
    }
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg


def list_videos(input_dir: Path):
    if input_dir.is_file() and input_dir.suffix.lower() in VIDEO_EXTS:
        return [input_dir]
    videos = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(videos)


def sample_indices(frame_count: int, seq_len: int):
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0] * seq_len
    return np.linspace(0, frame_count - 1, seq_len).round().astype(int).tolist()


def landmark_to_vec(result, landmark_count: int, values_per_landmark: int):
    # Always return fixed flat shape: landmark_count * values_per_landmark
    out = np.zeros((landmark_count, values_per_landmark), dtype=np.float32)
    if not getattr(result, "pose_landmarks", None):
        return out.reshape(-1), False
    if len(result.pose_landmarks) == 0:
        return out.reshape(-1), False

    landmarks = result.pose_landmarks[0]
    if not landmarks:
        return out.reshape(-1), False

    limit = min(landmark_count, len(landmarks))
    for i in range(limit):
        lm = landmarks[i]
        # Tasks NormalizedLandmark has x/y/z/visibility. Visibility may be absent in some builds.
        out[i, 0] = float(getattr(lm, "x", 0.0) or 0.0)
        out[i, 1] = float(getattr(lm, "y", 0.0) or 0.0)
        out[i, 2] = float(getattr(lm, "z", 0.0) or 0.0)
        if values_per_landmark >= 4:
            out[i, 3] = float(getattr(lm, "visibility", 0.0) or 0.0)
    return out.reshape(-1), True


def read_selected_frames(video_path: Path, indices):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
        else:
            frames.append(frame)
    cap.release()
    return frames


def process_video(video_path: Path, landmarker, cfg):
    seq_len = int(cfg["sequence_length"])
    landmark_count = int(cfg["landmark_count"])
    values_per_landmark = int(cfg["values_per_landmark"])
    flat_dim = landmark_count * values_per_landmark

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    indices = sample_indices(frame_count, seq_len)
    frames = read_selected_frames(video_path, indices)

    seq = np.zeros((seq_len, flat_dim), dtype=np.float32)
    pose_found_flags = []

    for t in range(seq_len):
        frame = frames[t] if t < len(frames) else None
        if frame is None:
            seq[t, :] = 0.0
            pose_found_flags.append(False)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        vec, ok = landmark_to_vec(result, landmark_count, values_per_landmark)
        if vec.shape[0] != flat_dim:
            fixed = np.zeros((flat_dim,), dtype=np.float32)
            n = min(flat_dim, vec.shape[0])
            fixed[:n] = vec[:n]
            vec = fixed
        seq[t, :] = vec.astype(np.float32)
        pose_found_flags.append(ok)

    no_pose_ratio = 1.0 - (sum(pose_found_flags) / max(1, len(pose_found_flags)))
    visibility_values = seq[:, 3::values_per_landmark] if values_per_landmark >= 4 else np.zeros((seq_len, landmark_count), dtype=np.float32)
    nonzero_vis = visibility_values[visibility_values > 0]
    pose_visibility_mean = float(nonzero_vis.mean()) if nonzero_vis.size else 0.0

    return seq, {
        "filename": video_path.name,
        "source_path": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sequence_length": seq_len,
        "feature_dim": flat_dim,
        "pose_found_frames": int(sum(pose_found_flags)),
        "no_pose_ratio": float(no_pose_ratio),
        "pose_visibility_mean": pose_visibility_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract fixed-shape MediaPipe PoseLandmarker sequences: N x 72 x 132")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose_landmarker .task model")
    parser.add_argument("--config", default=None, help="Optional JSON config")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)
    config_path = Path(args.config) if args.config else None

    cfg = load_config(config_path)
    seq_len = int(cfg["sequence_length"])
    flat_dim = int(cfg["landmark_count"]) * int(cfg["values_per_landmark"])

    if not model_path.exists():
        raise FileNotFoundError(f"Pose Landmarker model not found: {model_path}")
    videos = list_videos(input_path)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_path}")

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=float(cfg["min_detection_confidence"]),
        min_pose_presence_confidence=float(cfg["min_presence_confidence"]),
        min_tracking_confidence=float(cfg["min_tracking_confidence"]),
        output_segmentation_masks=False,
    )

    sequences = []
    filenames = []
    summaries = []
    failures = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, video_path in enumerate(videos, start=1):
            try:
                seq, meta = process_video(video_path, landmarker, cfg)
                if seq.shape != (seq_len, flat_dim):
                    raise ValueError(f"Bad sequence shape {seq.shape}, expected {(seq_len, flat_dim)}")
                sequences.append(seq)
                filenames.append(video_path.name)
                summaries.append(meta)
                print(f"[{i}/{len(videos)}] OK {video_path.name}: shape={seq.shape}, no_pose={meta['no_pose_ratio']:.3f}, vis={meta['pose_visibility_mean']:.3f}")
            except Exception as e:
                failures.append({"filename": video_path.name, "source_path": str(video_path), "error": repr(e)})
                print(f"[{i}/{len(videos)}] FAIL {video_path.name}: {repr(e)}")

    if not sequences:
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps({"videos_found": len(videos), "saved": 0, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"No sequences extracted. Summary written to {summary_path}")

    X = np.stack(sequences, axis=0).astype(np.float32)
    np.savez_compressed(
        output_path,
        X=X,
        filename=np.array(filenames, dtype=object),
        pose_visibility_mean=np.array([m["pose_visibility_mean"] for m in summaries], dtype=np.float32),
        no_pose_ratio=np.array([m["no_pose_ratio"] for m in summaries], dtype=np.float32),
    )

    summary = {
        "version": "extract_sequences_final_v4_fixed_shape_image_mode",
        "input": str(input_path),
        "output": str(output_path),
        "videos_found": len(videos),
        "saved": len(sequences),
        "failed": len(failures),
        "X_shape": list(X.shape),
        "sequence_length": seq_len,
        "feature_dim": flat_dim,
        "items": summaries,
        "failures": failures,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"X shape: {X.shape}")
    print(f"Failed: {len(failures)}")


if __name__ == "__main__":
    main()
