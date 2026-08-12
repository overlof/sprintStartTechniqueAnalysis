#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
enrich_result_for_1c_v6.py

Автоматическая постобработка результата нейронки для 1С.
Используется как модуль внутри scripts/run_from_1c_full_analysis.py.

1С нажимает кнопку -> запускает run_from_1c_full_analysis.py.
Дальше всё происходит автоматически:
- predict_valid_class_all_v1.py создает raw/reports/predictions_valid_class.json;
- run_from_1c_full_analysis.py вызывает create_1c_result(...);
- появляется outputs/from_1c_python/result_1c.json;
- result_paths.json получает поле result_1c_json.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ERROR_KEYS = [
    "body_rises_early",
    "front_knee_bad",
    "rear_knee_bad",
    "hip_position_bad",
    "first_step_bad",
    "arms_bad",
]

ERROR_TITLES = {
    "body_rises_early": "Ранний подъем корпуса",
    "front_knee_bad": "Неправильное положение переднего колена",
    "rear_knee_bad": "Неправильное положение заднего колена",
    "hip_position_bad": "Неправильное положение таза",
    "first_step_bad": "Ошибка первого шага",
    "arms_bad": "Ошибка работы рук",
}


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_video_meta(video_path: Optional[str | Path]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "video_path": str(video_path) if video_path else None,
        "file_name": Path(video_path).name if video_path else None,
        "file_size_mb": None,
        "fps": None,
        "width": None,
        "height": None,
        "resolution": None,
        "total_frames": None,
        "duration_seconds": None,
        "codec_fourcc": None,
    }

    if not video_path:
        return meta

    path = Path(video_path)
    if path.exists() and path.is_file():
        meta["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)

    # If input is folder, use first video for video meta.
    if path.exists() and path.is_dir():
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        vids = sorted([p for p in path.rglob("*") if p.suffix.lower() in video_exts])
        if vids:
            path = vids[0]
            meta["video_path"] = str(path)
            meta["file_name"] = path.name
            meta["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
        duration = total_frames / fps if fps else None
        cap.release()

        meta.update({
            "fps": round(float(fps), 3) if fps else None,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}" if width and height else None,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 3) if duration else None,
            "codec_fourcc": codec,
        })
    except Exception as exc:
        meta["video_meta_error"] = str(exc)

    return meta


def first_item_from_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Поддерживаемые форматы:
    1) raw predictor:
       {"items": [{"probabilities": {...}, "valid_start_probability": ..., "no_pose_ratio": ...}]}
    2) polished wrapper:
       {"items": [{"raw_probabilities": {...}, ...}]}
    3) simple:
       {"error_probabilities": {...}}
    4) list:
       [{...}]
    """
    if isinstance(data, list):
        return data[0] if data else {}

    if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
        return data["items"][0]

    return data if isinstance(data, dict) else {}


def normalize_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    item = first_item_from_predictions(data)

    probs = (
        item.get("error_probabilities")
        or item.get("probabilities")
        or item.get("raw_probabilities")
        or {}
    )

    # Remove valid_start from error probabilities, keep it separately.
    clean_probs = {}
    for key in ERROR_KEYS:
        try:
            clean_probs[key] = float(probs.get(key, item.get(key, 0.0)) or 0.0)
        except (TypeError, ValueError):
            clean_probs[key] = 0.0

    valid_start_probability = item.get("valid_start_probability", probs.get("valid_start"))
    try:
        valid_start_probability = float(valid_start_probability)
    except (TypeError, ValueError):
        valid_start_probability = max(0.0, 1.0 - max(clean_probs.values(), default=0.0))

    # Pose quality from raw predictor.
    no_pose_ratio = item.get("no_pose_ratio")
    pose_visibility = item.get("pose_visibility_mean")

    normalized = {
        "filename": item.get("filename"),
        "path": item.get("path"),
        "input_status": item.get("input_status"),
        "score": item.get("score"),
        "valid_start_probability": valid_start_probability,
        "error_probabilities": clean_probs,
        "pose_visibility_mean": pose_visibility,
        "no_pose_ratio": no_pose_ratio,
        "processed_frames": item.get("processed_frames"),
        "valid_pose_frames": item.get("valid_pose_frames"),
    }
    return normalized


def level_by_probability(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.45:
        return "medium"
    if probability >= 0.20:
        return "low"
    return "none"


def level_ru(level: str) -> str:
    return {
        "low": "слабое отклонение",
        "medium": "средняя выраженность",
        "high": "критическая ошибка",
        "none": "ошибка не выявлена",
    }.get(level, level)


def select_recommendations(catalog: Dict[str, Any], error_key: str, level: str, probability: float) -> list[str]:
    if level == "none":
        return []

    cfg = catalog.get(error_key, {})
    level_cfg = cfg.get("levels", {}).get(level, {})
    count = int(level_cfg.get("recommendations_count", {"low": 2, "medium": 3, "high": 4}.get(level, 2)))

    items = cfg.get(level, [])
    if not items:
        return []

    seed = f"{error_key}:{level}:{round(probability, 2)}"
    rng = random.Random(seed)
    return rng.sample(items, min(count, len(items)))


def quality_label(success_percent: Optional[float]) -> str:
    if success_percent is None:
        return "не определено"
    if success_percent >= 90:
        return "высокое"
    if success_percent >= 75:
        return "среднее"
    return "низкое"


def build_1c_result(
    predictions: Dict[str, Any],
    recommendations_catalog: Dict[str, Any],
    video_path: Optional[str | Path] = None,
    model_version: str = "neural_error_lstm_final.keras",
) -> Dict[str, Any]:
    pred = normalize_predictions(predictions)
    video_meta = get_video_meta(video_path or pred.get("path"))

    total_frames = int(video_meta.get("total_frames") or 0)

    processed_frames = pred.get("processed_frames")
    if processed_frames is None:
        processed_frames = total_frames if total_frames else None

    no_pose_ratio = pred.get("no_pose_ratio")
    valid_pose_frames = pred.get("valid_pose_frames")

    if valid_pose_frames is None and processed_frames is not None and no_pose_ratio is not None:
        try:
            valid_pose_frames = int(round(float(processed_frames) * (1.0 - float(no_pose_ratio))))
        except Exception:
            valid_pose_frames = None

    lost_pose_frames = None
    pose_loss_percent = None
    pose_success_percent = None

    if processed_frames:
        if valid_pose_frames is None:
            valid_pose_frames = processed_frames
        lost_pose_frames = max(0, int(processed_frames) - int(valid_pose_frames))
        pose_loss_percent = round(lost_pose_frames / int(processed_frames) * 100, 2)
        pose_success_percent = round(int(valid_pose_frames) / int(processed_frames) * 100, 2)

    error_probs = pred["error_probabilities"]

    detected_errors = []
    all_recommendations = []

    for key in ERROR_KEYS:
        probability = float(error_probs.get(key, 0.0))
        level = level_by_probability(probability)
        recs = select_recommendations(recommendations_catalog, key, level, probability)

        if level != "none":
            detected_errors.append({
                "code": key,
                "name": ERROR_TITLES[key],
                "probability": round(probability, 4),
                "probability_percent": round(probability * 100, 1),
                "level": level,
                "level_ru": level_ru(level),
                "recommendations": recs,
            })
            for rec in recs:
                all_recommendations.append({
                    "error_code": key,
                    "error_name": ERROR_TITLES[key],
                    "probability": round(probability, 4),
                    "level": level,
                    "level_ru": level_ru(level),
                    "text": rec,
                })

    valid_start_probability = float(pred["valid_start_probability"])
    athlete_score = pred.get("score")
    if athlete_score is None:
        athlete_score = int(round(valid_start_probability * 100))

    return {
        "analysis_created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "sprint_start_ai",
        "model_version": model_version,
        "input_status": pred.get("input_status"),
        "video": video_meta,
        "technical_quality": {
            "processed_frames": processed_frames,
            "valid_pose_frames": valid_pose_frames,
            "lost_pose_frames": lost_pose_frames,
            "pose_loss_percent": pose_loss_percent,
            "pose_success_percent": pose_success_percent,
            "recognition_quality_label": quality_label(pose_success_percent),
            "pose_visibility_mean": pred.get("pose_visibility_mean"),
            "no_pose_ratio": pred.get("no_pose_ratio"),
        },
        "valid_start_probability": round(valid_start_probability, 4),
        "valid_start_probability_percent": round(valid_start_probability * 100, 1),
        "athlete_score": athlete_score,
        "error_probabilities": {key: round(float(error_probs.get(key, 0.0)), 4) for key in ERROR_KEYS},
        "detected_errors": detected_errors,
        "recommendations": all_recommendations,
        "summary": {
            "errors_count": len(detected_errors),
            "critical_errors_count": sum(1 for e in detected_errors if e["level"] == "high"),
            "medium_errors_count": sum(1 for e in detected_errors if e["level"] == "medium"),
            "low_errors_count": sum(1 for e in detected_errors if e["level"] == "low"),
        },
        "for_1c": {
            "status": "ready",
            "main_document": "ЗагрузкаВидео",
            "recommended_registers": [
                "РезультатыАнализаВидео",
                "ВероятностиТехническихОшибок",
                "КачествоВидеоИРаспознавания",
                "ЖурналЗапусковАнализа",
            ],
        },
    }


def create_1c_result(
    predictions_path: str | Path,
    recommendations_path: str | Path,
    out_path: str | Path,
    video_path: Optional[str | Path] = None,
    model_version: str = "neural_error_lstm_final.keras",
) -> Dict[str, Any]:
    predictions = read_json(predictions_path)
    recommendations_catalog = read_json(recommendations_path)
    result = build_1c_result(
        predictions=predictions,
        recommendations_catalog=recommendations_catalog,
        video_path=video_path,
        model_version=model_version,
    )
    write_json(out_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать result_1c.json из результата нейронки.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--recommendations", default="config/recommendations_catalog_v6.json")
    parser.add_argument("--out", default="outputs/result_1c.json")
    parser.add_argument("--video", default=None)
    parser.add_argument("--model-version", default="neural_error_lstm_final.keras")
    args = parser.parse_args()

    create_1c_result(
        predictions_path=args.predictions,
        recommendations_path=args.recommendations,
        out_path=args.out,
        video_path=args.video,
        model_version=args.model_version,
    )
    print(f"JSON для 1С сохранен: {args.out}")


if __name__ == "__main__":
    main()
