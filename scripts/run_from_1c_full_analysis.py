# -*- coding: utf-8 -*-
"""Entry point for 1C. Shared implementation is kept in run_from_1c_full_analysis_core.py."""
from run_from_1c_full_analysis_core import *


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sprint Start AI full analysis directly from 1C without PowerShell.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/from_1c_python")
    parser.add_argument("--model")
    parser.add_argument("--meta")
    parser.add_argument("--pose-model")
    parser.add_argument("--config")
    parser.add_argument("--skip-skeleton", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--open-html", action="store_true", help="Compatibility flag; HTML is never opened automatically")
    parser.add_argument("--force-reanalyze", action="store_true")
    args = parser.parse_args()

    create_html = not args.skip_html
    create_skeleton = not args.skip_skeleton
    root = Path(args.project_root).resolve()
    video = project_path(root, args.video).resolve()
    out = project_path(root, args.output_dir).resolve()
    if not root.exists() or not video.exists():
        print(f"ERROR: path not found: {root if not root.exists() else video}", file=sys.stderr)
        return 2

    model = project_path(root, args.model).resolve() if args.model else (root / "models/neural_lstm_valid_class_all_v1.keras").resolve()
    meta = project_path(root, args.meta).resolve() if args.meta else (root / "models/neural_lstm_valid_class_all_v1_meta.json").resolve()
    pose_model = project_path(root, args.pose_model).resolve() if args.pose_model else (root / "models/pose_landmarker_full.task").resolve()
    config = project_path(root, args.config).resolve() if args.config else (root / "config/predict_valid_class_v1.json").resolve()

    try:
        if not args.force_reanalyze:
            cached_paths, cached_data, signature = find_cached_analysis(out, video, model, meta, pose_model, config, create_html, create_skeleton)
            if cached_paths is not None and cached_data is not None:
                current = write_cache_hit_result(out, cached_paths, cached_data, signature, video, root, model, meta, pose_model, config, create_html, create_skeleton)
                print("CACHE_HIT=1")
                print(f"CACHE_SOURCE={cached_paths.parent}")
                print(f"RESULT_PATHS={current}")
                return 0
        else:
            signature = build_cache_signature(video, model, meta, pose_model, config, create_html, create_skeleton)
    except Exception as exc:
        print(f"CACHE_LOOKUP_WARNING: {exc}", file=sys.stderr)
        signature = build_cache_signature(video, model, meta, pose_model, config, create_html, create_skeleton)

    if 'signature' not in locals():
        signature = build_cache_signature(video, model, meta, pose_model, config, create_html, create_skeleton)

    required_modules = {"numpy": "numpy", "cv2": "opencv-python", "tensorflow": "tensorflow", "mediapipe": "mediapipe"}
    missing_modules = [pkg for mod, pkg in required_modules.items() if importlib.util.find_spec(mod) is None]
    if missing_modules:
        ensure_dir(out)
        text = "В выбранном Python не установлены обязательные пакеты: " + ", ".join(missing_modules) + ". Установите: python -m pip install -r requirements.txt"
        save_json(out / "result_paths.json", {"status": "error", "error": text, "project_root": str(root), "source_video": str(video), "output_dir": str(out)})
        print("ERROR: " + text, file=sys.stderr)
        return 5

    raw_reports = out / "raw/reports"; raw_metrics = out / "raw/metrics"
    final_reports = out / "polished/reports"; final_metrics = out / "polished/metrics"
    skeleton_dir = out / "skeleton_videos"; logs_dir = out / "logs"; input_dir = out / "input_video"
    for p in (raw_reports, raw_metrics, final_reports, final_metrics, skeleton_dir, logs_dir, input_dir):
        ensure_dir(p)
    analysis_input = copy_or_link_video(video, input_dir) if video.is_file() else video

    raw_json = raw_reports / "predictions_valid_class.json"
    raw_csv = raw_metrics / "predictions_valid_class.csv"
    raw_html = raw_reports / "index_valid_class.html"
    final_json = final_reports / "predictions_valid_class_v1_1.json"
    final_csv = final_metrics / "predictions_valid_class_v1_1.csv"
    final_html = final_reports / "index_valid_class_v1_1.html"
    result_1c = out / "result_1c.json"; result_paths = out / "result_paths.json"; log = logs_dir / "run_from_1c_full_analysis.log"

    predictor = root / "scripts/predict_valid_class_all_v1.py"
    catalog = root / "config/recommendations_catalog_v6.json"
    enricher = root / "scripts/enrich_result_for_1c_v6.py"
    missing = [str(p) for p in (predictor, model, meta, pose_model, config, catalog, enricher) if not p.exists()]
    if missing:
        text = "Не найдены обязательные файлы:\n" + "\n".join(missing)
        save_json(result_paths, {"status": "error", "error": text, "project_root": str(root), "source_video": str(video), "output_dir": str(out)})
        print("ERROR: " + text, file=sys.stderr)
        return 4

    try:
        cmd = [sys.executable, str(predictor), "--input", str(analysis_input), "--model", str(model), "--meta", str(meta), "--pose-model", str(pose_model), "--config", str(config), "--output-json", str(raw_json), "--output-csv", str(raw_csv)]
        cmd += ["--output-html", str(raw_html)] if create_html else ["--skip-html"]
        print("CACHE_HIT=0")
        run_command(cmd, root, log, "LSTM valid-class predictor")
        polished = polish_predictions(raw_json, final_json, final_csv, final_html if create_html else None, config)

        try:
            from scripts.enrich_result_for_1c_v6 import create_1c_result
        except ModuleNotFoundError:
            from enrich_result_for_1c_v6 import create_1c_result
        create_1c_result(predictions_path=raw_json, recommendations_path=catalog, out_path=result_1c, video_path=analysis_input, model_version=model.name)

        skeleton_status = "skipped"
        annotator = root / "scripts/annotate_skeletons_for_lstm_check_v1.py"
        if create_skeleton and annotator.exists():
            try:
                run_command([sys.executable, str(annotator), "--input", str(analysis_input), "--output-dir", str(skeleton_dir), "--pose-model", str(pose_model), "--max-videos", "1", "--fps", "18"], root, log, "Skeleton video annotation")
                skeleton_status = "created"
            except Exception as exc:
                skeleton_status = f"failed: {exc}"

        data = {
            "status": "ok", "cache_hit": False, "cache_message": "Готового совместимого анализа не найдено; нейросеть выполнена заново.",
            "cache_signature": signature, "project_root": str(root), "source_video": str(video), "requested_video": str(video), "input": str(analysis_input), "output_dir": str(out),
            "model": str(model), "meta": str(meta), "pose_model": str(pose_model), "threshold_config": str(config),
            "analysis_options": {"create_html": create_html, "create_skeleton": create_skeleton},
            "raw_json": str(raw_json), "raw_csv": str(raw_csv), "raw_html": str(raw_html) if create_html else "", "html_status": "created" if create_html else "skipped",
            "final_json": str(final_json), "final_csv": str(final_csv), "final_html": str(final_html) if create_html else "", "result_1c_json": str(result_1c),
            "skeleton_dir": str(skeleton_dir) if skeleton_status == "created" else "", "skeleton_status": skeleton_status, "log": str(log),
            "items_count": polished.get("items_count"), "average_score": polished.get("average_score")
        }
        save_json(result_paths, data)
        print("DONE")
        print(f"RESULT_1C_JSON={result_1c}")
        print(f"RESULT_PATHS={result_paths}")
        return 0
    except Exception as exc:
        save_json(result_paths, {"status": "error", "error": str(exc), "cache_hit": False, "cache_signature": signature, "project_root": str(root), "source_video": str(video), "output_dir": str(out), "log": str(log)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
