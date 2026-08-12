# -*- coding: utf-8 -*-
"""
Sprint Start AI — one-file Python wrapper for 1C integration.

1C calls this script directly:
    python.exe scripts\run_from_1c_full_analysis.py --project-root ... --video ... --output-dir ...

The wrapper runs:
  1) valid_start + LSTM predictor;
  2) polished JSON/CSV/HTML report for 1C;
  3) optional skeleton video for visual quality control.

It does not require PowerShell.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ERROR_LABELS = [
    "body_rises_early",
    "front_knee_bad",
    "rear_knee_bad",
    "hip_position_bad",
    "first_step_bad",
    "arms_bad",
]

RU_NAMES = {
    "valid_start": "Низкий старт",
    "body_rises_early": "Ранний подъём корпуса",
    "front_knee_bad": "Неоптимальный угол переднего колена",
    "rear_knee_bad": "Неоптимальный угол заднего колена",
    "hip_position_bad": "Неоптимальное положение таза",
    "first_step_bad": "Замечание по первому шагу / стартовому выходу",
    "arms_bad": "Замечание по работе рук",
}

SOFT_TEXT = {
    "body_rises_early": {
        "short": "Корпус, вероятно, поднимается слишком рано после выхода из старта.",
        "details": "В первых шагах желательно дольше сохранять наклон корпуса и направлять усилие назад-вниз. Быстрый подъём корпуса может уменьшать горизонтальное ускорение.",
        "recommendations": [
            "Дольше сохранять наклон корпуса в первых 2–3 шагах.",
            "Следить, чтобы движение было направлено вперёд, а не вверх.",
            "Использовать wall drive, falling start и короткие ускорения на 3–5 шагов.",
        ],
        "coach_check": "Проверить, не поднимается ли корпус слишком рано в первые 2–3 шага.",
    },
    "front_knee_bad": {
        "short": "Переднее колено в стартовой позиции стоит проверить визуально.",
        "details": "Угол переднего колена влияет на направление и мощность первого толчка. Слишком закрытое или раскрытое положение может снижать эффективность выхода.",
        "recommendations": [
            "Проверить положение передней колодки и угол переднего колена в положении готовности.",
            "Сравнить стартовую позицию с боковым эталонным ракурсом.",
            "Подобрать расстояние передней колодки, при котором толчок получается наиболее мощным.",
        ],
        "coach_check": "Проверить угол переднего колена и постановку передней колодки.",
    },
    "rear_knee_bad": {
        "short": "Положение задней ноги может быть не самым выгодным для быстрого выноса.",
        "details": "Задняя нога должна быстро включаться в первый шаг. Неудачный угол может приводить к задержке выноса ноги или лишнему вертикальному движению.",
        "recommendations": [
            "Проверить положение задней колодки относительно передней.",
            "Следить, чтобы задняя нога быстро уходила вперёд после старта.",
            "Использовать старты на 1–2 шага с акцентом на быстрый вынос задней ноги.",
        ],
        "coach_check": "Проверить угол заднего колена и скорость выноса задней ноги.",
    },
    "hip_position_bad": {
        "short": "Положение таза желательно проверить визуально.",
        "details": "Таз может быть слишком низко или слишком высоко относительно эффективной стартовой позы. Это влияет на углы ног и направление стартового усилия.",
        "recommendations": [
            "Проверить высоту таза в положении готовности.",
            "Сохранять устойчивое положение плеч, таза и стоп перед стартом.",
            "Выполнять старты из колодок с видеоконтролем бокового ракурса.",
        ],
        "coach_check": "Проверить высоту таза в положении готовности и устойчивость стартовой позы.",
    },
    "first_step_bad": {
        "short": "Первый шаг или стартовый выход требуют внимания.",
        "details": "Первый шаг должен помогать ускорению, а не тормозить движение. Возможная зона риска — слишком длинный шаг, постановка стопы далеко перед центром массы или недостаточно активное движение назад-вниз.",
        "recommendations": [
            "Первый шаг делать активным, коротким и направленным назад-вниз.",
            "Избегать постановки стопы слишком далеко перед центром массы.",
            "Использовать упражнения на первые 2–3 шага из колодок, wall drive и sled push с небольшим сопротивлением.",
        ],
        "coach_check": "Проверить длину и направление первого шага, а также постановку стопы относительно центра массы.",
    },
    "arms_bad": {
        "short": "Работу рук желательно проверить визуально.",
        "details": "Руки помогают сохранять ритм и направление ускорения. Недостаточная амплитуда, запаздывание или пересечение средней линии корпуса могут ухудшать баланс.",
        "recommendations": [
            "Работать руками активно вперёд-назад, без пересечения средней линии корпуса.",
            "Следить за синхронизацией рук и ног в первых шагах.",
            "Добавить упражнения на работу рук сидя, стоя и в коротких ускорениях.",
        ],
        "coach_check": "Проверить активность рук и отсутствие пересечения средней линии корпуса.",
    },
}


def project_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(args: List[str], cwd: Path, log_path: Path, title: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {title} =====\n")
        log.write("COMMAND: " + " ".join([repr(a) for a in args]) + "\n")
        log.flush()
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        log.write(f"RETURN_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{title} failed with return code {proc.returncode}. See log: {log_path}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def score_text(score: Optional[float]) -> Tuple[str, str]:
    if score is None:
        return "оценка не сформирована", "Видео не принято к автоматической спортивной оценке."
    if score >= 90:
        return "отлично", "Техника выглядит сильной, обнаружены только минимальные зоны контроля."
    if score >= 80:
        return "хорошо", "Старт в целом хороший, но есть отдельные зоны для коррекции."
    if score >= 70:
        return "удовлетворительно", "Старт пригоден для анализа, но есть заметные зоны технической коррекции."
    if score >= 60:
        return "требуется доработка", "Обнаружены существенные технические замечания, результат желательно проверить тренеру."
    return "требуется визуальная проверка", "Автоматическая оценка низкая; необходимо проверить качество видео и технику вручную."


def reliability_text(pose_visibility: Optional[float], no_pose_ratio: Optional[float], input_status: str) -> Dict[str, Any]:
    pv = float(pose_visibility or 0.0)
    nr = float(no_pose_ratio or 0.0)
    warnings: List[str] = []

    if input_status != "valid_start":
        return {
            "level": "низкая",
            "text": "Результат не используется как спортивная оценка, потому что видео не прошло входную проверку.",
            "warnings": ["Видео не принято к автоматической спортивной оценке."],
        }

    if pv >= 0.82 and nr <= 0.20:
        level = "высокая"
        text = "Ключевые точки определялись достаточно стабильно; результат можно использовать как автоматическую оценку с обычной визуальной проверкой."
    elif pv >= 0.70 and nr <= 0.40:
        level = "средняя"
        text = "Видео принято к анализу, но часть кадров имела неуверенное распознавание позы. Результат желательно подтвердить визуально."
        warnings.append("Часть кадров имела неуверенное распознавание позы.")
    else:
        level = "низкая"
        text = "Качество распознавания позы снижено. Результат следует рассматривать только как предварительный."
        warnings.append("Низкая устойчивость keypoints или высокая доля кадров без позы.")

    return {"level": level, "text": text, "warnings": warnings}


def classify_label(label: str, prob: float, strong_thr: float, review_thr: float) -> Dict[str, Any]:
    if prob >= strong_thr:
        status = "strong"
        severity = "высокая уверенность"
    elif prob >= review_thr:
        status = "review"
        severity = "проверить визуально"
    else:
        status = "weak"
        severity = "слабый сигнал"
    base = SOFT_TEXT.get(label, {})
    return {
        "label": label,
        "ru_name": RU_NAMES.get(label, label),
        "probability": float(prob),
        "probability_percent": round(float(prob) * 100.0, 1),
        "status": status,
        "severity": severity,
        "short": base.get("short", "Требуется визуальная проверка."),
        "details": base.get("details", "Подробное описание отсутствует."),
        "recommendations": base.get("recommendations", []),
        "coach_check": base.get("coach_check", "Проверить визуально."),
    }


def get_config_thresholds(config_path: Path) -> Dict[str, float]:
    # Defaults are aligned with previous reports.
    defaults = {
        "valid_start_accept_threshold": 0.75,
        "valid_start_review_threshold": 0.50,
        "strong_error_threshold": 0.70,
        "review_error_threshold": 0.55,
    }
    if not config_path.exists():
        return defaults
    try:
        cfg = load_json(config_path)
        for key in list(defaults.keys()):
            if key in cfg:
                defaults[key] = float(cfg[key])
    except Exception:
        pass
    return defaults


def polish_predictions(raw_json: Path, polished_json: Path, polished_csv: Path, polished_html: Path, config_path: Path) -> Dict[str, Any]:
    data = load_json(raw_json)
    thresholds = get_config_thresholds(config_path)
    strong_thr = thresholds["strong_error_threshold"]
    review_thr = thresholds["review_error_threshold"]
    valid_accept = thresholds["valid_start_accept_threshold"]
    valid_review = thresholds["valid_start_review_threshold"]

    polished_items: List[Dict[str, Any]] = []
    for item in data.get("items", []):
        probs = item.get("probabilities", {}) or {}
        valid_prob = float(item.get("valid_start_probability", probs.get("valid_start", 0.0)) or 0.0)
        pose_visibility = float(item.get("pose_visibility_mean", 0.0) or 0.0)
        no_pose_ratio = float(item.get("no_pose_ratio", 1.0) or 1.0)

        if no_pose_ratio > 0.55 or pose_visibility < 0.50:
            input_status = "invalid_low_pose"
        elif valid_prob >= valid_accept:
            input_status = "valid_start"
        elif valid_prob >= valid_review:
            input_status = "borderline_start_review"
        else:
            input_status = "invalid_start"

        if input_status == "valid_start":
            all_labels = [classify_label(label, float(probs.get(label, 0.0) or 0.0), strong_thr, review_thr) for label in ERROR_LABELS]
            strong_errors = [x for x in all_labels if x["status"] == "strong"]
            visual_review = [x for x in all_labels if x["status"] == "review"]
            weak_signals = [x for x in all_labels if x["status"] == "weak"]
            # Prefer raw predictor score if present; otherwise compute conservative score.
            score = item.get("score")
            if score is None:
                penalty = sum(x["probability"] * 15 for x in strong_errors) + sum(x["probability"] * 6 for x in visual_review)
                score = max(0.0, min(100.0, 100.0 - penalty))
            score = round(float(score), 1)
        else:
            all_labels = []
            strong_errors = []
            visual_review = []
            weak_signals = []
            score = None

        st, sm = score_text(score)
        rel = reliability_text(pose_visibility, no_pose_ratio, input_status)

        summary: List[str] = []
        if input_status == "valid_start":
            summary.append(f"Видео распознано как низкий старт с вероятностью {round(valid_prob*100, 1)}%.")
            if strong_errors:
                summary.append("Основные зоны внимания: " + ", ".join([e["ru_name"] for e in strong_errors]) + ".")
            else:
                summary.append("Уверенных технических замечаний не выявлено.")
            if visual_review:
                summary.append("Дополнительно проверить визуально: " + ", ".join([e["ru_name"] for e in visual_review]) + ".")
            summary.append(f"Надёжность результата: {rel['level']}.")
        elif input_status == "borderline_start_review":
            summary.append(f"Видео похоже на низкий старт, но уверенность недостаточная: {round(valid_prob*100, 1)}%.")
            summary.append("Рекомендуется визуальная проверка перед сохранением спортивной оценки.")
        elif input_status == "invalid_low_pose":
            summary.append("Видео не принято к оценке: качество распознавания позы недостаточно стабильное.")
        else:
            summary.append(f"Видео не похоже на низкий старт: вероятность valid_start {round(valid_prob*100, 1)}%.")
            summary.append("Спортивная оценка и технические рекомендации не формируются.")

        coach_checks = [e["coach_check"] for e in strong_errors + visual_review]
        recommendations: List[str] = []
        for e in strong_errors + visual_review:
            for rec in e.get("recommendations", []):
                if rec not in recommendations:
                    recommendations.append(rec)

        polished_items.append({
            "filename": item.get("filename"),
            "path": item.get("path"),
            "input_status": input_status,
            "valid_start_probability": valid_prob,
            "valid_start_probability_percent": round(valid_prob * 100.0, 1),
            "score": score,
            "score_text": st,
            "score_meaning": sm,
            "pose_visibility_mean": round(pose_visibility, 4),
            "no_pose_ratio": round(no_pose_ratio, 4),
            "reliability": rel,
            "summary": summary,
            "strong_errors": strong_errors,
            "visual_review": visual_review,
            "weak_signals": weak_signals,
            "all_labels": all_labels,
            "coach_checks": coach_checks,
            "recommendations": recommendations,
            "raw_probabilities": probs,
            "source_raw_item": item,
        })

    avg_scores = [x["score"] for x in polished_items if isinstance(x.get("score"), (int, float))]
    out = {
        "version": "run_from_1c_full_analysis_py_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_predictions": str(raw_json),
        "average_score": round(sum(avg_scores) / len(avg_scores), 1) if avg_scores else None,
        "items_count": len(polished_items),
        "items": polished_items,
    }
    save_json(polished_json, out)

    ensure_dir(polished_csv.parent)
    with polished_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "input_status", "valid_start_probability_percent", "score", "score_text",
            "reliability", "pose_visibility_mean", "no_pose_ratio", "strong_errors", "visual_review"
        ], delimiter=";")
        writer.writeheader()
        for x in polished_items:
            writer.writerow({
                "filename": x.get("filename"),
                "input_status": x.get("input_status"),
                "valid_start_probability_percent": x.get("valid_start_probability_percent"),
                "score": x.get("score"),
                "score_text": x.get("score_text"),
                "reliability": (x.get("reliability") or {}).get("level"),
                "pose_visibility_mean": x.get("pose_visibility_mean"),
                "no_pose_ratio": x.get("no_pose_ratio"),
                "strong_errors": ", ".join([e["ru_name"] for e in x.get("strong_errors", [])]),
                "visual_review": ", ".join([e["ru_name"] for e in x.get("visual_review", [])]),
            })

    write_html_report(polished_html, out)
    return out


def html_list(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return "<p class='muted'>Нет.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in values) + "</ul>"


def write_html_report(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    cards = []
    for item in data.get("items", []):
        score = item.get("score")
        score_show = "N/A" if score is None else str(score)
        score_class = "bad" if score is None or (isinstance(score, (int, float)) and score < 70) else ("mid" if score < 85 else "good")
        errors_html = ""
        for title, key in [("Основные зоны внимания", "strong_errors"), ("Проверить визуально", "visual_review")]:
            entries = item.get(key, []) or []
            blocks = []
            for e in entries:
                blocks.append(f"""
                <div class='error'>
                  <div class='error-title'>{html.escape(e['ru_name'])} <span>{e['probability_percent']}% — {html.escape(e['severity'])}</span></div>
                  <p><b>Что означает:</b> {html.escape(e['short'])}</p>
                  <p><b>Подробно:</b> {html.escape(e['details'])}</p>
                  <p><b>Рекомендации:</b></p>{html_list(e.get('recommendations', []))}
                </div>
                """)
            if blocks:
                errors_html += f"<h3>{title}</h3>" + "".join(blocks)

        all_rows = ""
        for label, prob in (item.get("raw_probabilities") or {}).items():
            if label == "valid_start":
                continue
            all_rows += f"<tr><td>{html.escape(RU_NAMES.get(label, label))}</td><td>{round(float(prob)*100,1)}%</td></tr>"

        cards.append(f"""
        <section class='card'>
          <div class='top'><div><h2>{html.escape(str(item.get('filename')))}</h2><p class='muted'>{html.escape(str(item.get('input_status')))}</p></div><div class='score {score_class}'>{score_show}</div></div>
          <div class='grid'>
            <div><b>Valid start:</b><br>{item.get('valid_start_probability_percent')}%</div>
            <div><b>Надёжность:</b><br>{html.escape(str((item.get('reliability') or {}).get('level')))}</div>
            <div><b>Кадры без позы:</b><br>{item.get('no_pose_ratio')}</div>
          </div>
          <h3>Краткий вывод</h3>{html_list(item.get('summary', []))}
          <h3>Что проверить тренеру</h3>{html_list(item.get('coach_checks', []))}
          {errors_html}
          <h3>Вероятности по классам</h3>
          <table><thead><tr><th>Показатель</th><th>Вероятность</th></tr></thead><tbody>{all_rows}</tbody></table>
        </section>
        """)

    html_text = f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<title>Sprint Start AI — отчёт для 1С</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#222;margin:0;padding:24px}}h1{{margin:0 0 8px}}.muted{{color:#666}}
.summary{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px;margin:22px 0}}.summary div,.card{{background:white;border-radius:14px;box-shadow:0 2px 10px #0001;padding:18px}}
.big{{font-size:30px;font-weight:bold;margin-top:6px}}.top{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}.score{{font-size:34px;font-weight:bold;border-radius:14px;padding:12px 18px;min-width:82px;text-align:center}}
.good{{background:#e7f7ec;color:#146b2e}}.mid{{background:#fff4d8;color:#805400}}.bad{{background:#ffe2e2;color:#9b1c1c}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:10px;background:#fafafa;border-radius:12px;padding:12px;margin:12px 0}}
.error{{border-left:4px solid #2f5bea;background:#f8faff;border-radius:10px;padding:12px;margin:10px 0}}.error-title{{font-weight:bold;font-size:17px}}.error-title span{{font-weight:normal;color:#666;font-size:14px;margin-left:8px}}
li{{margin:6px 0;line-height:1.4}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;border-bottom:1px solid #eee;padding:8px}}th{{background:#fafafa}}
</style></head><body>
<h1>Sprint Start AI — отчёт для 1С</h1>
<p class='muted'>Python-обёртка без PowerShell. MediaPipe PoseLandmarker + LSTM valid_start + 6 технических замечаний.</p>
<div class='summary'><div><div>Видео</div><div class='big'>{data.get('items_count')}</div></div><div><div>Средняя оценка</div><div class='big'>{data.get('average_score') if data.get('average_score') is not None else 'N/A'}</div></div><div><div>Версия</div><div class='big'>1C Py</div></div></div>
{''.join(cards)}
</body></html>"""
    path.write_text(html_text, encoding="utf-8")


def copy_or_link_video(video: Path, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    if video.is_file():
        dst = output_dir / video.name
        if video.resolve() != dst.resolve():
            shutil.copy2(video, dst)
        return dst
    return video


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sprint Start AI full analysis directly from 1C without PowerShell.")
    parser.add_argument("--project-root", required=True, help="Path to sprint_start_ai project root")
    parser.add_argument("--video", required=True, help="Path to selected video file or folder")
    parser.add_argument("--output-dir", default="outputs/from_1c_python", help="Output directory, relative to project root or absolute")
    parser.add_argument("--skip-skeleton", action="store_true", help="Do not create skeleton video")
    parser.add_argument("--open-html", action="store_true", help="Open final HTML report after analysis")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    video_path = project_path(project_root, args.video).resolve()
    output_dir = project_path(project_root, args.output_dir).resolve()

    if not project_root.exists():
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2
    if not video_path.exists():
        print(f"ERROR: video/folder not found: {video_path}", file=sys.stderr)
        return 3

    raw_dir = output_dir / "raw"
    raw_reports = raw_dir / "reports"
    raw_metrics = raw_dir / "metrics"
    polished_dir = output_dir / "polished"
    polished_reports = polished_dir / "reports"
    polished_metrics = polished_dir / "metrics"
    skeleton_dir = output_dir / "skeleton_videos"
    logs_dir = output_dir / "logs"
    input_copy_dir = output_dir / "input_video"

    for p in [raw_reports, raw_metrics, polished_reports, polished_metrics, skeleton_dir, logs_dir, input_copy_dir]:
        ensure_dir(p)

    # Use copied input for file mode so output remains reproducible.
    analysis_input = copy_or_link_video(video_path, input_copy_dir) if video_path.is_file() else video_path

    raw_json = raw_reports / "predictions_valid_class.json"
    raw_csv = raw_metrics / "predictions_valid_class.csv"
    raw_html = raw_reports / "index_valid_class.html"
    polished_json = polished_reports / "predictions_valid_class_v1_1.json"
    polished_csv = polished_metrics / "predictions_valid_class_v1_1.csv"
    polished_html = polished_reports / "index_valid_class_v1_1.html"
    result_paths_json = output_dir / "result_paths.json"
    log_path = logs_dir / "run_from_1c_full_analysis.log"
    result_1c_json = output_dir / "result_1c.json"

    predictor = project_root / "scripts" / "predict_valid_class_all_v1.py"
    model = project_root / "models" / "neural_lstm_valid_class_all_v1.keras"
    meta = project_root / "models" / "neural_lstm_valid_class_all_v1_meta.json"
    pose_model = project_root / "models" / "pose_landmarker_full.task"
    config = project_root / "config" / "predict_valid_class_v1.json"

    required = [predictor, model, meta, pose_model, config]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: required files not found:\n" + "\n".join(missing), file=sys.stderr)
        return 4

    print("Sprint Start AI — 1C Python wrapper")
    print(f"Project root: {project_root}")
    print(f"Input: {analysis_input}")
    print(f"Output: {output_dir}")

    try:
        run_command([
            sys.executable,
            str(predictor),
            "--input", str(analysis_input),
            "--model", str(model),
            "--meta", str(meta),
            "--pose-model", str(pose_model),
            "--config", str(config),
            "--output-json", str(raw_json),
            "--output-csv", str(raw_csv),
            "--output-html", str(raw_html),
        ], cwd=project_root, log_path=log_path, title="LSTM valid-class predictor")

        polished = polish_predictions(raw_json, polished_json, polished_csv, polished_html, config)

        # v6: automatic postprocessing for 1C.
        # This creates a compact and structured JSON that 1C can load immediately after button click.
        try:
            from scripts.enrich_result_for_1c_v6 import create_1c_result
        except ModuleNotFoundError:
            from enrich_result_for_1c_v6 import create_1c_result

        create_1c_result(
            predictions_path=raw_json,
            recommendations_path=project_root / "config" / "recommendations_catalog_v6.json",
            out_path=result_1c_json,
            video_path=analysis_input,
            model_version=model.name,
        )

        skeleton_status = "skipped"
        skeleton_annotator = project_root / "scripts" / "annotate_skeletons_for_lstm_check_v1.py"
        if not args.skip_skeleton and skeleton_annotator.exists():
            try:
                run_command([
                    sys.executable,
                    str(skeleton_annotator),
                    "--input", str(analysis_input),
                    "--output-dir", str(skeleton_dir),
                    "--pose-model", str(pose_model),
                    "--max-videos", "1",
                    "--fps", "18",
                ], cwd=project_root, log_path=log_path, title="Skeleton video annotation")
                skeleton_status = "created"
            except Exception as exc:
                skeleton_status = f"failed: {exc}"
        elif not skeleton_annotator.exists():
            skeleton_status = "skipped: annotator script not found"

        result_paths = {
            "status": "ok",
            "project_root": str(project_root),
            "input": str(analysis_input),
            "output_dir": str(output_dir),
            "raw_json": str(raw_json),
            "raw_csv": str(raw_csv),
            "raw_html": str(raw_html),
            "final_json": str(polished_json),
            "final_csv": str(polished_csv),
            "final_html": str(polished_html),
            "result_1c_json": str(result_1c_json),
            "skeleton_dir": str(skeleton_dir),
            "skeleton_status": skeleton_status,
            "log": str(log_path),
            "items_count": polished.get("items_count"),
            "average_score": polished.get("average_score"),
        }
        save_json(result_paths_json, result_paths)

        print("DONE")
        print(f"FINAL_JSON={polished_json}")
        print(f"FINAL_HTML={polished_html}")
        print(f"RESULT_1C_JSON={result_1c_json}")
        print(f"SKELETON_DIR={skeleton_dir}")
        print(f"RESULT_PATHS={result_paths_json}")

        if args.open_html:
            try:
                os.startfile(str(polished_html))  # type: ignore[attr-defined]
            except Exception:
                pass
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": str(exc),
            "project_root": str(project_root),
            "input": str(video_path),
            "output_dir": str(output_dir),
            "log": str(log_path),
        }
        save_json(result_paths_json, error)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"See log: {log_path}", file=sys.stderr)
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
