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
import hashlib
import importlib.util
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


def read_log_tail(log_path: Path, max_lines: int = 18, max_chars: int = 2600) -> str:
    """Return a compact UTF-8 diagnostic tail suitable for showing inside 1C."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def run_command(args: List[str], cwd: Path, log_path: Path, title: str) -> None:
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    # Child Python writes UTF-8 to the log even on a Russian Windows code page.
    env["PYTHONUTF8"] = "1"
    # Hide routine TensorFlow INFO messages; warnings/errors and traceback remain visible.
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
            env=env,
        )
        log.write(f"RETURN_CODE: {proc.returncode}\n")
    if proc.returncode != 0:
        tail = read_log_tail(log_path)
        message = f"{title} завершился с кодом {proc.returncode}."
        if tail:
            message += "\nПоследние строки журнала:\n" + tail
        message += f"\nПолный журнал: {log_path}"
        raise RuntimeError(message)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def score_text(score: Optional[float], thresholds: Dict[str, float]) -> Tuple[str, str]:
    if score is None:
        return "оценка не сформирована", "Видео не принято к автоматической спортивной оценке."
    if score >= thresholds["report_score_excellent_threshold"]:
        return "отлично", "Техника выглядит сильной, обнаружены только минимальные зоны контроля."
    if score >= thresholds["report_score_good_threshold"]:
        return "хорошо", "Старт в целом хороший, но есть отдельные зоны для коррекции."
    if score >= thresholds["report_score_satisfactory_threshold"]:
        return "удовлетворительно", "Старт пригоден для анализа, но есть заметные зоны технической коррекции."
    if score >= thresholds["report_score_rework_threshold"]:
        return "требуется доработка", "Обнаружены существенные технические замечания, результат желательно проверить тренеру."
    return "требуется визуальная проверка", "Автоматическая оценка низкая; необходимо проверить качество видео и технику вручную."


def reliability_text(pose_visibility: Optional[float], no_pose_ratio: Optional[float], input_status: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
    pv = float(pose_visibility or 0.0)
    nr = float(no_pose_ratio or 0.0)
    warnings: List[str] = []

    if input_status != "valid_start":
        return {
            "level": "низкая",
            "text": "Результат не используется как спортивная оценка, потому что видео не прошло входную проверку.",
            "warnings": ["Видео не принято к автоматической спортивной оценке."],
        }

    if pv >= thresholds["reliability_high_visibility_min"] and nr <= thresholds["reliability_high_no_pose_max"]:
        level = "высокая"
        text = "Ключевые точки определялись достаточно стабильно; результат можно использовать как автоматическую оценку с обычной визуальной проверкой."
    elif pv >= thresholds["reliability_medium_visibility_min"] and nr <= thresholds["reliability_medium_no_pose_max"]:
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


ONE_C_THRESHOLD_ARGUMENTS = {
    "valid_start_threshold_pct": ("valid_start_threshold", True),
    "borderline_start_min_pct": ("borderline_start_min", True),
    "pose_visibility_min_pct": ("pose_visibility_min", True),
    "no_pose_ratio_max_pct": ("no_pose_ratio_max", True),
    "error_signal_min_pct": ("error_signal_min", True),
    "visual_review_threshold_pct": ("visual_review_threshold", True),
    "strong_error_threshold_pct": ("strong_error_threshold", True),
    "recommendation_medium_threshold_pct": ("recommendation_medium_threshold", True),
    "recognition_quality_medium_pct": ("recognition_quality_medium_percent", False),
    "recognition_quality_high_pct": ("recognition_quality_high_percent", False),
    "reliability_medium_visibility_pct": ("reliability_medium_visibility_min", True),
    "reliability_high_visibility_pct": ("reliability_high_visibility_min", True),
    "reliability_high_no_pose_pct": ("reliability_high_no_pose_max", True),
    "reliability_medium_no_pose_pct": ("reliability_medium_no_pose_max", True),
    "report_score_excellent_pct": ("report_score_excellent_threshold", False),
    "report_score_good_pct": ("report_score_good_threshold", False),
    "report_score_satisfactory_pct": ("report_score_satisfactory_threshold", False),
    "report_score_rework_pct": ("report_score_rework_threshold", False),
    "score_good_pct": ("score_good_threshold", False),
    "score_mid_pct": ("score_mid_threshold", False),
}


def build_effective_config_from_1c(base_config: Path, output_dir: Path, args: argparse.Namespace) -> tuple[Path, Dict[str, float]]:
    """Create the exact threshold config for this analysis from values supplied by 1C.

    Non-threshold settings (sequence length, labels, score penalties, etc.) stay in the
    project config. Every decision threshold is overridden by the catalog values sent
    by 1C. The generated file is stored with the analysis artifacts for reproducibility.
    """
    cfg = load_json(base_config)
    supplied: Dict[str, float] = {}
    missing = []
    for arg_name, (config_key, divide_by_100) in ONE_C_THRESHOLD_ARGUMENTS.items():
        value = getattr(args, arg_name, None)
        if value is None:
            missing.append(arg_name)
            continue
        value = float(value)
        if not (0.0 < value <= 100.0):
            raise ValueError(f"Invalid 1C threshold {arg_name}: {value}; expected 0..100")
        cfg_value = value / 100.0 if divide_by_100 else value
        cfg[config_key] = cfg_value
        supplied[config_key] = cfg_value

    if missing:
        raise ValueError(
            "1C did not pass all required threshold settings: " + ", ".join(missing)
        )

    # Same presentation thresholds are used by the predictor HTML and the polished report.
    cfg["html_score_good_threshold"] = cfg["score_good_threshold"]
    cfg["html_score_mid_threshold"] = cfg["score_mid_threshold"]

    # Cross-check ordering to fail before TensorFlow starts.
    if not (cfg["error_signal_min"] < cfg["visual_review_threshold"] < cfg["strong_error_threshold"]):
        raise ValueError("1C error thresholds must satisfy: weak < visual review < strong")
    if not (cfg["borderline_start_min"] < cfg["valid_start_threshold"]):
        raise ValueError("1C borderline valid-start threshold must be below acceptance threshold")
    if not (cfg["report_score_rework_threshold"] < cfg["report_score_satisfactory_threshold"] < cfg["report_score_good_threshold"] < cfg["report_score_excellent_threshold"]):
        raise ValueError("1C report score thresholds are not strictly increasing")
    if not (cfg["reliability_high_no_pose_max"] < cfg["reliability_medium_no_pose_max"]):
        raise ValueError("High-reliability no-pose limit must be below medium-reliability limit")

    ensure_dir(output_dir)
    runtime_config = output_dir / "thresholds_from_1c.json"
    save_json(runtime_config, cfg)
    return runtime_config, supplied


def get_config_thresholds(config_path: Path) -> Dict[str, float]:
    """Load all interpretation thresholds from the predictor config.

    Thresholds are configuration data and are intentionally not duplicated as
    numeric literals in the Python program. Missing settings are treated as a
    configuration error instead of silently changing analysis behavior.
    """
    cfg = load_json(config_path)
    aliases = {
        "valid_start_accept_threshold": ("valid_start_threshold", "valid_start_accept_threshold"),
        "valid_start_review_threshold": ("borderline_start_min", "valid_start_review_threshold"),
        "strong_error_threshold": ("strong_error_threshold",),
        "review_error_threshold": ("visual_review_threshold", "review_error_threshold"),
        "pose_visibility_min": ("pose_visibility_min",),
        "no_pose_ratio_max": ("no_pose_ratio_max",),
    }
    result: Dict[str, float] = {}
    for target_key, source_keys in aliases.items():
        for source_key in source_keys:
            if source_key in cfg:
                result[target_key] = float(cfg[source_key])
                break
        if target_key not in result:
            raise KeyError(f"Missing threshold in config: {target_key}")
    extra_keys = [
        "report_score_excellent_threshold", "report_score_good_threshold",
        "report_score_satisfactory_threshold", "report_score_rework_threshold",
        "reliability_high_visibility_min", "reliability_high_no_pose_max",
        "reliability_medium_visibility_min", "reliability_medium_no_pose_max",
        "html_score_good_threshold", "html_score_mid_threshold",
    ]
    for key in extra_keys:
        if key not in cfg:
            raise KeyError(f"Missing threshold in config: {key}")
        result[key] = float(cfg[key])
    return result


def polish_predictions(raw_json: Path, polished_json: Path, polished_csv: Path, polished_html: Optional[Path], config_path: Path) -> Dict[str, Any]:
    data = load_json(raw_json)
    thresholds = get_config_thresholds(config_path)
    strong_thr = thresholds["strong_error_threshold"]
    review_thr = thresholds["review_error_threshold"]
    valid_accept = thresholds["valid_start_accept_threshold"]
    valid_review = thresholds["valid_start_review_threshold"]
    pose_visibility_min = thresholds["pose_visibility_min"]
    no_pose_ratio_max = thresholds["no_pose_ratio_max"]

    polished_items: List[Dict[str, Any]] = []
    for item in data.get("items", []):
        probs = item.get("probabilities", {}) or {}
        valid_prob = float(item.get("valid_start_probability", probs.get("valid_start", 0.0)) or 0.0)
        pose_visibility = float(item.get("pose_visibility_mean", 0.0) or 0.0)
        no_pose_ratio = float(item.get("no_pose_ratio", 1.0) or 1.0)

        if no_pose_ratio > no_pose_ratio_max or pose_visibility < pose_visibility_min:
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

        st, sm = score_text(score, thresholds)
        rel = reliability_text(pose_visibility, no_pose_ratio, input_status, thresholds)

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

    if polished_html is not None:
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
        score_class = "bad" if score is None or (isinstance(score, (int, float)) and score < thresholds["html_score_mid_threshold"]) else ("mid" if score < thresholds["html_score_good_threshold"] else "good")
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



def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash a video once so cache reuse is based on file contents, not only its name."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_signature(video_path: Path, model: Path, meta: Path, pose_model: Path, config: Path, create_html: bool, create_skeleton: bool) -> Dict[str, Any]:
    if not video_path.is_file():
        return {
            "version": 2,
            "analysis_options": {"create_html": bool(create_html), "create_skeleton": bool(create_skeleton)},
            "video": {"kind": "directory", "name": video_path.name.lower()},
            "model": {"name": model.name.lower()},
            "meta": {"name": meta.name.lower()},
            "pose_model": {"name": pose_model.name.lower()},
            "threshold_config": {"name": config.name.lower(), "sha256": sha256_file(config)},
        }
    stat = video_path.stat()
    return {
        "version": 2,
        "analysis_options": {"create_html": bool(create_html), "create_skeleton": bool(create_skeleton)},
        "video": {
            "kind": "file",
            "name": video_path.name.lower(),
            "size": int(stat.st_size),
            "sha256": sha256_file(video_path),
        },
        "model": {"name": model.name.lower()},
        "meta": {"name": meta.name.lower()},
        "pose_model": {"name": pose_model.name.lower()},
        "threshold_config": {"name": config.name.lower(), "sha256": sha256_file(config)},
    }


def _component_name_from_result(data: Dict[str, Any], field: str) -> str:
    signature = data.get("cache_signature")
    if isinstance(signature, dict):
        item = signature.get(field)
        if isinstance(item, dict) and item.get("name"):
            return str(item.get("name")).lower()
    value = data.get(field)
    if value:
        try:
            return Path(str(value)).name.lower()
        except Exception:
            return str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return ""


def _same_analysis_components(data: Dict[str, Any], model: Path, meta: Path, pose_model: Path, config: Path) -> bool:
    current = {
        "model": model.name.lower(),
        "meta": meta.name.lower(),
        "pose_model": pose_model.name.lower(),
    }
    for field, expected in current.items():
        found = _component_name_from_result(data, field)
        if found and found != expected:
            return False

    # Threshold values are part of the analysis identity. A changed value in 1C
    # must invalidate the cache even though the generated config filename is the same.
    signature = data.get("cache_signature")
    if isinstance(signature, dict):
        old_cfg = signature.get("threshold_config")
        if isinstance(old_cfg, dict) and old_cfg.get("sha256"):
            return str(old_cfg.get("sha256")).lower() == sha256_file(config).lower()
    # Results created before thresholds were centralized in 1C are not reused,
    # because their effective values cannot be proven identical.
    return False


def _same_analysis_options(candidate_dir: Path, data: Dict[str, Any], create_html: bool, create_skeleton: bool) -> bool:
    """Return True when a cached analysis can satisfy the currently enabled outputs.

    Extra files in an old cache are harmless when the corresponding checkbox is off;
    they are simply not exposed by the new result_paths. But if the user currently
    asks for HTML or a skeleton video, that artifact must actually exist.
    """
    if create_html:
        html_value = data.get("final_html") or data.get("raw_html")
        if not html_value:
            return False
        try:
            html_path = Path(str(html_value))
            if not html_path.is_absolute():
                html_path = candidate_dir / html_path
            if not html_path.is_file():
                return False
        except Exception:
            return False

    if create_skeleton:
        if str(data.get("skeleton_status", "")).lower() != "created":
            return False
        skeleton_value = data.get("skeleton_video") or data.get("skeleton_dir") or data.get("skeleton_videos")
        if not skeleton_value:
            return False
        try:
            skeleton_path = Path(str(skeleton_value))
            if not skeleton_path.is_absolute():
                skeleton_path = candidate_dir / skeleton_path
            if skeleton_path.is_file():
                pass
            elif skeleton_path.is_dir():
                if not any(x.is_file() for x in skeleton_path.iterdir()):
                    return False
            else:
                return False
        except Exception:
            return False

    return True


def _cached_video_matches(candidate_dir: Path, data: Dict[str, Any], current_signature: Dict[str, Any]) -> bool:
    current_video = current_signature.get("video") or {}
    if current_video.get("kind") != "file":
        return False

    old_signature = data.get("cache_signature")
    if isinstance(old_signature, dict):
        old_video = old_signature.get("video")
        if isinstance(old_video, dict) and old_video.get("sha256"):
            return (
                int(old_video.get("size", -1)) == int(current_video.get("size", -2))
                and str(old_video.get("sha256")).lower() == str(current_video.get("sha256")).lower()
            )

    # Compatibility with analyses produced before v41: hash the preserved copy
    # in input_video only after inexpensive name/size filtering.
    possible: List[Path] = []
    for key in ("source_video", "requested_video", "input"):
        value = data.get(key)
        if value:
            try:
                possible.append(Path(str(value)))
            except Exception:
                pass
    input_dir = candidate_dir / "input_video"
    if input_dir.exists():
        exact = input_dir / str(current_video.get("name", ""))
        possible.append(exact)
        try:
            possible.extend(input_dir.glob("*"))
        except Exception:
            pass

    seen = set()
    for path in possible:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.is_file():
                continue
            if path.name.lower() != str(current_video.get("name", "")).lower():
                continue
            if path.stat().st_size != int(current_video.get("size", -1)):
                continue
            if sha256_file(path).lower() == str(current_video.get("sha256", "")).lower():
                return True
        except Exception:
            continue
    return False


def _resolve_cached_result_1c(candidate_dir: Path, data: Dict[str, Any]) -> Optional[Path]:
    for key in ("result_1c_json", "result_1c", "result_1c_path"):
        value = data.get(key)
        if value:
            try:
                path = Path(str(value))
                if path.is_file():
                    return path.resolve()
            except Exception:
                pass
    fallback = candidate_dir / "result_1c.json"
    return fallback.resolve() if fallback.is_file() else None


def find_cached_analysis(output_dir: Path, video_path: Path, model: Path, meta: Path, pose_model: Path, config: Path, create_html: bool, create_skeleton: bool) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Find the newest successful analysis of the same video and same model set."""
    signature = build_cache_signature(video_path, model, meta, pose_model, config, create_html, create_skeleton)
    if not video_path.is_file():
        return None, None, signature

    cache_root = output_dir.parent if output_dir.name.lower().startswith("analysis_") else output_dir
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    try:
        result_files = list(cache_root.glob("analysis_*/result_paths.json"))
    except Exception:
        result_files = []

    for result_paths in result_files:
        candidate_dir = result_paths.parent
        try:
            if candidate_dir.resolve() == output_dir.resolve():
                continue
            data = load_json(result_paths)
            if str(data.get("status", "")).strip().lower() != "ok":
                continue
            if not _same_analysis_components(data, model, meta, pose_model, config):
                continue
            if not _same_analysis_options(candidate_dir, data, create_html, create_skeleton):
                continue
            cached_result = _resolve_cached_result_1c(candidate_dir, data)
            if cached_result is None:
                continue
            if not _cached_video_matches(candidate_dir, data, signature):
                continue
            candidates.append((result_paths.stat().st_mtime, result_paths, data))
        except Exception:
            continue

    if not candidates:
        return None, None, signature
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, result_paths, data = candidates[0]
    return result_paths, data, signature


def write_cache_hit_result(
    output_dir: Path,
    source_result_paths: Path,
    source_data: Dict[str, Any],
    signature: Dict[str, Any],
    video_path: Path,
    project_root: Path,
    model: Path,
    meta: Path,
    pose_model: Path,
    config: Path,
    create_html: bool,
    create_skeleton: bool,
) -> Path:
    ensure_dir(output_dir)
    logs_dir = output_dir / "logs"
    ensure_dir(logs_dir)
    log_path = logs_dir / "cache_reuse.log"
    source_dir = source_result_paths.parent.resolve()
    result_1c = _resolve_cached_result_1c(source_dir, source_data)
    if result_1c is None:
        raise RuntimeError("Найденный кэш не содержит result_1c.json.")

    result = dict(source_data)
    source_log = str(source_data.get("log", ""))
    result.update({
        "status": "ok",
        "cache_hit": True,
        "cache_message": "Найден ранее выполненный анализ этого видео; повторный запуск нейросети не выполнялся.",
        "cache_source_output_dir": str(source_dir),
        "cache_source_result_paths": str(source_result_paths.resolve()),
        "cache_source_log": source_log,
        "source_video": str(video_path),
        "requested_video": str(video_path),
        "input": str(video_path),
        "output_dir": str(output_dir),
        "project_root": str(project_root),
        "model": str(model),
        "meta": str(meta),
        "pose_model": str(pose_model),
        "threshold_config": str(config),
        "analysis_options": {"create_html": bool(create_html), "create_skeleton": bool(create_skeleton)},
        "result_1c_json": str(result_1c),
        "cache_signature": signature,
        "log": str(log_path),
    })
    if not create_html:
        result["raw_html"] = ""
        result["final_html"] = ""
        result["html_status"] = "skipped"
    else:
        result["html_status"] = "created"

    if not create_skeleton:
        result["skeleton_video"] = ""
        result["skeleton_dir"] = ""
        result["skeleton_videos"] = ""
        result["skeleton_status"] = "skipped"

    log_path.write_text(
        "CACHE HIT\n"
        f"Requested video: {video_path}\n"
        f"Reused analysis: {source_dir}\n"
        f"Reused result_1c.json: {result_1c}\n"
        f"Source log: {source_log}\n",
        encoding="utf-8",
    )
    current_result_paths = output_dir / "result_paths.json"
    save_json(current_result_paths, result)

    return current_result_paths

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sprint Start AI full analysis directly from 1C without PowerShell.")
    parser.add_argument("--project-root", required=True, help="Path to sprint_start_ai project root")
    parser.add_argument("--video", required=True, help="Path to selected video file or folder")
    parser.add_argument("--output-dir", default="outputs/from_1c_python", help="Output directory, relative to project root or absolute")
    parser.add_argument("--model", help="Keras model selected in 1C; absolute or relative to project root")
    parser.add_argument("--meta", help="Model metadata JSON selected in 1C; absolute or relative to project root")
    parser.add_argument("--pose-model", help="MediaPipe PoseLandmarker model selected in 1C; absolute or relative to project root")
    parser.add_argument("--config", help="Prediction thresholds config; absolute or relative to project root")
    # Thresholds are passed by 1C in percent units. They override the project JSON config.
    parser.add_argument("--valid-start-threshold-pct", type=float)
    parser.add_argument("--borderline-start-min-pct", type=float)
    parser.add_argument("--pose-visibility-min-pct", type=float)
    parser.add_argument("--no-pose-ratio-max-pct", type=float)
    parser.add_argument("--error-signal-min-pct", type=float)
    parser.add_argument("--visual-review-threshold-pct", type=float)
    parser.add_argument("--strong-error-threshold-pct", type=float)
    parser.add_argument("--recommendation-medium-threshold-pct", type=float)
    parser.add_argument("--recognition-quality-medium-pct", type=float)
    parser.add_argument("--recognition-quality-high-pct", type=float)
    parser.add_argument("--reliability-medium-visibility-pct", type=float)
    parser.add_argument("--reliability-high-visibility-pct", type=float)
    parser.add_argument("--reliability-high-no-pose-pct", type=float)
    parser.add_argument("--reliability-medium-no-pose-pct", type=float)
    parser.add_argument("--report-score-excellent-pct", type=float)
    parser.add_argument("--report-score-good-pct", type=float)
    parser.add_argument("--report-score-satisfactory-pct", type=float)
    parser.add_argument("--report-score-rework-pct", type=float)
    parser.add_argument("--score-good-pct", type=float)
    parser.add_argument("--score-mid-pct", type=float)
    parser.add_argument("--skip-skeleton", action="store_true", help="Do not create skeleton video")
    parser.add_argument("--skip-html", action="store_true", help="Do not create HTML reports")
    parser.add_argument("--open-html", action="store_true", help="Deprecated compatibility flag; HTML is never opened automatically")
    parser.add_argument("--force-reanalyze", action="store_true", help="Ignore an existing cached analysis and run the neural network again")
    args = parser.parse_args()
    create_html = not args.skip_html
    create_skeleton = not args.skip_skeleton

    project_root = Path(args.project_root).resolve()
    video_path = project_path(project_root, args.video).resolve()
    output_dir = project_path(project_root, args.output_dir).resolve()

    if not project_root.exists():
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2
    if not video_path.exists():
        print(f"ERROR: video/folder not found: {video_path}", file=sys.stderr)
        return 3

    model = project_path(project_root, args.model).resolve() if args.model else (project_root / "models" / "neural_lstm_valid_class_all_v1.keras").resolve()
    meta = project_path(project_root, args.meta).resolve() if args.meta else (project_root / "models" / "neural_lstm_valid_class_all_v1_meta.json").resolve()
    pose_model = project_path(project_root, args.pose_model).resolve() if args.pose_model else (project_root / "models" / "pose_landmarker_full.task").resolve()
    base_config = project_path(project_root, args.config).resolve() if args.config else (project_root / "config" / "predict_valid_class_v1.json").resolve()
    if not base_config.is_file():
        print(f"ERROR: base prediction config not found: {base_config}", file=sys.stderr)
        return 4
    try:
        config, thresholds_from_1c = build_effective_config_from_1c(base_config, output_dir, args)
    except Exception as exc:
        print(f"ERROR: invalid thresholds received from 1C: {exc}", file=sys.stderr)
        return 6

    # First action after clicking «Анализировать»: look for a successful result
    # of this exact video.  A SHA-256 content match prevents accidental reuse of
    # a different file with the same name.  Model/meta/pose/config names must also
    # match, so changing the selected model forces a real new analysis.
    if not args.force_reanalyze:
        try:
            cached_result_paths, cached_data, cache_signature = find_cached_analysis(
                output_dir, video_path, model, meta, pose_model, config, create_html, create_skeleton
            )
            if cached_result_paths is not None and cached_data is not None:
                current_result_paths = write_cache_hit_result(
                    output_dir=output_dir,
                    source_result_paths=cached_result_paths,
                    source_data=cached_data,
                    signature=cache_signature,
                    video_path=video_path,
                    project_root=project_root,
                    model=model,
                    meta=meta,
                    pose_model=pose_model,
                    config=config,
                    create_html=create_html,
                    create_skeleton=create_skeleton,
                )
                print("CACHE_HIT=1")
                print(f"CACHE_SOURCE={cached_result_paths.parent}")
                print(f"RESULT_PATHS={current_result_paths}")
                return 0
        except Exception as cache_exc:
            # Cache lookup must never prevent a genuine fresh analysis.
            print(f"CACHE_LOOKUP_WARNING: {cache_exc}", file=sys.stderr)
    else:
        cache_signature = build_cache_signature(video_path, model, meta, pose_model, config, create_html, create_skeleton)

    # Fail fast with a readable message before starting a hidden process from 1C.
    required_modules = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "tensorflow": "tensorflow",
        "mediapipe": "mediapipe",
    }
    missing_modules = [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_modules:
        error_text = (
            "В выбранном Python не установлены обязательные пакеты: "
            + ", ".join(missing_modules)
            + ". Установите их командой: python -m pip install -r requirements.txt"
        )
        try:
            save_json(output_dir / "result_paths.json", {
                "status": "error",
                "error": error_text,
                "project_root": str(project_root),
                "input": str(video_path),
                "source_video": str(video_path),
                "output_dir": str(output_dir),
            })
        except Exception:
            pass
        print("ERROR: " + error_text, file=sys.stderr)
        return 5

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

    # Preserve the input in the analysis folder; v41 also records the original
    # source path and content hash to make later reuse deterministic.
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
    recommendations_catalog = project_root / "config" / "recommendations_catalog_v6.json"
    enricher = project_root / "scripts" / "enrich_result_for_1c_v6.py"
    required = [predictor, model, meta, pose_model, config, recommendations_catalog, enricher]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        error_text = "Не найдены обязательные файлы:\n" + "\n".join(missing)
        save_json(result_paths_json, {
            "status": "error",
            "error": error_text,
            "project_root": str(project_root),
            "input": str(video_path),
            "source_video": str(video_path),
            "output_dir": str(output_dir),
        })
        print("ERROR: " + error_text, file=sys.stderr)
        return 4

    if 'cache_signature' not in locals():
        cache_signature = build_cache_signature(video_path, model, meta, pose_model, config, create_html, create_skeleton)

    print("Sprint Start AI — 1C Python wrapper")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Project root: {project_root}")
    print(f"Source video: {video_path}")
    print(f"Input copy: {analysis_input}")
    print(f"Output: {output_dir}")
    print(f"Model: {model}")
    print(f"Meta: {meta}")
    print(f"Pose model: {pose_model}")
    print(f"Base config: {base_config}")
    print(f"Effective thresholds config from 1C: {config}")
    print(f"Thresholds from 1C: {json.dumps(thresholds_from_1c, ensure_ascii=False, sort_keys=True)}")
    print("CACHE_HIT=0")

    try:
        predictor_command = [
            sys.executable,
            str(predictor),
            "--input", str(analysis_input),
            "--model", str(model),
            "--meta", str(meta),
            "--pose-model", str(pose_model),
            "--config", str(config),
            "--output-json", str(raw_json),
            "--output-csv", str(raw_csv),
        ]
        if create_html:
            predictor_command += ["--output-html", str(raw_html)]
        else:
            predictor_command += ["--skip-html"]
        run_command(predictor_command, cwd=project_root, log_path=log_path, title="LSTM valid-class predictor")

        polished = polish_predictions(
            raw_json, polished_json, polished_csv, polished_html if create_html else None, config
        )

        try:
            from scripts.enrich_result_for_1c_v6 import create_1c_result
        except ModuleNotFoundError:
            from enrich_result_for_1c_v6 import create_1c_result

        create_1c_result(
            predictions_path=raw_json,
            recommendations_path=recommendations_catalog,
            out_path=result_1c_json,
            config_path=config,
            video_path=analysis_input,
            model_version=model.name,
        )

        skeleton_status = "skipped"
        skeleton_annotator = project_root / "scripts" / "annotate_skeletons_for_lstm_check_v1.py"
        if create_skeleton and skeleton_annotator.exists():
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
            "cache_hit": False,
            "cache_message": "Готового совместимого анализа не найдено; нейросеть выполнена заново.",
            "cache_signature": cache_signature,
            "project_root": str(project_root),
            "source_video": str(video_path),
            "requested_video": str(video_path),
            "input": str(analysis_input),
            "output_dir": str(output_dir),
            "model": str(model),
            "meta": str(meta),
            "pose_model": str(pose_model),
            "threshold_config": str(config),
            "threshold_source": "1C:ПараметрыНейросетевогоАнализа",
            "thresholds_from_1c": thresholds_from_1c,
            "analysis_options": {"create_html": bool(create_html), "create_skeleton": bool(create_skeleton)},
            "raw_json": str(raw_json),
            "raw_csv": str(raw_csv),
            "raw_html": str(raw_html) if create_html else "",
            "html_status": "created" if create_html else "skipped",
            "final_json": str(polished_json),
            "final_csv": str(polished_csv),
            "final_html": str(polished_html) if create_html else "",
            "result_1c_json": str(result_1c_json),
            "skeleton_dir": str(skeleton_dir) if skeleton_status == "created" else "",
            "skeleton_status": skeleton_status,
            "log": str(log_path),
            "items_count": polished.get("items_count"),
            "average_score": polished.get("average_score"),
        }
        save_json(result_paths_json, result_paths)

        print("DONE")
        print(f"FINAL_JSON={polished_json}")
        if create_html:
            print(f"FINAL_HTML={polished_html}")
        else:
            print("FINAL_HTML=SKIPPED")
        print(f"RESULT_1C_JSON={result_1c_json}")
        print(f"SKELETON_DIR={skeleton_dir if skeleton_status == 'created' else 'SKIPPED'}")
        print(f"RESULT_PATHS={result_paths_json}")

        # v50: HTML is generated only when enabled; it is never opened automatically.
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": str(exc),
            "cache_hit": False,
            "cache_signature": cache_signature,
            "project_root": str(project_root),
            "source_video": str(video_path),
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
