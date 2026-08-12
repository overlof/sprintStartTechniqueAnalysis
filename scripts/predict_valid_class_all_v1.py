import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mediapipe as mp


ERROR_META = {
    "body_rises_early": {
        "ru": "Ранний подъём корпуса",
        "short": "Корпус слишком рано поднимается вверх после выхода из старта.",
        "details": "В первых шагах важно сохранять наклон корпуса и направлять усилие назад-вниз. Ранний подъём корпуса уменьшает горизонтальное ускорение.",
        "recommendations": [
            "Дольше сохранять наклон корпуса в первых 2–3 шагах.",
            "Следить, чтобы движение было направлено вперёд, а не вверх.",
            "Использовать wall drive, falling start и короткие ускорения на 3–5 шагов."
        ]
    },
    "front_knee_bad": {
        "ru": "Неоптимальный угол переднего колена",
        "short": "Переднее колено в стартовой позиции может быть слишком закрыто или раскрыто.",
        "details": "Угол переднего колена влияет на направление и мощность первого толчка. Неверная постановка может задерживать выход или снижать стартовую мощность.",
        "recommendations": [
            "Проверить положение передней колодки и угол переднего колена в положении готовности.",
            "Сравнить стартовую позицию с боковым эталонным ракурсом.",
            "Подобрать расстояние передней колодки, при котором толчок получается наиболее мощным."
        ]
    },
    "rear_knee_bad": {
        "ru": "Неоптимальный угол заднего колена",
        "short": "Задняя нога может находиться в невыгодном положении для быстрого выноса.",
        "details": "Задняя нога должна быстро включаться в первый шаг. Неудачный угол может приводить к задержке выноса ноги или лишнему вертикальному движению.",
        "recommendations": [
            "Проверить положение задней колодки относительно передней.",
            "Следить, чтобы задняя нога быстро уходила вперёд после старта.",
            "Использовать старты на 1–2 шага с акцентом на быстрый вынос задней ноги."
        ]
    },
    "hip_position_bad": {
        "ru": "Неоптимальное положение таза",
        "short": "Таз может быть слишком низко или слишком высоко в стартовой позиции.",
        "details": "Положение таза влияет на углы ног и направление стартового усилия. Слишком низкий таз затрудняет выход, слишком высокий может нарушать устойчивость.",
        "recommendations": [
            "Проверить высоту таза в положении готовности.",
            "Сохранять устойчивое положение плеч, таза и стоп перед стартом.",
            "Выполнять старты из колодок с видеоконтролем бокового ракурса."
        ]
    },
    "first_step_bad": {
        "ru": "Ошибка первого шага",
        "short": "Первый шаг может быть слишком длинным, слишком коротким или направленным неэффективно.",
        "details": "Первый шаг должен помогать ускорению, а не тормозить движение. Частая проблема — overstride, когда стопа ставится слишком далеко перед центром массы.",
        "recommendations": [
            "Первый шаг делать активным, коротким и направленным назад-вниз.",
            "Избегать постановки стопы слишком далеко перед центром массы.",
            "Использовать упражнения на первые 2–3 шага из колодок, wall drive и sled push с небольшим сопротивлением."
        ]
    },
    "arms_bad": {
        "ru": "Ошибка работы рук",
        "short": "Работа рук может быть недостаточно активной или уходить через среднюю линию корпуса.",
        "details": "Руки помогают сохранять ритм и направление ускорения. Пересечение рук перед корпусом или слабая амплитуда ухудшают баланс.",
        "recommendations": [
            "Работать руками активно вперёд-назад, без пересечения средней линии корпуса.",
            "Следить за синхронизацией рук и ног в первых шагах.",
            "Добавить упражнения на работу рук сидя, стоя и в коротких ускорениях."
        ]
    }
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def esc(x):
    return html.escape("" if x is None else str(x))


def list_videos(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXTS])


def make_landmarker(model_path):
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return vision.PoseLandmarker.create_from_options(options)


def sample_frame_indices(total_frames, sequence_length):
    if total_frames <= 0:
        return [0] * sequence_length
    if total_frames == 1:
        return [0] * sequence_length
    return np.linspace(0, total_frames - 1, sequence_length).round().astype(int).tolist()


def landmarks_to_vec(result):
    if not result.pose_landmarks:
        return np.zeros((33, 4), dtype=np.float32), False, 0.0
    lms = result.pose_landmarks[0]
    arr = np.zeros((33, 4), dtype=np.float32)
    vis = []
    for i, lm in enumerate(lms[:33]):
        arr[i, 0] = float(lm.x)
        arr[i, 1] = float(lm.y)
        arr[i, 2] = float(lm.z)
        # Tasks NormalizedLandmark has visibility often; fallback to presence/1.0
        v = getattr(lm, "visibility", None)
        if v is None:
            v = getattr(lm, "presence", 1.0)
        arr[i, 3] = float(v if v is not None else 1.0)
        vis.append(arr[i, 3])
    return arr, True, float(np.mean(vis)) if vis else 0.0


def extract_sequence(video_path, landmarker, seq_len):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((seq_len, 132), dtype=np.float32), {
            "pose_visibility_mean": 0.0,
            "no_pose_ratio": 1.0,
            "frames_used": 0,
            "video_opened": False,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices(total_frames, seq_len)
    seq = []
    pose_ok = 0
    vis_values = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            seq.append(np.zeros(132, dtype=np.float32))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = landmarker.detect(mp_image)
            arr, found, vis = landmarks_to_vec(result)
        except Exception:
            arr, found, vis = np.zeros((33, 4), dtype=np.float32), False, 0.0
        if found:
            pose_ok += 1
            vis_values.append(vis)
        seq.append(arr.reshape(-1))
    cap.release()

    X = np.stack(seq, axis=0).astype(np.float32)
    no_pose_ratio = 1.0 - (pose_ok / max(1, seq_len))
    pose_visibility_mean = float(np.mean(vis_values)) if vis_values else 0.0
    return X, {
        "pose_visibility_mean": pose_visibility_mean,
        "no_pose_ratio": no_pose_ratio,
        "frames_used": seq_len,
        "video_opened": True,
    }


def normalize_X(X, meta):
    norm = meta.get("normalization") or {}
    mean = norm.get("mean")
    std = norm.get("std")
    if mean is None or std is None:
        return X
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def classify_error(prob, cfg):
    if prob >= cfg["strong_error_threshold"]:
        return "strong", "высокая уверенность"
    if prob >= cfg["visual_review_threshold"]:
        return "review", "проверить визуально"
    return "weak", "слабый сигнал"


def compute_score(probs, cfg):
    score = float(cfg.get("base_score", 100.0))
    penalties = cfg.get("penalties", {})
    for key, prob in probs.items():
        if key == "valid_start":
            continue
        if prob >= cfg["strong_error_threshold"]:
            # Smooth penalty: stronger probs receive larger penalty.
            strength = (prob - cfg["strong_error_threshold"]) / max(1e-6, 1.0 - cfg["strong_error_threshold"])
            score -= penalties.get(key, 8.0) * (0.65 + 0.35 * strength)
        elif prob >= cfg["visual_review_threshold"]:
            score -= penalties.get(key, 8.0) * 0.25
    return round(max(0.0, min(100.0, score)), 1)


def score_text(score):
    if score is None:
        return "оценка не сформирована"
    if score >= 85:
        return "хорошо"
    if score >= 70:
        return "удовлетворительно"
    return "требует доработки"


def score_class(score):
    if score is None:
        return "bad"
    if score >= 85:
        return "good"
    if score >= 70:
        return "mid"
    return "bad"


def analyze_video(video, model, meta, landmarker, cfg):
    X, pose_info = extract_sequence(video, landmarker, int(cfg["sequence_length"]))
    Xn = normalize_X(X[None, :, :], meta)
    pred = model.predict(Xn, verbose=0)[0]

    labels = meta.get("label_names") or meta.get("labels") or cfg["labels"]
    if len(labels) != len(pred):
        labels = cfg["labels"][:len(pred)]
    probs = {label: float(pred[i]) for i, label in enumerate(labels)}
    valid_prob = float(probs.get("valid_start", 1.0))

    flags = []
    input_status = "valid_start"
    if pose_info["pose_visibility_mean"] < cfg["pose_visibility_min"] or pose_info["no_pose_ratio"] > cfg["no_pose_ratio_max"]:
        input_status = "invalid_low_pose"
        flags.append("low_pose_quality")
    elif valid_prob < cfg["borderline_start_min"]:
        input_status = "invalid_start"
    elif valid_prob < cfg["valid_start_threshold"]:
        input_status = "borderline_start_review"

    strong, review, weak = [], [], []
    all_labels = []

    if input_status == "valid_start":
        for key, meta_err in ERROR_META.items():
            prob = float(probs.get(key, 0.0))
            status, severity = classify_error(prob, cfg)
            item = {
                "label": key,
                "ru_name": meta_err["ru"],
                "probability": prob,
                "probability_percent": round(prob * 100, 1),
                "status": status,
                "severity": severity,
                "short": meta_err["short"],
                "details": meta_err["details"],
                "recommendations": meta_err["recommendations"],
            }
            all_labels.append(item)
            if status == "strong":
                strong.append(item)
            elif status == "review":
                review.append(item)
            else:
                weak.append(item)
        score = compute_score(probs, cfg)
    else:
        # Keep raw probabilities for diagnostics, but do not interpret as sports errors.
        for key, meta_err in ERROR_META.items():
            prob = float(probs.get(key, 0.0))
            status, severity = classify_error(prob, cfg)
            all_labels.append({
                "label": key,
                "ru_name": meta_err["ru"],
                "probability": prob,
                "probability_percent": round(prob * 100, 1),
                "status": "diagnostic_only",
                "severity": "диагностический сигнал",
                "short": meta_err["short"],
                "details": meta_err["details"],
                "recommendations": meta_err["recommendations"],
            })
        score = None

    summary = []
    if input_status == "valid_start":
        if strong:
            summary.append("Нейросеть выявила вероятные технические ошибки: " + ", ".join(x["ru_name"] for x in strong) + ".")
        else:
            summary.append("Уверенных технических ошибок не выявлено.")
        if review:
            summary.append("Есть признаки, которые желательно проверить визуально: " + ", ".join(x["ru_name"] for x in review) + ".")
        summary.append(f"Вероятность, что видео является низким стартом: {valid_prob*100:.1f}%.")
    elif input_status == "borderline_start_review":
        summary.append(f"Видео похоже на низкий старт неуверенно: valid_start={valid_prob*100:.1f}%. Требуется визуальная проверка.")
    elif input_status == "invalid_low_pose":
        summary.append("Видео не принято к спортивной оценке из-за низкого качества распознавания позы.")
    else:
        summary.append(f"Видео не похоже на низкий старт: valid_start={valid_prob*100:.1f}%. Спортивная оценка не сформирована.")

    return {
        "filename": video.name,
        "path": str(video),
        "input_status": input_status,
        "valid_start_probability": valid_prob,
        "valid_start_probability_percent": round(valid_prob * 100, 1),
        "score": score,
        "score_text": score_text(score),
        "pose_visibility_mean": round(float(pose_info["pose_visibility_mean"]), 4),
        "no_pose_ratio": round(float(pose_info["no_pose_ratio"]), 4),
        "flags": flags,
        "probabilities": probs,
        "strong_errors": strong,
        "visual_review": review,
        "weak_signals": weak,
        "all_labels": all_labels,
        "summary": summary,
    }


def render_errors(items):
    if not items:
        return "<p class='muted'>Нет</p>"
    out = []
    for it in items:
        recs = "".join(f"<li>{esc(r)}</li>" for r in it.get("recommendations", []))
        out.append(f"""
        <div class='error'>
          <div class='error-title'>{esc(it['ru_name'])} <span>{it['probability_percent']}% — {esc(it['severity'])}</span></div>
          <p><b>Что означает:</b> {esc(it['short'])}</p>
          <p><b>Подробно:</b> {esc(it['details'])}</p>
          <p><b>Рекомендации:</b></p><ul>{recs}</ul>
        </div>
        """)
    return "\n".join(out)


def render_html(path, items, version):
    valid_scores = [x["score"] for x in items if x.get("score") is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else None
    avg_txt = "N/A" if avg is None else f"{avg:.1f}"

    cards = []
    for item in items:
        status_ru = {
            "valid_start": "Видео принято к спортивной оценке",
            "invalid_start": "Видео не является низким стартом",
            "borderline_start_review": "Видео требует визуальной проверки",
            "invalid_low_pose": "Видео не принято: низкое качество позы",
        }.get(item["input_status"], item["input_status"])
        score = item.get("score")
        score_html = "N/A" if score is None else str(score)
        probs_rows = []
        for lab in item.get("all_labels", []):
            probs_rows.append(f"<tr><td>{esc(lab['ru_name'])}</td><td>{lab['probability_percent']}%</td><td>{esc(lab['severity'])}</td></tr>")
        cards.append(f"""
        <section class='card'>
          <div class='top'><div><h2>{esc(item['filename'])}</h2><p class='muted'>{esc(status_ru)}</p></div><div class='score {score_class(score)}'>{score_html}</div></div>
          <div class='grid'>
            <div><b>Valid start:</b><br>{item['valid_start_probability_percent']}%</div>
            <div><b>Средняя видимость точек:</b><br>{item['pose_visibility_mean']}</div>
            <div><b>Кадры без позы:</b><br>{item['no_pose_ratio']}</div>
          </div>
          <h3>Краткий вывод</h3><ul>{''.join(f'<li>{esc(s)}</li>' for s in item['summary'])}</ul>
          <h3>Уверенные ошибки</h3>{render_errors(item['strong_errors'])}
          <h3>Проверить визуально</h3>{render_errors(item['visual_review'])}
          <h3>Вероятности по всем классам ошибок</h3>
          <table><thead><tr><th>Показатель</th><th>Вероятность</th><th>Интерпретация</th></tr></thead><tbody>{''.join(probs_rows)}</tbody></table>
        </section>
        """)

    html_text = f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<title>Sprint Start AI — valid_start + LSTM</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f5f7;color:#222;margin:0;padding:24px}}h1{{margin:0 0 8px}}.muted{{color:#666}}
.summary{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px;margin:22px 0}}.summary div,.card{{background:white;border-radius:14px;box-shadow:0 2px 10px #0001;padding:18px}}
.big{{font-size:30px;font-weight:bold;margin-top:6px}}.top{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}.score{{font-size:34px;font-weight:bold;border-radius:14px;padding:12px 18px;min-width:82px;text-align:center}}
.good{{background:#e7f7ec;color:#146b2e}}.mid{{background:#fff4d8;color:#805400}}.bad{{background:#ffe2e2;color:#9b1c1c}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:10px;background:#fafafa;border-radius:12px;padding:12px;margin:12px 0}}
.error{{border-left:4px solid #2f5bea;background:#f8faff;border-radius:10px;padding:12px;margin:10px 0}}.error-title{{font-weight:bold;font-size:17px}}.error-title span{{font-weight:normal;color:#666;font-size:14px;margin-left:8px}}
li{{margin:6px 0;line-height:1.4}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;border-bottom:1px solid #eee;padding:8px}}th{{background:#fafafa}}
</style></head><body>
<h1>Sprint Start AI — финальный отчёт valid_start + LSTM</h1>
<p class='muted'>Версия {esc(version)}. Модель предсказывает класс valid_start и 6 классов технических ошибок.</p>
<div class='summary'><div><div>Видео</div><div class='big'>{len(items)}</div></div><div><div>Средняя оценка</div><div class='big'>{avg_txt}</div></div><div><div>Модель</div><div class='big'>valid-class LSTM</div></div></div>
{''.join(cards)}
</body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html_text, encoding="utf-8")


def write_csv(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "input_status", "valid_start_probability", "score", "score_text", "pose_visibility_mean", "no_pose_ratio", "strong_errors", "visual_review"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in items:
            w.writerow({
                "filename": item["filename"],
                "input_status": item["input_status"],
                "valid_start_probability": item["valid_start_probability"],
                "score": item["score"],
                "score_text": item["score_text"],
                "pose_visibility_mean": item["pose_visibility_mean"],
                "no_pose_ratio": item["no_pose_ratio"],
                "strong_errors": "; ".join(x["ru_name"] for x in item.get("strong_errors", [])),
                "visual_review": "; ".join(x["ru_name"] for x in item.get("visual_review", [])),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="models/neural_lstm_valid_class_all_v1.keras")
    ap.add_argument("--meta", default="models/neural_lstm_valid_class_all_v1_meta.json")
    ap.add_argument("--pose-model", default="models/pose_landmarker_full.task")
    ap.add_argument("--config", default="config/predict_valid_class_v1.json")
    ap.add_argument("--output-json", default="outputs/final_predict_valid_class/reports/predictions_valid_class.json")
    ap.add_argument("--output-csv", default="outputs/final_predict_valid_class/metrics/predictions_valid_class.csv")
    ap.add_argument("--output-html", default="outputs/final_predict_valid_class/reports/index_valid_class.html")
    args = ap.parse_args()

    cfg = load_json(args.config)
    meta = load_json(args.meta)
    model = tf.keras.models.load_model(args.model)
    videos = list_videos(args.input)
    if not videos:
        raise SystemExit(f"No videos found in {args.input}")

    landmarker = make_landmarker(Path(args.pose_model))
    items = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        items.append(analyze_video(video, model, meta, landmarker, cfg))
    landmarker.close()

    out = {
        "version": cfg.get("version", "predict_valid_class_v1"),
        "model_path": str(args.model),
        "meta_path": str(args.meta),
        "items": items,
    }
    save_json(args.output_json, out)
    write_csv(args.output_csv, items)
    render_html(args.output_html, items, cfg.get("version", "predict_valid_class_v1"))
    print("Saved:")
    print("  JSON:", args.output_json)
    print("  CSV :", args.output_csv)
    print("  HTML:", args.output_html)


if __name__ == "__main__":
    main()
