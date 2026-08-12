# Sprint Start Technique Analysis

Система анализа техники низкого старта спринтера по видео с использованием MediaPipe Pose и двунаправленной LSTM-модели. Проект извлекает позу спортсмена, формирует последовательность признаков, определяет корректность старта и технические ошибки, формирует отчёт и результат для загрузки в 1С:Предприятие.

## Возможности

- извлечение 33 ключевых точек MediaPipe Pose из видео;
- формирование последовательности фиксированной длины `72 x 132`;
- классификация корректности низкого старта (`valid_start`);
- многометочная классификация технических ошибок;
- расчёт итоговой оценки техники;
- генерация JSON, CSV и HTML-отчётов;
- формирование `result_1c.json` для интеграции с 1С;
- построение видео со скелетом спортсмена;
- обучение BiLSTM-модели на подготовленных наборах данных.

## Распознаваемые классы

- `valid_start` — в видео присутствует корректная фаза низкого старта;
- `body_rises_early` — слишком ранний подъём корпуса;
- `front_knee_bad` — ошибка положения переднего колена;
- `rear_knee_bad` — ошибка положения заднего колена;
- `hip_position_bad` — некорректное положение таза;
- `first_step_bad` — ошибка первого шага;
- `arms_bad` — ошибка работы рук.

## Структура проекта

```text
sprintStartTechniqueAnalysis/
├── config/                         # конфигурация обучения, инференса и рекомендаций
│   ├── neural_valid_class_all_v1.json
│   ├── predict_valid_class_v1.json
│   └── recommendations_catalog_v6.json
├── data/                           # подготовленные обучающие выборки
│   ├── clean29_labels/
│   ├── clean29_sequences/
│   ├── invalid_start_sequences/
│   ├── real_104_labels/
│   ├── real_104_sequences/
│   └── synthetic_v12/
├── models/                         # обученная BiLSTM и MediaPipe PoseLandmarker
│   ├── neural_lstm_valid_class_all_v1.keras
│   ├── neural_lstm_valid_class_all_v1_meta.json
│   └── pose_landmarker_full.task
├── outputs/                        # генерируемые результаты анализа
├── scripts/
│   ├── annotate_skeletons_for_lstm_check_v1.py
│   ├── enrich_result_for_1c_v6.py
│   ├── extract_sequences_final_v4_fixed_shape.py
│   ├── predict_valid_class_all_v1.py
│   ├── run_from_1c_full_analysis.py
│   └── train_lstm_valid_class_all_v1.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Общая схема работы

```text
Видео
  ↓
MediaPipe PoseLandmarker
  ↓
33 landmarks × (x, y, z, visibility)
  ↓
последовательность 72 × 132
  ↓
Bidirectional LSTM
  ↓
valid_start + вероятности технических ошибок
  ↓
JSON / CSV / HTML
  ↓
result_1c.json
  ↓
1С:Предприятие
```

## Установка

```bash
git clone <repository-url>
cd sprintStartTechniqueAnalysis
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Анализ одного видео

```bash
python scripts/predict_valid_class_all_v1.py \
  --input path/to/video.mp4 \
  --model models/neural_lstm_valid_class_all_v1.keras \
  --meta models/neural_lstm_valid_class_all_v1_meta.json \
  --pose-model models/pose_landmarker_full.task \
  --config config/predict_valid_class_v1.json
```

По умолчанию результаты сохраняются в `outputs/final_predict_valid_class/`.

## Полный запуск для 1С

```bash
python scripts/run_from_1c_full_analysis.py \
  --project-root . \
  --video path/to/video.mp4
```

Скрипт выполняет полный конвейер анализа, создаёт отчёты и подготавливает результат для дальнейшей загрузки в 1С.

## Формирование результата для 1С отдельно

```bash
python scripts/enrich_result_for_1c_v6.py \
  --predictions outputs/final_predict_valid_class/reports/predictions_valid_class.json \
  --recommendations config/recommendations_catalog_v6.json \
  --out outputs/result_1c.json \
  --video path/to/video.mp4
```

## Извлечение последовательностей из видео

```bash
python scripts/extract_sequences_final_v4_fixed_shape.py \
  --input path/to/videos \
  --output data/sequences.npz \
  --model models/pose_landmarker_full.task
```

## Обучение модели

```bash
python scripts/train_lstm_valid_class_all_v1.py \
  --synthetic-sequences data/synthetic_v12/synthetic_sequences_v12.npz \
  --synthetic-labels data/synthetic_v12/synthetic_labels.csv \
  --real104-sequences data/real_104_sequences/real104_sequences_final.npz \
  --real104-labels data/real_104_labels/real104_trimmed_labels_for_training_v13.csv \
  --clean29-sequences data/clean29_sequences/clean29_sequences_final.npz \
  --clean29-labels data/clean29_labels/clean29_labels_for_training.csv \
  --invalid-sequences data/invalid_start_sequences/invalid_sequences_final.npz \
  --output-model models/neural_lstm_valid_class_all_v1.keras \
  --output-meta models/neural_lstm_valid_class_all_v1_meta.json \
  --report-json outputs/training_report.json
```

## Примечание о `outputs/`

Результаты запусков, временные видео, логи и сгенерированные отчёты специально не хранятся в репозитории. Они создаются локально во время работы программы и исключены через `.gitignore`.
