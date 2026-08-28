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
- выбор модели, metadata и PoseLandmarker со стороны 1С;
- повторное использование ранее выполненного анализа при совпадении видео и параметров модели;
- принудительный повторный анализ через `--force-reanalyze`;
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
├── config/
│   ├── neural_valid_class_all_v1.json
│   ├── predict_valid_class_v1.json
│   └── recommendations_catalog_v6.json
├── data/
├── models/
│   ├── neural_lstm_valid_class_all_v1.keras
│   ├── neural_lstm_valid_class_all_v1_meta.json
│   └── pose_landmarker_full.task
├── outputs/
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
result_1c.json + result_paths.json
  ↓
1С:Предприятие
```

Вероятность технической ошибки показывает уверенность модели в наличии соответствующего признака и не является методической критичностью ошибки. Приоритет коррекции определяется в 1С после проверки тренером.

## Установка

```bash
git clone https://github.com/overlof/sprintStartTechniqueAnalysis.git
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

## Полный запуск для 1С

```bash
python scripts/run_from_1c_full_analysis.py \
  --project-root . \
  --video path/to/video.mp4 \
  --output-dir outputs/from_1c_python \
  --model models/neural_lstm_valid_class_all_v1.keras \
  --meta models/neural_lstm_valid_class_all_v1_meta.json \
  --pose-model models/pose_landmarker_full.task \
  --config config/predict_valid_class_v1.json
```

Дополнительные флаги:

- `--skip-skeleton` — не создавать видео со скелетом;
- `--skip-html` — не создавать HTML-отчёты;
- `--force-reanalyze` — игнорировать найденный кэш и повторно запустить нейросеть;
- `--open-html` — оставлен для совместимости, HTML автоматически не открывается.

## Повторное использование анализа

`run_from_1c_full_analysis.py` ищет ранее выполненный анализ и повторно использует его только при совпадении исходного видео и существенных параметров анализа, включая модель, metadata, PoseLandmarker и конфигурацию порогов. Для видео проверяются имя, размер и SHA-256. При совпадении возвращается `CACHE_HIT=1`, и TensorFlow/MediaPipe повторно не запускаются.

## Примечание о `outputs/`

Результаты запусков, временные видео, логи и сгенерированные отчёты не хранятся в репозитории. Они создаются локально во время работы программы и исключены через `.gitignore`.
