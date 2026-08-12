import argparse, json, csv, os, random
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ["valid_start","body_rises_early","front_knee_bad","rear_knee_bad","hip_position_bad","first_step_bad","arms_bad"]
ERRORS = LABELS[1:]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_x(npz):
    candidates = []
    for k in npz.files:
        arr = npz[k]
        if np.issubdtype(arr.dtype, np.number) and arr.ndim in (3,4) and arr.shape[0] > 0:
            candidates.append((k, arr))
    # prefer X
    for k, arr in candidates:
        if k.lower() in ('x','sequences','keypoints','data'):
            return k, normalize_x(arr)
    if candidates:
        k, arr = candidates[0]
        return k, normalize_x(arr)
    raise ValueError('No numeric 3D/4D sequence array found in NPZ')


def normalize_x(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        n,t,a,b = arr.shape
        arr = arr.reshape(n, t, a*b)
    if arr.ndim != 3:
        raise ValueError(f'X must be 3D after normalization, got {arr.shape}')
    return arr.astype('float32')


def get_filenames(npz, n):
    for k in ['filename','filenames','video','videos','file','files']:
        if k in npz.files:
            vals = [Path(str(x)).name for x in npz[k].tolist()]
            if len(vals) == n:
                return vals
    return None


def labels_from_npz(npz, n, valid_start_value=None):
    if 'y_all' in npz.files:
        y = np.asarray(npz['y_all']).astype('float32')
        if y.shape == (n, len(LABELS)):
            return y
    if 'y_errors' in npz.files:
        yerr = np.asarray(npz['y_errors']).astype('float32')
        if yerr.shape[0] == n and yerr.shape[1] >= len(ERRORS):
            y = np.zeros((n, len(LABELS)), dtype='float32')
            y[:,0] = 1.0 if valid_start_value is None else float(valid_start_value)
            y[:,1:] = yerr[:, :len(ERRORS)]
            return y
    if 'y_valid_start' in npz.files:
        yv = np.asarray(npz['y_valid_start']).reshape(-1).astype('float32')
        if len(yv) == n:
            y = np.zeros((n, len(LABELS)), dtype='float32')
            y[:,0] = yv
            if 'y_errors' in npz.files:
                yerr = np.asarray(npz['y_errors']).astype('float32')
                if yerr.shape[0] == n:
                    y[:,1:] = yerr[:, :len(ERRORS)]
            return y
    if valid_start_value is not None:
        y = np.zeros((n, len(LABELS)), dtype='float32')
        y[:,0] = float(valid_start_value)
        return y
    return None


def load_labels_csv(path):
    df = pd.read_csv(path)
    if 'filename' not in df.columns:
        # try video_name
        for c in ['video','video_name','file_name','source_video']:
            if c in df.columns:
                df = df.rename(columns={c:'filename'})
                break
    if 'filename' not in df.columns:
        raise ValueError(f'{path}: no filename/video column')
    df['__fname'] = df['filename'].astype(str).map(lambda x: Path(x).name)
    return df


def y_from_csv_for_filenames(csv_path, filenames, default_valid_start=1):
    df = load_labels_csv(csv_path)
    by = {r['__fname']: r for _, r in df.iterrows()}
    rows = []
    missing = []
    for fn in filenames:
        key = Path(str(fn)).name
        r = by.get(key)
        if r is None:
            missing.append(key)
            rows.append(None)
        else:
            y = np.zeros((len(LABELS),), dtype='float32')
            y[0] = float(r.get('valid_start', default_valid_start)) if str(r.get('valid_start', '')).strip() != '' else default_valid_start
            for i, lab in enumerate(ERRORS, start=1):
                val = r.get(lab, 0)
                if str(val).strip() in ('?', ''):
                    val = 0
                y[i] = float(val)
            rows.append(y)
    if missing:
        print(f'WARNING: {csv_path}: missing labels for {len(missing)} sequences; they will be dropped')
    keep = [i for i,r in enumerate(rows) if r is not None]
    if not keep:
        raise ValueError(f'No matching labels between {csv_path} and sequences')
    return np.stack([rows[i] for i in keep]), keep, missing


def load_dataset(name, npz_path, labels_csv=None, valid_start_value=None):
    npz = np.load(npz_path, allow_pickle=True)
    x_key, X = find_x(npz)
    n = X.shape[0]
    filenames = get_filenames(npz, n)
    if labels_csv:
        if filenames is None:
            # assume CSV order if no filenames in NPZ
            df = load_labels_csv(labels_csv)
            df = df.reset_index(drop=True)
            if len(df) < n:
                raise ValueError(f'{labels_csv}: rows {len(df)} < sequences {n}, cannot align by order')
            filenames = df['__fname'].iloc[:n].tolist()
            y = np.zeros((n, len(LABELS)), dtype='float32')
            for idx, (_, r) in enumerate(df.iloc[:n].iterrows()):
                y[idx,0] = float(r.get('valid_start', 1)) if str(r.get('valid_start','')).strip() != '' else 1
                for j, lab in enumerate(ERRORS, start=1):
                    val = r.get(lab, 0)
                    if str(val).strip() in ('?', ''):
                        val = 0
                    y[idx,j] = float(val)
            keep = list(range(n))
            missing = []
        else:
            y, keep, missing = y_from_csv_for_filenames(labels_csv, filenames, default_valid_start=1 if valid_start_value is None else valid_start_value)
            X = X[keep]
            filenames = [filenames[i] for i in keep]
    else:
        y = labels_from_npz(npz, n, valid_start_value=valid_start_value)
        if y is None:
            raise ValueError(f'{name}: labels_csv not provided and no labels in NPZ')
        if filenames is None:
            filenames = [f'{name}_{i:04d}.mp4' for i in range(n)]
        keep = list(range(n)); missing = []
    return {
        'name': name, 'X': X.astype('float32'), 'y': y.astype('float32'),
        'filenames': filenames, 'x_key': x_key,
        'rows': int(X.shape[0]), 'shape': list(X.shape), 'labels_csv': labels_csv,
        'missing_labels': missing[:20], 'missing_count': len(missing)
    }


def standardize(X_train, datasets):
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True)
    std[std < 1e-6] = 1.0
    for d in datasets:
        d['X'] = ((d['X'] - mean) / std).astype('float32')
    return mean.reshape(-1).astype(float).tolist(), std.reshape(-1).astype(float).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/neural_valid_class_all_v1.json')
    ap.add_argument('--synthetic-sequences', required=True)
    ap.add_argument('--synthetic-labels', default=None)
    ap.add_argument('--real104-sequences', required=True)
    ap.add_argument('--real104-labels', required=True)
    ap.add_argument('--clean29-sequences', default=None)
    ap.add_argument('--clean29-labels', default=None)
    ap.add_argument('--invalid-sequences', required=True)
    ap.add_argument('--output-model', required=True)
    ap.add_argument('--output-meta', required=True)
    ap.add_argument('--report-json', required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed = int(cfg.get('training',{}).get('random_seed',42))
    random.seed(seed); np.random.seed(seed)

    datasets = []
    datasets.append(load_dataset('synthetic', args.synthetic_sequences, args.synthetic_labels, valid_start_value=1))
    datasets.append(load_dataset('real104', args.real104_sequences, args.real104_labels, valid_start_value=1))
    if args.clean29_sequences and args.clean29_labels and Path(args.clean29_sequences).exists() and Path(args.clean29_labels).exists():
        datasets.append(load_dataset('clean29', args.clean29_sequences, args.clean29_labels, valid_start_value=1))
    datasets.append(load_dataset('invalid', args.invalid_sequences, None, valid_start_value=0))

    X = np.concatenate([d['X'] for d in datasets], axis=0)
    y = np.concatenate([d['y'] for d in datasets], axis=0)
    src = []
    for d in datasets:
        src.extend([d['name']] * d['X'].shape[0])
    src = np.array(src)

    # shuffle split
    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_frac = float(cfg.get('training',{}).get('validation_split', 0.2))
    n_val = max(1, int(round(len(idx)*val_frac)))
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    X_train = X[tr_idx]; y_train = y[tr_idx]
    X_val = X[val_idx]; y_val = y[val_idx]

    mean, std = standardize(X_train, [{'X': X_train}, {'X': X_val}])
    # standardize returned copies not assigned due dict local: do manually
    mean_arr = np.array(mean, dtype='float32').reshape(1,1,-1)
    std_arr = np.array(std, dtype='float32').reshape(1,1,-1)
    X_train = ((X_train - mean_arr) / std_arr).astype('float32')
    X_val = ((X_val - mean_arr) / std_arr).astype('float32')

    import tensorflow as tf
    tf.random.set_seed(seed)
    inp = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(inp)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation='sigmoid', name='labels')(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg.get('training',{}).get('learning_rate',0.001))),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True), tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(cfg.get('training',{}).get('early_stopping_patience',12)), restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=int(cfg.get('training',{}).get('reduce_lr_patience',5)), factor=0.5, min_lr=1e-5)
    ]
    hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)
    meta = {
        'version': cfg.get('version','valid_class_all_v1'),
        'model_type': 'Bidirectional LSTM multi-label classifier with valid_start class',
        'input_shape': [int(X_train.shape[1]), int(X_train.shape[2])],
        'label_names': LABELS,
        'error_label_names': ERRORS,
        'thresholds': cfg.get('thresholds',{}),
        'normalization': {'mean': mean, 'std': std},
        'sources': [{k:v for k,v in d.items() if k not in ('X','y','filenames')} for d in datasets]
    }
    save_json(args.output_meta, meta)
    eval_vals = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    report = {
        'version': cfg.get('version','valid_class_all_v1'),
        'training_rows': int(X_train.shape[0]),
        'validation_rows': int(X_val.shape[0]),
        'sources': meta['sources'],
        'source_counts_total': {name:int((src==name).sum()) for name in sorted(set(src.tolist()))},
        'label_positive_counts_total': {lab:int(y[:,i].sum()) for i, lab in enumerate(LABELS)},
        'final_validation_metrics': {k:float(v) for k,v in eval_vals.items()},
        'history': {k:[float(x) for x in v] for k,v in hist.history.items()},
        'model_path': args.output_model,
        'meta_path': args.output_meta
    }
    save_json(args.report_json, report)
    print('TRAINING COMPLETE')
    print(json.dumps(report['source_counts_total'], ensure_ascii=False, indent=2))
    print(json.dumps(report['final_validation_metrics'], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
