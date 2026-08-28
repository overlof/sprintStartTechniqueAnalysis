import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32)
]

def draw_skeleton(frame, landmarks, visibility_thr=0.35):
    h,w = frame.shape[:2]
    pts=[]
    for lm in landmarks:
        x=int(lm.x*w); y=int(lm.y*h); v=getattr(lm,'visibility',1.0)
        pts.append((x,y,v))
    for a,b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts) and pts[a][2] >= visibility_thr and pts[b][2] >= visibility_thr:
            cv2.line(frame, pts[a][:2], pts[b][:2], (0,220,0), 2)
    for i,(x,y,v) in enumerate(pts):
        if v >= visibility_thr:
            cv2.circle(frame, (x,y), 4, (0,255,0), -1)
    return frame

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--pose-model', default='models/pose_landmarker_full.task')
    ap.add_argument('--max-videos', type=int, default=20)
    ap.add_argument('--fps', type=int, default=18)
    args=ap.parse_args()
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    BaseOptions = mp_python.BaseOptions
    pose_model_path = Path(args.pose_model)
    if not pose_model_path.is_file():
        raise FileNotFoundError(f"PoseLandmarker model file not found: {pose_model_path}")
    pose_model_bytes = pose_model_path.read_bytes()
    if not pose_model_bytes:
        raise RuntimeError(f"PoseLandmarker model file is empty: {pose_model_path}")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=pose_model_bytes),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    in_path=Path(args.input)
    vids=sorted(list(in_path.rglob('*.mp4'))) if in_path.is_dir() else [in_path]
    vids=vids[:args.max_videos]
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for idx,vp in enumerate(vids,1):
        cap=cv2.VideoCapture(str(vp))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=cap.get(cv2.CAP_PROP_FPS) or args.fps
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        out_path=out_dir / f'{idx:03d}_{vp.stem}_skeleton.mp4'
        writer=cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width,height))
        frame_i=0; no_pose=0
        while True:
            ok, frame=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                res=landmarker.detect(mp_img)
                if res.pose_landmarks:
                    frame=draw_skeleton(frame, res.pose_landmarks[0])
                else:
                    no_pose += 1
                    cv2.putText(frame,'NO POSE',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            except Exception as e:
                no_pose += 1
                cv2.putText(frame,'POSE ERROR',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(frame, vp.name, (20,height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            writer.write(frame); frame_i += 1
        cap.release(); writer.release()
        print(f'[{idx}/{len(vids)}] {out_path} frames={frame_i} no_pose={no_pose}')
    landmarker.close()

if __name__=='__main__': main()
