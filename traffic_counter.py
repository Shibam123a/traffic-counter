#!/usr/bin/env python3
import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import yt_dlp


VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck


def download_youtube_video(youtube_url: str, output_path: str = "traffic_input.mp4") -> str:
    """Download a YouTube video with yt-dlp (cached if already present)."""
    if os.path.exists(output_path):
        print(f"[INFO] Using cached video: {output_path}")
        return output_path
    print("[INFO] Downloading video from YouTube...")
    ydl_opts = {
        "format": "mp4",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


def define_lanes(frame_width: int, num_lanes: int = 3, mode: str = "equal"):
    """Return lane x-bounds: list of (x1, x2) for each lane."""
    if mode == "equal":
        w = frame_width // num_lanes
        return [(i * w, (i + 1) * w if i < num_lanes - 1 else frame_width) for i in range(num_lanes)]
    raise ValueError("Only 'equal' lane mode implemented.")


def which_lane(x_center: int, lanes):
    """Return 1-based lane index for a given x center."""
    for i, (x1, x2) in enumerate(lanes, start=1):
        if x1 <= x_center < x2:
            return i
    return None


def parse_args():
    p = argparse.ArgumentParser("Lane-based vehicle counter with YOLO + tracking")
    p.add_argument("--youtube_url", required=True, help="YouTube video URL")
    p.add_argument("--input_video", default=None, help="Optional local video path (skips download)")
    p.add_argument("--output_video", default="overlay_output.mp4", help="Output video with overlay")
    p.add_argument("--output_csv", default="results.csv", help="Per-vehicle crossing log")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    p.add_argument("--line_pos", type=float, default=0.5, help="Horizontal counting line y as a fraction of height (0-1)")
    p.add_argument("--display", action="store_true", help="Show live window")
    p.add_argument("--resize_width", type=int, default=0, help="Optional processing width (keeps aspect). 0 = original")
    p.add_argument("--skip", type=int, default=0, help="Process every (skip+1)th frame for speed")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Video I/O ---
    video_path = args.input_video or download_youtube_video(args.youtube_url)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Optional resize (speeds up processing)
    if args.resize_width and args.resize_width < src_w:
        scale = args.resize_width / src_w
        W, H = args.resize_width, int(src_h * scale)
    else:
        W, H = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (W, H))

    # --- Model + Tracker ---
    print("[INFO] Loading YOLOv8 with tracking...")
    model = YOLO("yolov8n.pt")  # small & fast; COCO-pretrained
    # We'll use the built-in tracker (ByteTrack) to get stable IDs.
    # persist=True keeps tracks across frames.

    # --- Lanes + counting line ---
    lanes = define_lanes(W, num_lanes=3, mode="equal")
    line_y = int(H * float(np.clip(args.line_pos, 0.0, 1.0)))  # counting line
    lane_counts = defaultdict(int)

    # Track state: last Y and whether already counted in that lane
    last_center_y = {}           # id -> last y center
    counted_in_lane = {}         # id -> set of lanes already credited

    # Logging
    records = []  # [track_id, lane, frame_idx, epoch_time]

    frame_idx = 0
    t0 = time.time()

    print("[INFO] Starting...")
    # Stream results frame by frame for custom overlay
    stream = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=args.conf,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,
        imgsz=max(640, W if W < 640 else W),  # keep default-ish; YOLO will letterbox
    )

    for res in stream:
        # Optionally drop frames for speed
        if args.skip and (frame_idx % (args.skip + 1) != 0):
            frame_idx += 1
            continue

        frame = res.orig_img  # BGR (original shape)
        if frame is None:
            break

        # Resize for processing/overlay
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

        # Draw lanes and counting line
        lane_overlay = frame.copy()
        # lane bands
        for i, (x1, x2) in enumerate(lanes, start=1):
            cv2.rectangle(lane_overlay, (x1, 0), (x2, H), (60, 60, 60), -1)
            cv2.putText(frame, f"Lane {i}: {lane_counts[i]}", (x1 + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.addWeighted(lane_overlay, 0.07, frame, 0.93, 0, frame)
        # counting line
        cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)

        # Parse tracked boxes
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None
            clss = res.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                if ids is None:
                    # If tracker produced no IDs, skip counting to avoid duplicates.
                    continue
                track_id = int(ids[i])
                x1, y1, x2, y2 = boxes[i].astype(int)
                cls_id = int(clss[i])

                # center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # which lane?
                lane = which_lane(cx, lanes)
                if lane is None:
                    continue

                # init sets
                if track_id not in counted_in_lane:
                    counted_in_lane[track_id] = set()

                # Directional crossing: count when moving DOWN across the line
                prev_y = last_center_y.get(track_id, None)
                crossed_down = prev_y is not None and prev_y < line_y <= cy

                if crossed_down and (lane not in counted_in_lane[track_id]):
                    lane_counts[lane] += 1
                    counted_in_lane[track_id].add(lane)
                    records.append([track_id, lane, frame_idx, time.time()])

                # update last y
                last_center_y[track_id] = cy

                # draw box/center/id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {track_id}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # write and (optionally) show
        out.write(frame)
        if args.display:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()

    # Save CSV
    df = pd.DataFrame(records, columns=["TrackID", "Lane", "Frame", "EpochTime"])
    df.to_csv(args.output_csv, index=False)

    print("\n=== Vehicle Count Summary ===")
    for lane in range(1, 4):
        print(f"Lane {lane}: {lane_counts[lane]} vehicles")
    print(f"\nVideo saved: {args.output_video}")
    print(f"CSV saved:   {args.output_csv}")
    print(f"Runtime:     {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
