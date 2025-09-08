# Traffic Flow Analyzer — Vehicle Counting in 3 Lanes (YOLOv5 + SORT)

This repository contains a Python script to detect, track, and count vehicles across three lanes in a traffic video. The script downloads the provided YouTube video, processes frames with a pretrained YOLOv5 (COCO) detector, tracks vehicles using a compact SORT implementation (Kalman + Hungarian), and outputs CSV, overlay video, and a summary.

## Features

- Downloads YouTube video (pytube)
- Vehicle detection using pretrained YOLOv5 (COCO)
- Tracking with SORT to avoid duplicate counting
- Lane definition: default = 3 equal vertical lanes (configurable)
- CSV output: `vehicle_id, lane_number, frame_count, first_timestamp, class`
- Overlay video with lane boundaries, bounding boxes and live counts
- Command-line options for device, fps, and frame skipping for speed

## Requirements

- Python 3.8+
- GPU recommended (CUDA + torch) for real-time performance, but CPU will work (slower)
- Disk space to store downloaded video and outputs

## Dependencies

Install dependencies with pip:

```bash
pip install torch torchvision opencv-python pytube tqdm pandas filterpy scipy
