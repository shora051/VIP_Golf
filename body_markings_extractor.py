import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# This project uses a newer MediaPipe Python package that provides the Tasks API
# (not `mediapipe.solutions`). We use PoseLandmarker with an external `.task` model.

# MediaPipe Pose landmark indices we care about (PoseLandmarker numbering):
# shoulders: 11 (left_shoulder), 12 (right_shoulder)
# elbows:    13 (left_elbow), 14 (right_elbow)
# hips:      23 (left_hip), 24 (right_hip)
# wrists:    15 (left_wrist), 16 (right_wrist)
LANDMARK_IDS = [11, 12, 13, 14, 23, 24, 15, 16]


@dataclass
class Node2D:
    x_px: float
    y_px: float
    z: float
    visibility: float


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def spine_angle_2d_abs_deg(
    hip_mid: Tuple[float, float],
    shoulder_mid: Tuple[float, float],
) -> float:
    """
    Computes absolute spine angle in degrees (0..180).

    We define the spine vector in image-plane pixels as:
      v = shoulder_mid - hip_mid
    and compare it to the "vertical up" axis (0, -1) in image coordinates.

    Because image y increases downward, the "up" axis is (0, -1).
    """
    sx, sy = shoulder_mid
    hx, hy = hip_mid
    vx = sx - hx
    vy = sy - hy

    v_norm = np.hypot(vx, vy)
    if v_norm < 1e-6:
        return float("nan")

    vertical_up = np.array([0.0, -1.0], dtype=np.float64)
    v_vec = np.array([vx, vy], dtype=np.float64)
    cos_theta = float(np.dot(v_vec, vertical_up) / (np.linalg.norm(v_vec) * 1.0))
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


def line_angle_2d_signed_deg(
    left_xy: Tuple[float, float],
    right_xy: Tuple[float, float],
) -> float:
    """
    Signed 2D line angle (degrees) relative to the image's +x axis.

    - Uses vector from left point to right point: v = right - left
    - Image coordinates: x increases right, y increases downward.
    - atan2(dy, dx) gives a signed result in (-180, 180] degrees.
    """
    lx, ly = left_xy
    rx, ry = right_xy
    dx = rx - lx
    dy = ry - ly
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, dx)))


def elbow_angle_2d_abs_deg(
    shoulder_xy: Tuple[float, float],
    elbow_xy: Tuple[float, float],
    wrist_xy: Tuple[float, float],
) -> float:
    """
    Absolute elbow joint angle in degrees (0..180) in image-plane using (x,y).
    Angle is computed at the elbow between vectors:
      v1 = shoulder - elbow
      v2 = wrist - elbow
    """
    sx, sy = shoulder_xy
    ex, ey = elbow_xy
    wx, wy = wrist_xy
    v1x = sx - ex
    v1y = sy - ey
    v2x = wx - ex
    v2y = wy - ey
    n1 = np.hypot(v1x, v1y)
    n2 = np.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_theta = float((v1x * v2x + v1y * v2y) / (n1 * n2))
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def landmark_to_node2d(lm, w: int, h: int) -> Node2D:
    # MediaPipe pose landmark coordinates are normalized to [0..1] for x,y relative to image size.
    return Node2D(
        x_px=float(lm.x * w),
        y_px=float(lm.y * h),
        z=float(lm.z),  # z is already normalized relative to person scale (not in pixels)
        visibility=float(getattr(lm, "visibility", 1.0)),
    )


def draw_debug(
    frame_bgr: np.ndarray,
    nodes: Dict[int, Node2D],
    spine_angle_deg: Optional[float],
    spine_valid: bool,
    shoulder_line_angle_deg: Optional[float],
    hip_line_angle_deg: Optional[float],
    xfactor_proxy_deg: Optional[float],
    lines_valid: bool,
    left_elbow_angle_deg: Optional[float],
    right_elbow_angle_deg: Optional[float],
    arms_valid: bool,
):
    # Draw midline and key pairs for quick visual sanity checks.
    h, w = frame_bgr.shape[:2]

    # Convenience helpers
    def pt(node_id: int) -> Optional[Tuple[int, int]]:
        if node_id not in nodes:
            return None
        n = nodes[node_id]
        if np.isnan(n.x_px) or np.isnan(n.y_px):
            return None
        return int(round(n.x_px)), int(round(n.y_px))

    # Shoulders midline
    ls = pt(11)
    rs = pt(12)
    le = pt(13)
    re = pt(14)
    lh = pt(23)
    rh = pt(24)
    lw = pt(15)
    rw = pt(16)

    color = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)

    # Key points
    for p in [ls, rs, le, re, lh, rh, lw, rw]:
        if p is not None:
            cv2.circle(frame_bgr, p, 5, yellow, -1)

    # Left-right shoulder and hip lines
    if ls is not None and rs is not None:
        cv2.line(frame_bgr, ls, rs, color, 2)
    if lh is not None and rh is not None:
        cv2.line(frame_bgr, lh, rh, color, 2)

    # Wrist positions (just points and a line between wrists if both visible)
    if lw is not None and rw is not None:
        cv2.line(frame_bgr, lw, rw, blue, 2)

    # Arm skeleton (optional): shoulder-elbow-wrist lines
    if ls is not None and le is not None:
        cv2.line(frame_bgr, ls, le, (0, 128, 255), 2)
    if le is not None and lw is not None:
        cv2.line(frame_bgr, le, lw, (0, 128, 255), 2)
    if rs is not None and re is not None:
        cv2.line(frame_bgr, rs, re, (255, 128, 0), 2)
    if re is not None and rw is not None:
        cv2.line(frame_bgr, re, rw, (255, 128, 0), 2)

    # Spine midline visualization
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        sm = (int(round(shoulder_mid[0])), int(round(shoulder_mid[1])))
        hm = (int(round(hip_mid[0])), int(round(hip_mid[1])))
        cv2.line(frame_bgr, hm, sm, red, 3)

    # Text overlay
    if spine_valid and spine_angle_deg is not None and not np.isnan(spine_angle_deg):
        txt = f"SpineAngle2DAbs: {spine_angle_deg:.1f} deg"
        cv2.putText(frame_bgr, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, red, 2, cv2.LINE_AA)
    else:
        cv2.putText(
            frame_bgr,
            "SpineAngle2DAbs: N/A (low vis)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )

    if (
        lines_valid
        and shoulder_line_angle_deg is not None
        and hip_line_angle_deg is not None
        and xfactor_proxy_deg is not None
        and not np.isnan(shoulder_line_angle_deg)
        and not np.isnan(hip_line_angle_deg)
        and not np.isnan(xfactor_proxy_deg)
    ):
        txt2 = (
            f"Shoulder2DAngle: {shoulder_line_angle_deg:.1f}  "
            f"Hip2DAngle: {hip_line_angle_deg:.1f}  "
            f"Xfactor: {xfactor_proxy_deg:.1f}"
        )
        cv2.putText(frame_bgr, txt2, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2, cv2.LINE_AA)
    else:
        cv2.putText(
            frame_bgr,
            "Line angles: N/A (low vis)",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )

    def fmt_angle(angle: Optional[float]) -> str:
        if angle is None or np.isnan(angle):
            return "N/A"
        return f"{angle:.1f}"

    txt3 = f"Elbow2DAngle: L {fmt_angle(left_elbow_angle_deg)}  R {fmt_angle(right_elbow_angle_deg)}"
    cv2.putText(
        frame_bgr,
        txt3,
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .mp4 or .mov input video")
    ap.add_argument("--output_dir", required=True, help="Folder to write CSV and debug video")
    ap.add_argument("--visibility_threshold", type=float, default=0.5, help="Min landmark visibility to accept")
    ap.add_argument(
        "--model_path",
        default="",
        help="Optional path to pose landmarker model (.task). If not provided, model will be auto-downloaded.",
    )
    ap.add_argument("--max_frames", type=int, default=-1, help="Optional limit for debugging")
    args = ap.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
        "float16/latest/pose_landmarker_full.task"
    )
    if args.model_path:
        model_path = args.model_path
    else:
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "pose_landmarker_full.task")
        if not os.path.exists(model_path):
            # Lazy download: keeps the repo lightweight.
            import urllib.request

            print(f"Downloading model to: {model_path}")
            urllib.request.urlretrieve(model_url, model_path)

    csv_path = os.path.join(output_dir, "body_landmarks_timeseries.csv")
    debug_video_path = os.path.join(output_dir, "debug_body_landmarks.mp4")
    angles_csv_path = os.path.join(output_dir, "body_angles_timeseries.csv")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0  # fallback

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec choice: mp4v works in most local environments.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {debug_video_path}")

    # MediaPipe Tasks PoseLandmarker setup.
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe.tasks.python.vision.core.image import Image as MPImage
    from mediapipe.tasks.python.vision.pose_landmarker import (
        PoseLandmarker,
        PoseLandmarkerOptions,
    )

    base_options = BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    with open(csv_path, "w", newline="", encoding="utf-8") as f, open(
        angles_csv_path, "w", newline="", encoding="utf-8"
    ) as f_angles:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame_index", "timestamp", "landmark_id", "x", "y", "z", "visibility"])
        writer_angles = csv.writer(f_angles)
        writer_angles.writerow(
            [
                "frame_index",
                "timestamp",
                "shoulder_line_angle_2d_signed_deg",
                "hip_line_angle_2d_signed_deg",
                "xfactor_proxy_deg",
                "valid",
                "left_elbow_angle_2d_abs_deg",
                "right_elbow_angle_2d_abs_deg",
            ]
        )

        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            timestamp_sec = frame_index / fps
            timestamp_ms = int(round(timestamp_sec * 1000.0))

            # MediaPipe Tasks PoseLandmarker expects an Image container.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)
            res = landmarker.detect_for_video(mp_image, timestamp_ms)

            nodes: Dict[int, Node2D] = {}
            spine_valid = False
            spine_angle_deg: Optional[float] = None
            shoulder_line_angle_deg: Optional[float] = None
            hip_line_angle_deg: Optional[float] = None
            xfactor_proxy_deg: Optional[float] = None
            lines_valid = False
            left_elbow_angle_deg: Optional[float] = None
            right_elbow_angle_deg: Optional[float] = None
            arms_valid = False

            if res.pose_landmarks:
                # With `num_poses=1`, the first element is the single pose.
                landmarks = res.pose_landmarks[0]

                # Extract targeted nodes and write to CSV regardless of validity.
                # PoseLandmarker landmarks are normalized to [0..1] for x/y.
                for lm_id in LANDMARK_IDS:
                    lm = landmarks[lm_id]
                    visibility = float(lm.visibility) if lm.visibility is not None else 1.0
                    node = Node2D(
                        x_px=float(lm.x * frame_w),
                        y_px=float(lm.y * frame_h),
                        z=float(lm.z) if lm.z is not None else 0.0,
                        visibility=visibility,
                    )
                    nodes[lm_id] = node
                    writer_csv.writerow(
                        [
                            frame_index,
                            f"{timestamp_sec:.6f}",
                            lm_id,
                            f"{node.x_px:.3f}",
                            f"{node.y_px:.3f}",
                            f"{node.z:.6f}",
                            f"{node.visibility:.3f}",
                        ]
                    )

                # Compute 2D spine angle using midpoints if both sides are visible enough.
                # Shoulders: 11,12. Hips: 23,24.
                if (
                    nodes[11].visibility >= args.visibility_threshold
                    and nodes[12].visibility >= args.visibility_threshold
                    and nodes[23].visibility >= args.visibility_threshold
                    and nodes[24].visibility >= args.visibility_threshold
                ):
                    shoulder_mid = (
                        (nodes[11].x_px + nodes[12].x_px) / 2.0,
                        (nodes[11].y_px + nodes[12].y_px) / 2.0,
                    )
                    hip_mid = (
                        (nodes[23].x_px + nodes[24].x_px) / 2.0,
                        (nodes[23].y_px + nodes[24].y_px) / 2.0,
                    )
                    spine_angle_deg = spine_angle_2d_abs_deg(
                        hip_mid=hip_mid, shoulder_mid=shoulder_mid
                    )
                    spine_valid = not np.isnan(spine_angle_deg)

                # Shoulder/hip line angles (2D) and X-factor proxy (shoulder - hip).
                if (
                    nodes[11].visibility >= args.visibility_threshold
                    and nodes[12].visibility >= args.visibility_threshold
                    and nodes[23].visibility >= args.visibility_threshold
                    and nodes[24].visibility >= args.visibility_threshold
                ):
                    shoulder_line_angle_deg = line_angle_2d_signed_deg(
                        left_xy=(nodes[11].x_px, nodes[11].y_px),
                        right_xy=(nodes[12].x_px, nodes[12].y_px),
                    )
                    hip_line_angle_deg = line_angle_2d_signed_deg(
                        left_xy=(nodes[23].x_px, nodes[23].y_px),
                        right_xy=(nodes[24].x_px, nodes[24].y_px),
                    )
                    xfactor_proxy_deg = float(shoulder_line_angle_deg - hip_line_angle_deg)
                    lines_valid = True

                # Elbow joint angles (2D absolute) from shoulder-elbow-wrist.
                if (
                    nodes[11].visibility >= args.visibility_threshold
                    and nodes[13].visibility >= args.visibility_threshold
                    and nodes[15].visibility >= args.visibility_threshold
                ):
                    left_elbow_angle_deg = elbow_angle_2d_abs_deg(
                        shoulder_xy=(nodes[11].x_px, nodes[11].y_px),
                        elbow_xy=(nodes[13].x_px, nodes[13].y_px),
                        wrist_xy=(nodes[15].x_px, nodes[15].y_px),
                    )

                if (
                    nodes[12].visibility >= args.visibility_threshold
                    and nodes[14].visibility >= args.visibility_threshold
                    and nodes[16].visibility >= args.visibility_threshold
                ):
                    right_elbow_angle_deg = elbow_angle_2d_abs_deg(
                        shoulder_xy=(nodes[12].x_px, nodes[12].y_px),
                        elbow_xy=(nodes[14].x_px, nodes[14].y_px),
                        wrist_xy=(nodes[16].x_px, nodes[16].y_px),
                    )

                if (
                    left_elbow_angle_deg is not None
                    and right_elbow_angle_deg is not None
                    and not np.isnan(left_elbow_angle_deg)
                    and not np.isnan(right_elbow_angle_deg)
                ):
                    arms_valid = True

            # Always write one row per frame for angles CSV.
            writer_angles.writerow(
                [
                    frame_index,
                    f"{timestamp_sec:.6f}",
                    "" if shoulder_line_angle_deg is None else f"{shoulder_line_angle_deg:.6f}",
                    "" if hip_line_angle_deg is None else f"{hip_line_angle_deg:.6f}",
                    "" if xfactor_proxy_deg is None else f"{xfactor_proxy_deg:.6f}",
                    int(lines_valid),
                    "" if left_elbow_angle_deg is None else f"{left_elbow_angle_deg:.6f}",
                    "" if right_elbow_angle_deg is None else f"{right_elbow_angle_deg:.6f}",
                ]
            )

            # Debug overlay video
            draw_debug(
                frame_bgr,
                nodes,
                spine_angle_deg=spine_angle_deg,
                spine_valid=spine_valid,
                shoulder_line_angle_deg=shoulder_line_angle_deg,
                hip_line_angle_deg=hip_line_angle_deg,
                xfactor_proxy_deg=xfactor_proxy_deg,
                lines_valid=lines_valid,
                left_elbow_angle_deg=left_elbow_angle_deg,
                right_elbow_angle_deg=right_elbow_angle_deg,
                arms_valid=arms_valid,
            )
            writer.write(frame_bgr)

            frame_index += 1

    cap.release()
    writer.release()
    landmarker.close()

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote CSV: {angles_csv_path}")
    print(f"Wrote debug video: {debug_video_path}")


if __name__ == "__main__":
    main()

