"""Independent synthetic video demonstration: PhysioTrack MouthMovement/Velocity.

Install as validation/mouth_movement_velocity/mouth_movement_velocity_synthetic_demo.py
Input:   validation/synthetic_media/mouth_movement_velocity_media/ (recursively)
Output:  validation/mouth_movement_velocity/results/synthetic_demo/ (script-owned)

REAL Face detector -> FaceAnalysis + OC-SORT FaceTracker -> FaceLandmarks ->
MouthOpenness -> MouthMovement (mouth movement and velocity). Uses the decoded
video's actual FPS. Each source video gets a fresh tracker and temporal state.

Outputs native, unmodified, frame/track-level measurements, with independent
geometry and temporal arithmetic checks; annotated videos with a dedicated side panel; timeline plots;
illustrative still frames/contact sheet; CSV and identical-data JSON/TXT reports.
All outputs are verified in staging before publication, with rollback of old
synthetic outputs. Existing FELT/RAVDESS benchmarks and isolated component
execution outputs are never read or modified.

Synthetic motion descriptions are NOT frame-level ground truth: this is NOT
an accuracy, tracking-ID accuracy, temporal robustness, or detection benchmark.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPO_ROOT = VALIDATION_DIR.parent
SOURCE_ROOT = REPO_ROOT / 'src' / 'physiotrack'
INPUT_DIR = VALIDATION_DIR / 'synthetic_media' / 'mouth_movement_velocity_media'
RESULTS_DIR = SCRIPT_DIR / 'results'
FINAL_DIR = RESULTS_DIR / 'synthetic_demo'
EXTENSIONS = {'.mp4', '.m4v', '.mov', '.avi', '.mkv', '.webm', '.mpg',
              '.mpeg', '.wmv', '.ts', '.mts', '.m2ts', '.3gp', '.ogv'}
DEVICE = 'cpu'
TRACKER_TYPE = 'ocsort'
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.45
MAX_DETECTIONS = 10000
EXPECTED_LANDMARKS = 478
EXPECTED_LANDMARK_MODEL_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
GEOMETRY_TOL = 1e-12
MOVEMENT_TOL = 1e-12
VELOCITY_TOL = 1e-10
OUTPUT_CODECS = ('mp4v', 'avc1')
STILL_FRACTIONS = (0.25, 0.50, 0.75)
ARTIFACT_CSV = 'mouth_movement_velocity_observations.csv'
ARTIFACT_JSON = 'synthetic_demo_summary.json'
ARTIFACT_TXT = 'synthetic_demo_summary.txt'
ARTIFACT_SHEET = 'synthetic_demo_examples.png'

FIELDS = (
    'video', 'source_sha256', 'frame', 'timestamp_seconds', 'source_fps',
    'image_width', 'image_height', 'faces_in_frame', 'track_index', 'track_id',
    'detector_confidence', 'box_x1', 'box_y1', 'box_x2', 'box_y2',
    'landmarks_available', 'landmark_count', 'mouth_available',
    'mouth_motion_available', 'mouth_openness', 'mouth_width', 'mouth_height',
    'mouth_movement', 'mouth_velocity', 'independent_mouth_openness',
    'independent_mouth_width', 'independent_mouth_height',
    'geometry_max_absolute_error', 'temporal_segment_start',
    'previous_observed_frame', 'consecutive_valid_observation',
    'independent_mouth_movement', 'independent_mouth_velocity',
    'movement_absolute_check_error', 'velocity_absolute_check_error',
    'upper_lip_x', 'upper_lip_y', 'lower_lip_x', 'lower_lip_y',
    'left_corner_x', 'left_corner_y', 'right_corner_x', 'right_corner_y',
    'status', 'drawing_status',
)


class DemoError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def ver(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def native_components() -> dict[str, Any]:
    if not SOURCE_ROOT.is_dir():
        raise DemoError('Local src/physiotrack is absent; install script in the PhysioTrack checkout')
    source = str(REPO_ROOT / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig, FaceTracker
    from physiotrack.face.landmarks import FaceLandmarks
    from physiotrack.face.mouth import MouthOpenness
    from physiotrack.face.mouth_motion import MouthMovement
    from physiotrack.models import Models
    found = dict(Face=Face, FaceAnalysis=FaceAnalysis, FaceAnalysisConfig=FaceAnalysisConfig,
                 FaceTracker=FaceTracker, FaceLandmarks=FaceLandmarks,
                 MouthOpenness=MouthOpenness, MouthMovement=MouthMovement)
    imports = {}
    for name, cls in found.items():
        path = Path(inspect.getfile(cls)).resolve()
        if not path.is_relative_to(SOURCE_ROOT.resolve()):
            raise DemoError(f'{name} is not from the current repository source')
        imports[name] = path.relative_to(REPO_ROOT).as_posix()
    model = Path(Models.resolve(Models.Face.MediaPipe.Landmarks.face_landmarker)).resolve()
    if not model.is_file():
        raise DemoError('MediaPipe face_landmarker.task model is missing')
    if sha(model) != EXPECTED_LANDMARK_MODEL_SHA256:
        raise DemoError('Face-landmarker model SHA256 differs from the previously accepted model')
    found['landmark_model'] = model
    found['imports'] = imports
    return found


def sources() -> list[Path]:
    if not INPUT_DIR.is_dir() or INPUT_DIR.is_symlink():
        raise DemoError('Missing/unsafe validation/synthetic_media/mouth_movement_velocity_media')
    videos = sorted((p for p in INPUT_DIR.rglob('*') if p.is_file()
                     and not p.is_symlink() and p.suffix.lower() in EXTENSIONS),
                    key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(),
                                   p.relative_to(INPUT_DIR).as_posix()))
    if not videos:
        raise DemoError('No supported videos found; previously accepted outputs untouched')
    names = [p.relative_to(INPUT_DIR).as_posix().casefold() for p in videos]
    if len(names) != len(set(names)):
        raise DemoError('Ambiguous video paths differing only by letter case')
    return videos


def names(path: Path) -> dict[str, str]:
    rel = path.relative_to(INPUT_DIR).as_posix()
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', path.stem)[:55].strip('._-') or 'video'
    base = safe + '_' + hashlib.sha256(rel.encode('utf-8')).hexdigest()[:12]
    return dict(video=f'videos/{base}_annotated.mp4',
                figure=f'figures/{base}_signals.png',
                stillprefix=f'images/{base}_frame_')


def valid_number(value: Any, label: str, lower: float | None = None,
                 upper: float | None = None) -> float:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise DemoError(f'{label}: missing/non-numeric')
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise DemoError(f'{label}: not a number') from exc
    if not math.isfinite(x) or (lower is not None and x < lower) or (upper is not None and x > upper):
        raise DemoError(f'{label}: nonfinite or outside allowable range ({x})')
    return x


def box_values(box: Any) -> tuple[float, float, float, float]:
    arr = np.asarray(box, dtype=float).reshape(-1)
    if arr.size != 4 or not np.isfinite(arr).all():
        raise DemoError('Invalid native face box (missing/non-finite)')
    x1, y1, x2, y2 = map(float, arr)
    if x2 <= x1 or y2 <= y1:
        raise DemoError('Invalid native face box (non-positive size)')
    return (x1, y1, x2, y2)


class LandmarkCapture:
    """Transparent proxy; captures EXACT inputs to real MouthOpenness."""
    def __init__(self, actual: Any):
        self.actual = actual
        self.calls: list[tuple[np.ndarray, Any]] = []

    def predict_face(self, frame: np.ndarray, box: Any) -> Any:
        pts = self.actual.predict_face(frame, box)
        self.calls.append((np.asarray(box, dtype=float).reshape(-1).copy(), pts))
        return pts

    def close(self) -> None:
        if hasattr(self.actual, 'close'):
            self.actual.close()

    def reset(self) -> None:
        self.calls.clear()

    def match(self, box: tuple[float, ...], used: set[int]) -> Any:
        for i, (called_box, pts) in enumerate(self.calls):
            if i not in used and called_box.size == 4 and np.allclose(called_box, box, rtol=0, atol=1e-5):
                used.add(i)
                return pts
        return None


def make_pipeline(classes: dict[str, Any], detector: Any, fps: float) -> tuple[Any, LandmarkCapture]:
    config = classes['FaceAnalysisConfig'](
        tracking=True, head_pose=False, landmarks=True, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=True, mouth_motion=True, emotion=False, regions=False, temporal=False,
        tracker_type=TRACKER_TYPE,
    )
    config.validate()
    tracker = classes['FaceTracker'](tracker_type=TRACKER_TYPE, device=DEVICE)
    pipeline = classes['FaceAnalysis'](
        detector=detector, tracker=tracker, config=config, fps=fps,
        landmark_model_path=classes['landmark_model'], device=DEVICE, verbose=False,
    )
    try:
        if pipeline.detector is not detector or pipeline.tracker is not tracker:
            raise DemoError('FaceAnalysis is not using the configured native detector/tracker')
        for key, cls in (('landmarks', 'FaceLandmarks'), ('mouth', 'MouthOpenness'),
                         ('mouth_motion', 'MouthMovement')):
            if not isinstance(getattr(pipeline, key), classes[cls]):
                raise DemoError(f'Active {key} is not the real {cls} component')
        if abs(valid_number(pipeline.fps, 'FaceAnalysis.fps') - fps) > 1e-6:
            raise DemoError('FaceAnalysis did not use source-video FPS')
        for key in ('orientation', 'quality', 'eyes', 'blink', 'gaze', 'gaze_estimation',
                    'emotion', 'regions', 'temporal'):
            if getattr(pipeline, key) is not None:
                raise DemoError(f'Unrelated component active: {key}')
        capture = LandmarkCapture(pipeline.landmarks)
        pipeline.landmarks = capture
        return pipeline, capture
    except BaseException:
        pipeline.close()
        raise


def independent_geometry(points: dict[str, tuple[float, float]], width: int,
                         height: int) -> tuple[float, float, float]:
    def dist(a: str, b: str) -> float:
        x1, y1 = points[a]; x2, y2 = points[b]
        return math.hypot((x2 - x1) * width, (y2 - y1) * height) / width
    w = dist('left_corner', 'right_corner')
    h = dist('upper_lip', 'lower_lip')
    if not (math.isfinite(w) and w > 0 and math.isfinite(h) and h >= 0):
        raise DemoError('Invalid captured landmark geometry')
    return (h / w, w, h)



def dashboard_width(source_width: int) -> int:
    """Reserve a separate, non-overlapping panel; never stretch the source image."""
    width = max(320, min(440, int(round(source_width * 0.43))))
    return width + (width % 2)


def annotated_dimensions(source_width: int, source_height: int) -> tuple[int, int]:
    return source_width + dashboard_width(source_width), source_height


def panel_text_width(value: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def short_text(value: Any, available_width: int, scale: float = 0.58,
               thickness: int = 1) -> str:
    """Clip by measured glyph width, never draw beyond the allotted panel."""
    value = str(value).encode('ascii', 'replace').decode('ascii')
    if panel_text_width(value, scale, thickness) <= available_width:
        return value
    if available_width <= panel_text_width('...', scale, thickness):
        return ''
    while value and panel_text_width(value + '...', scale, thickness) > available_width:
        value = value[:-1]
    return value + '...'


def wrap_panel_text(value: str, maximum_width: int, scale: float,
                    max_lines: int = 3) -> list[str]:
    """Split a long video filename into clear bounded lines."""
    remaining = str(value).encode('ascii', 'replace').decode('ascii')
    lines: list[str] = []
    while remaining and len(lines) < max_lines:
        if panel_text_width(remaining, scale) <= maximum_width:
            lines.append(remaining)
            break
        split = 1
        while split < len(remaining) and panel_text_width(remaining[:split + 1], scale) <= maximum_width:
            split += 1
        cut = remaining.rfind(' ', 0, split + 1)
        if cut < max(4, split // 2):
            cut = split
        lines.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
        if len(lines) == max_lines and remaining:
            lines[-1] = short_text(lines[-1] + '...', maximum_width, scale)
    return lines


def render_dashboard(source: np.ndarray, video_name: str, frame_no: int,
                     fps: float, details: list[dict[str, Any]]) -> np.ndarray:
    """Source stays 1:1 at left. All text is isolated in a matte right panel."""
    height, width = source.shape[:2]
    panel_width = dashboard_width(width)
    result = np.empty((height, width + panel_width, 3), dtype=np.uint8)
    result[:, :width] = source
    result[:, width:] = (31, 30, 26)
    x0 = width
    pad = 22
    left, right = x0 + pad, width + panel_width - pad
    max_text = right - left
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.63 if height >= 600 else 0.49
    line_height = 32 if height >= 600 else 25

    def draw(label: str, baseline: int, color: tuple[int, int, int] = (239, 242, 245),
             scale: float | None = None, thickness: int = 1, x: int | None = None) -> None:
        scale = sc if scale is None else scale
        x = left if x is None else x
        if baseline > height - 4 or baseline < 14:
            return
        clipped = short_text(label, right - x, scale, thickness)
        if clipped:
            cv2.putText(result, clipped, (x, baseline), font, scale, color, thickness, cv2.LINE_AA)

    def divider(y: int) -> None:
        if 1 < y < height - 2:
            cv2.line(result, (left, y), (right, y), (73, 68, 58), 1, cv2.LINE_AA)

    # Consistent high contrast, no outlines/shadows, and no overlay on the original image.
    cv2.rectangle(result, (width, 0), (width + 4, height - 1), (158, 188, 53), -1)
    draw('PhysioTrack', 48, (255, 255, 255), 0.81 if height >= 600 else sc, 2)
    draw('MOUTH MOVEMENT / VELOCITY', 78, (177, 212, 233), 0.52 if height >= 600 else 0.41)
    divider(97)
    draw('VIDEO', 126, (175, 198, 213), 0.53 if height >= 600 else 0.42)
    caption_scale = 0.55 if height >= 600 else 0.42
    display_name = Path(video_name).name.replace('_', ' ')
    captions = wrap_panel_text(display_name, max_text, caption_scale, 3)
    cursor = 154
    for item in captions:
        draw(item, cursor, scale=caption_scale)
        cursor += (25 if height >= 600 else 21)
    cursor += 11
    stat_scale = sc
    for label in (f'Frame: {frame_no}', f'Time: {(frame_no - 1) / fps:.3f} s',
                  f'FPS: {fps:.3f}', f'Tracked faces: {len(details)}'):
        draw(label, cursor, scale=stat_scale)
        cursor += line_height
    divider(cursor + 5)
    cursor += 37

    if not details:
        draw('NO TRACKED FACE', cursor, (108, 191, 255), 0.68 if height >= 600 else sc, 2)
        draw('No face measurements available.', cursor + line_height + 5, scale=0.48)
    else:
        # Each tracked face gets its own solid, separated card. No stacked text.
        card_line = 27 if height >= 600 else 22
        card_height = 6 * card_line + 29
        room_end = height - 20
        shown = 0
        for i, rec in enumerate(details):
            if cursor + card_height > room_end:
                break
            card_top = cursor - 25
            card_bottom = card_top + card_height
            color = (92, 168, 66) if rec['motion_available'] else (60, 145, 206)
            cv2.rectangle(result, (left - 7, card_top), (right + 6, card_bottom), (46, 44, 39), -1)
            cv2.rectangle(result, (left - 7, card_top), (left - 3, card_bottom), color, -1)
            rec_label = short_text(f'Track {rec["track_id"]}  (face {i + 1})',
                                   max_text, sc, 1)
            draw(rec_label, cursor, (245, 247, 248), thickness=2 if height >= 600 else 1)
            cursor += card_line
            confidence = rec['confidence']
            draw('Confidence: ' + ('N/A' if confidence == '' else f'{confidence:.3f}'),
                 cursor, scale=0.55 if height >= 600 else 0.44)
            cursor += card_line
            row = rec['row']
            if rec['motion_available']:
                items = (f'Openness: {row["mouth_openness"]:.5f}',
                         f'Movement: {row["mouth_movement"]:.5f}',
                         f'Velocity: {row["mouth_velocity"]:.5f}/s',
                         'Segment: NEW' if row['temporal_segment_start'] else 'Segment: continuous')
            else:
                items = (f'Status: {row["status"]}', 'Openness: ' + (
                    f'{row["mouth_openness"]:.5f}' if rec['mouth_available'] else 'N/A'),
                    'Movement: N/A', 'Velocity: N/A')
            for entry in items:
                draw(entry, cursor, (236, 240, 242), scale=0.55 if height >= 600 else 0.44)
                cursor += card_line
            cursor = card_bottom + 38
            shown += 1
        if shown != len(details):
            draw(f'+ {len(details) - shown} more tracked face(s)',
                 min(cursor, height - 24), (125, 210, 245), scale=0.49)

    # Distinct image and data regions also make montage stills readable at full size.
    return result

def process_frame(frame: np.ndarray, result: Any, capture: LandmarkCapture,
                  name: str, src_hash: str, frame_no: int, fps: float,
                  temporal: dict[str, tuple[int, float]],
                  rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    height, width = frame.shape[:2]
    faces = list(result.instances)
    counters = Counter(faces=len(faces), openness=0, available=0, unavailable=0, no_face=int(len(faces) == 0))
    drawn = frame.copy()
    detail_rows: list[dict[str, Any]] = []
    used_calls: set[int] = set()
    seen_ids: set[str] = set()
    if not faces:
        row = {k: '' for k in FIELDS}
        row.update(video=name, source_sha256=src_hash, frame=frame_no,
                   timestamp_seconds=(frame_no - 1) / fps, source_fps=fps,
                   image_width=width, image_height=height, faces_in_frame=0,
                   status='NO_FACE', drawing_status='NO_FACE')
        rows.append(row)
        return render_dashboard(drawn, name, frame_no, fps, detail_rows), counters

    for index, face in enumerate(faces, 1):
        identity = str(face.id)
        if face.id is None or not identity or identity in seen_ids:
            raise DemoError('Native tracker provided missing/duplicate identity in one frame')
        seen_ids.add(identity)
        coords = box_values(face.box)
        confidence = '' if face.confidence is None else valid_number(face.confidence, 'confidence', 0, 1)
        feats = face.face_features if isinstance(face.face_features, dict) else {}
        lmk = feats.get('landmarks')
        mouth = feats.get('mouth')
        motion = feats.get('mouth_motion')
        if not all(isinstance(item, dict) for item in (lmk, mouth, motion)):
            raise DemoError('Native FaceAnalysis did not export all enabled feature dictionaries')
        landmark_available = bool(lmk.get('available', False))
        count = int(lmk.get('count', 0) or 0)
        mouth_available = bool(mouth.get('available', False))
        motion_available = bool(motion.get('available', False))
        if landmark_available and count != EXPECTED_LANDMARKS:
            raise DemoError('Unexpected available landmark count')
        if motion_available and not mouth_available:
            raise DemoError('MouthMovement available with missing MouthOpenness')
        row = {k: '' for k in FIELDS}
        row.update(video=name, source_sha256=src_hash, frame=frame_no,
                   timestamp_seconds=(frame_no - 1) / fps, source_fps=fps,
                   image_width=width, image_height=height, faces_in_frame=len(faces),
                   track_index=index, track_id=identity, detector_confidence=confidence,
                   box_x1=coords[0], box_y1=coords[1], box_x2=coords[2], box_y2=coords[3],
                   landmarks_available=landmark_available, landmark_count=count,
                   mouth_available=mouth_available, mouth_motion_available=motion_available)
        pts = capture.match(coords, used_calls)
        if landmark_available:
            if pts is None or len(pts) != EXPECTED_LANDMARKS:
                raise DemoError('Could not associate real captured landmarks with native face result')
        if mouth_available:
            if pts is None or len(pts) != EXPECTED_LANDMARKS:
                raise DemoError('Available mouth result without the exact captured landmark points')
            landmarks = {'left_corner': 61, 'right_corner': 291, 'upper_lip': 13, 'lower_lip': 14}
            xy = {}
            for label, id_ in landmarks.items():
                p = pts[id_]
                x, y = valid_number(p.x, label + '.x'), valid_number(p.y, label + '.y')
                xy[label] = x, y
                row[label + '_x'], row[label + '_y'] = x, y
            expected = independent_geometry(xy, width, height)
            real = tuple(valid_number(mouth.get(k), k, 0) for k in ('mouth_openness', 'mouth_width', 'mouth_height'))
            for k, x, e in zip(('mouth_openness', 'mouth_width', 'mouth_height'), real, expected):
                row[k] = x
                row['independent_' + k] = e
            row['geometry_max_absolute_error'] = max(abs(a - b) for a, b in zip(real, expected))
            if row['geometry_max_absolute_error'] > GEOMETRY_TOL or real[1] <= 0:
                raise DemoError('Real MouthOpenness differs from captured MediaPipe landmark geometry')
        else:
            if any(mouth.get(key) is not None for key in ('mouth_openness', 'mouth_width', 'mouth_height')):
                raise DemoError('Unavailable MouthOpenness exposed numerical values')

        previous = temporal.get(identity)
        if mouth_available and motion_available:
            movement = valid_number(motion.get('mouth_movement'), 'mouth_movement', 0)
            velocity = valid_number(motion.get('mouth_velocity'), 'mouth_velocity', 0)
            row['mouth_movement'], row['mouth_velocity'] = movement, velocity
            current_openness = float(row['mouth_openness'])
            consecutive = previous is not None and previous[0] == frame_no - 1
            start = not consecutive
            expected_movement = 0.0 if start else abs(current_openness - previous[1])
            expected_velocity = expected_movement * fps
            row.update(temporal_segment_start=start,
                       previous_observed_frame='' if previous is None else previous[0],
                       consecutive_valid_observation=consecutive,
                       independent_mouth_movement=expected_movement,
                       independent_mouth_velocity=expected_velocity,
                       movement_absolute_check_error=abs(movement - expected_movement),
                       velocity_absolute_check_error=abs(velocity - expected_velocity))
            if abs(movement - expected_movement) > MOVEMENT_TOL or abs(velocity - expected_velocity) > VELOCITY_TOL:
                # Do not quietly mask a discrepancy caused by identity switches / occlusion.
                raise DemoError(f'Native temporal outputs differ from per-track arithmetic: '
                                f'video={name} frame={frame_no} ID={identity} '
                                f'previous={previous}; movement={movement:.8g} expected={expected_movement:.8g}; '
                                f'velocity={velocity:.8g} expected={expected_velocity:.8g}')
            temporal[identity] = (frame_no, current_openness)
            row['status'] = 'MOTION_AVAILABLE'
            counters['available'] += 1
        else:
            temporal.pop(identity, None)
            row['status'] = 'MOUTH_UNAVAILABLE' if not mouth_available else 'MOTION_UNAVAILABLE'
            counters['unavailable'] += 1
            # An available openness without a native motion output is NOT silently fabricated.
            if motion_available:
                raise DemoError('Contradictory MouthMovement state')

        if mouth_available:
            counters['openness'] += 1

        x1, y1, x2, y2 = coords
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(0, min(width - 1, int(round(x2))))
        bottom = max(0, min(height - 1, int(round(y2))))
        if right <= left or bottom <= top:
            row['drawing_status'] = 'CLIPPED_BOX_NOT_DRAWN'
        else:
            color = (60, 220, 60) if motion_available else (15, 160, 255)
            cv2.rectangle(drawn, (left, top), (right, bottom), color, max(1, min(width, height) // 500))
            if mouth_available:
                for a, b, clr in (('left_corner', 'right_corner', (255, 215, 0)),
                                  ('upper_lip', 'lower_lip', (0, 190, 255))):
                    pa = (max(0, min(width - 1, int(round(float(row[a + '_x']) * width)))),
                          max(0, min(height - 1, int(round(float(row[a + '_y']) * height)))))
                    pb = (max(0, min(width - 1, int(round(float(row[b + '_x']) * width)))),
                          max(0, min(height - 1, int(round(float(row[b + '_y']) * height)))))
                    cv2.line(drawn, pa, pb, clr, 2, cv2.LINE_AA)
            row['drawing_status'] = 'DRAWN'
        detail_rows.append(dict(track_id=identity, confidence=confidence,
                                row=row, mouth_available=mouth_available,
                                motion_available=motion_available))
        rows.append(row)
    if len(used_calls) != len(capture.calls):
        raise DemoError('Some native FaceLandmarks calls were not associated with output faces')
    return render_dashboard(drawn, name, frame_no, fps, detail_rows), counters


def input_capture(path: Path, workspace: Path, idx: int) -> tuple[Any, Path | None]:
    capture = cv2.VideoCapture(str(path))
    temporary = None
    if not capture.isOpened():
        capture.release()
        temporary = workspace / f'input_{idx:05d}{path.suffix.lower()}'
        shutil.copyfile(path, temporary)
        capture = cv2.VideoCapture(str(temporary))
    if not capture.isOpened():
        capture.release()
        raise DemoError(f'OpenCV cannot read source video: {path.name}')
    return capture, temporary


def output_writer(path: Path, fps: float, width: int, height: int) -> tuple[Any, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in OUTPUT_CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
        path.unlink(missing_ok=True)
    raise DemoError('No MP4 encoder available in OpenCV (mp4v/avc1)')


def verify_mp4(path: Path, width: int, height: int, fps: float, n: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise DemoError('Annotated MP4 cannot be reopened: ' + path.name)
    try:
        vw = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        vh = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vf = float(cap.get(cv2.CAP_PROP_FPS))
        if (vw, vh) != (width, height) or not math.isfinite(vf) or abs(vf - fps) > max(0.05, fps * 0.01):
            raise DemoError('Annotated MP4 width/height/FPS incorrect: ' + path.name)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape != (height, width, 3):
                raise DemoError('Corrupt annotated MP4 frame: ' + path.name)
            count += 1
        if count != n:
            raise DemoError(f'Annotated MP4 frame count mismatch: {count} vs {n}')
    finally:
        cap.release()


def save_png(path: Path, img: np.ndarray) -> None:
    ok, compressed = cv2.imencode('.png', img)
    if not ok:
        raise DemoError('PNG encoding failed: ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed.tofile(str(path))


def read_png(path: Path) -> np.ndarray:
    try:
        result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise DemoError('Cannot read PNG: ' + path.name) from exc
    if result is None:
        raise DemoError('Unusable PNG: ' + path.name)
    return result


def stills_from_video(mp4: Path, prefix: str, stage: Path,
                      frame_count: int) -> list[str]:
    indices = sorted({max(1, min(frame_count, int(round(frame_count * x)))) for x in STILL_FRACTIONS})
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        cap.release()
        raise DemoError('Cannot reopen annotated video for stills')
    output: list[str] = []
    try:
        for idx in range(1, max(indices) + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise DemoError('Cannot decode selected illustrative video frame')
            if idx in indices:
                name = prefix + f'{idx:06d}.png'
                save_png(stage / name, frame)
                output.append(name)
    finally:
        cap.release()
    return output


def save_plot(dest: Path, rows: list[dict[str, Any]], label: str) -> None:
    # Matplotlib is for illustrative timelines only. No synthetic accuracy curves.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['status'] == 'MOTION_AVAILABLE':
            by_id[str(row['track_id'])].append(row)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = (('mouth_openness', 'Mouth openness (ratio)'),
              ('mouth_movement', 'Mouth movement (ratio/frame)'),
              ('mouth_velocity', 'Mouth velocity (ratio/second)'))
    for key, values in sorted(by_id.items(), key=lambda p: p[0]):
        # Never interpolate visually over lost tracks or temporal resets.
        segments: list[list[dict[str, Any]]] = []
        segment: list[dict[str, Any]] = []
        for row in values:
            if segment and (int(row['frame']) != int(segment[-1]['frame']) + 1
                            or bool(row.get('temporal_segment_start'))):
                segments.append(segment)
                segment = []
            segment.append(row)
        if segment:
            segments.append(segment)
        for j, segment in enumerate(segments):
            for ax, (field, ylabel) in zip(axes, labels):
                ax.plot([float(r['timestamp_seconds']) for r in segment],
                        [float(r[field]) for r in segment],
                        label=(f'ID {key}' if j == 0 else None),
                        linewidth=1.1, marker='.', markersize=2)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_ylim(bottom=0)
    axes[0].set_title('PhysioTrack native per-track mouth signals | ' + label)
    axes[-1].set_xlabel('Source time (seconds)')
    if by_id:
        axes[0].legend(loc='best', fontsize=8)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No available mouth signals',
                    transform=ax.transAxes, ha='center', va='center')
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def sheet(stage: Path, records: list[dict[str, Any]]) -> np.ndarray:
    """Readable montage: preview the source ROI, NOT a miniature data dashboard."""
    tile_w, tile_h = 520, 385
    cols = 2
    n_rows = math.ceil(len(records) / cols)
    montage = np.full((n_rows * tile_h, cols * tile_w, 3), (239, 238, 235), dtype=np.uint8)
    for i, record in enumerate(records):
        options = record['stills']
        if not options:
            raise DemoError('No illustrative still exists for a processed video')
        annotated = read_png(stage / options[len(options) // 2])
        original_width = int(record['width'])
        original_height = int(record['height'])
        if annotated.shape[:2] != (original_height, int(record['annotated_width'])):
            raise DemoError('Illustrative still dimensions conflict with JSON manifest')
        image = annotated[:, :original_width]
        canvas = np.full((tile_h, tile_w, 3), (239, 238, 235), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (tile_w - 1, 46), (36, 34, 29), -1)
        name = Path(str(record['input'])).name.replace('_', ' ')
        title = short_text(name, tile_w - 20, 0.60, 1)
        cv2.putText(canvas, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)
        scale = min((tile_w - 16) / original_width, (tile_h - 114) / original_height)
        resized = cv2.resize(image, (max(1, round(original_width * scale)),
                                     max(1, round(original_height * scale))),
                             interpolation=cv2.INTER_AREA)
        x = (tile_w - resized.shape[1]) // 2
        y = 53 + (tile_h - 114 - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        stats = (f'Frames {record["frames_processed"]}   |   '
                 f'FPS {float(record["source_fps"]):.2f}   |   '
                 f'Motion rows {record["mouth_motion_available"]}')
        cv2.putText(canvas, short_text(stats, tile_w - 20, 0.56),
                    (12, tile_h - 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (36, 36, 36), 1, cv2.LINE_AA)
        y0, x0 = (i // cols) * tile_h, (i % cols) * tile_w
        montage[y0:y0 + tile_h, x0:x0 + tile_w] = canvas
    return montage


def write_csv(dest: Path, rows: list[dict[str, Any]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def text_summary(summary: dict[str, Any]) -> str:
    lines = ['PhysioTrack MouthMovement/Velocity Synthetic Demonstration',
             'This is a non-benchmark qualitative software demonstration.', '']
    def walk(x: Any, indent: int = 0, key: str | None = None) -> None:
        pre = ' ' * indent
        if isinstance(x, dict):
            if key is not None:
                lines.append(f'{pre}{key}:')
                indent += 2
            for k, v in x.items():
                walk(v, indent, str(k))
        elif isinstance(x, list):
            if key is not None:
                lines.append(f'{pre}{key}: {len(x)} items')
                indent += 2
            for i, v in enumerate(x, 1):
                walk(v, indent, f'item_{i:03d}')
        else:
            lines.append(f'{pre}{key}: {json.dumps(x, ensure_ascii=False, allow_nan=False)}')
    walk(summary)
    return '\n'.join(lines) + '\n'


def write_reports(stage: Path, summary: dict[str, Any]) -> None:
    (stage / ARTIFACT_JSON).write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + '\n', encoding='utf-8')
    (stage / ARTIFACT_TXT).write_text(text_summary(summary), encoding='utf-8')


def check_rows(rows: list[dict[str, str]], records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row['video']].append(row)
    if set(groups) != {r['input'] for r in records}:
        raise DemoError('CSV video names do not match JSON video manifest')
    for r in records:
        block = groups[r['input']]
        frames = int(r['frames_processed'])
        if len({int(x['frame']) for x in block}) != frames:
            raise DemoError('CSV missing or duplicate source video frame')
        frame_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
        prev: dict[str, tuple[int, float]] = {}
        for x in block:
            fr = int(x['frame']); frame_rows[fr].append(x)
            if x['source_sha256'] != r['source_sha256']:
                raise DemoError('CSV source SHA mismatch')
            if abs(float(x['source_fps']) - float(r['source_fps'])) > 1e-9:
                raise DemoError('CSV source FPS differs from source metadata')
            if (int(x['image_width']), int(x['image_height'])) != (int(r['width']), int(r['height'])):
                raise DemoError('CSV video dimensions disagree with the manifest')
            if abs(float(x['timestamp_seconds']) - (fr - 1) / float(r['source_fps'])) > 1e-9:
                raise DemoError('CSV source time does not match FPS')
            if x['status'] == 'NO_FACE':
                if x['track_id'] or int(x['faces_in_frame']) != 0:
                    raise DemoError('Contradictory NO_FACE CSV row')
                continue
            if not x['track_id']:
                raise DemoError('Tracked observation without identity')
            if x['status'] == 'MOTION_AVAILABLE':
                o = float(x['mouth_openness'])
                m = float(x['mouth_movement']); v = float(x['mouth_velocity'])
                independent = (float(x['independent_mouth_openness']),
                               float(x['independent_mouth_width']), float(x['independent_mouth_height']))
                observed = (o, float(x['mouth_width']), float(x['mouth_height']))
                if max(abs(a-b) for a,b in zip(observed, independent)) > GEOMETRY_TOL:
                    raise DemoError('Serialized mouth geometry failed independent checks')
                if abs(o - observed[2] / observed[1]) > GEOMETRY_TOL:
                    raise DemoError('Serialized openness ratio is inconsistent')
                old = prev.get(x['track_id'])
                consecutive = old is not None and old[0] == fr - 1
                expected_m = abs(o - old[1]) if consecutive else 0.0
                expected_v = expected_m * float(r['source_fps'])
                if (abs(m - expected_m) > MOVEMENT_TOL or
                        abs(v - expected_v) > VELOCITY_TOL or
                        abs(float(x['independent_mouth_movement']) - expected_m) > MOVEMENT_TOL or
                        abs(float(x['independent_mouth_velocity']) - expected_v) > VELOCITY_TOL):
                    raise DemoError('Serialized MouthMovement/Velocity failed temporal check')
                if x['temporal_segment_start'] != str(not consecutive):
                    raise DemoError('Serialized temporal segment start inconsistent')
                prev[x['track_id']] = fr, o
            elif x['status'] in ('MOUTH_UNAVAILABLE', 'MOTION_UNAVAILABLE'):
                prev.pop(x['track_id'], None)
            else:
                raise DemoError('Unexpected serialized native motion status')
        if sorted(frame_rows) != list(range(1, frames+1)):
            raise DemoError('Noncontiguous video frames in CSV')
        actual_counts = Counter()
        for fr, sample in frame_rows.items():
            if len(sample) == 1 and sample[0]['status'] == 'NO_FACE':
                actual_counts['no_face_frames'] += 1
            else:
                expected_face_count = int(sample[0]['faces_in_frame'])
                if len(sample) != expected_face_count:
                    raise DemoError('Inconsistent number of per-frame tracked face observations')
                if len({s['track_id'] for s in sample}) != len(sample):
                    raise DemoError('Duplicate face identity in a video frame')
                actual_counts['face_observations'] += len(sample)
                actual_counts['mouth_available'] += sum(s['mouth_available'] == 'True' for s in sample)
                actual_counts['mouth_motion_available'] += sum(s['status'] == 'MOTION_AVAILABLE' for s in sample)
                actual_counts['mouth_unavailable'] += sum(s['status'] != 'MOTION_AVAILABLE' for s in sample)
        for k in ('no_face_frames', 'face_observations', 'mouth_available', 'mouth_motion_available', 'mouth_unavailable'):
            if actual_counts[k] != r[k]:
                raise DemoError(f'CSV/JSON per-video count mismatch: {k}')


def verify(stage: Path, summary: dict[str, Any], all_rows: list[dict[str, Any]] | None = None) -> None:
    loaded = json.loads((stage / ARTIFACT_JSON).read_text(encoding='utf-8'))
    if loaded != summary or (stage / ARTIFACT_TXT).read_text(encoding='utf-8') != text_summary(summary):
        raise DemoError('JSON/TXT summary disagree')
    if summary['status'] != 'COMPLETE' or summary['files_failed'] != 0:
        raise DemoError('Incomplete synthetic execution cannot be published')
    if summary['files_processed'] != summary['files_discovered'] or not summary['files_processed']:
        raise DemoError('Incomplete video processing in summary')
    with (stage / ARTIFACT_CSV).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise DemoError('Unexpected CSV schema')
        rows = list(reader)
    if all_rows is not None and len(all_rows) != len(rows):
        raise DemoError('CSV row accounting mismatch')
    check_rows(rows, summary['videos'])
    if summary['observations'] != len(rows):
        raise DemoError('Incorrect serialized observation count')
    expected_totals = dict(frames=sum(r['frames_processed'] for r in summary['videos']),
                           face_observations=sum(r['face_observations'] for r in summary['videos']),
                           mouth_available=sum(r['mouth_available'] for r in summary['videos']),
                           mouth_motion_available=sum(r['mouth_motion_available'] for r in summary['videos']),
                           mouth_unavailable=sum(r['mouth_unavailable'] for r in summary['videos']),
                           frames_with_no_face=sum(r['no_face_frames'] for r in summary['videos']))
    if 'totals' in summary and summary['totals'] != expected_totals:
        raise DemoError('Incorrect summary totals versus per-video manifest')
    if sha(stage / ARTIFACT_CSV) != summary['artifact_sha256'][ARTIFACT_CSV]:
        raise DemoError('CSV SHA256 mismatch')
    if sha(stage / ARTIFACT_SHEET) != summary['artifact_sha256'][ARTIFACT_SHEET]:
        raise DemoError('Contact sheet SHA256 mismatch')
    if not np.array_equal(read_png(stage / ARTIFACT_SHEET), sheet(stage, summary['videos'])):
        raise DemoError('Illustrative contact-sheet pixels disagree')
    expected = {ARTIFACT_CSV, ARTIFACT_JSON, ARTIFACT_TXT, ARTIFACT_SHEET}
    for info in summary['videos']:
        expected.add(info['output_video']); expected.add(info['figure'])
        expected.update(info['stills'])
        if sha(stage / info['output_video']) != info['output_sha256']:
            raise DemoError('Annotated MP4 SHA256 mismatch')
        if sha(stage / info['figure']) != info['figure_sha256']:
            raise DemoError('Timeline figure SHA256 mismatch')
        for f in info['stills']:
            if sha(stage / f) != info['still_sha256'][f]:
                raise DemoError('Preview PNG SHA256 mismatch')
            if read_png(stage / f).ndim != 3:
                raise DemoError('Preview PNG invalid')
        if ((int(info['annotated_width']), int(info['annotated_height'])) !=
                annotated_dimensions(int(info['width']), int(info['height']))):
            raise DemoError('Recorded dashboard output dimensions are inconsistent')
        verify_mp4(stage / info['output_video'], int(info['annotated_width']),
                   int(info['annotated_height']), float(info['source_fps']),
                   int(info['frames_processed']))
    present = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    if expected != present:
        raise DemoError('Missing/unexpected output files: ' + str(expected ^ present))


def publish(stage: Path, workspace: Path, summary: dict[str, Any]) -> None:
    verify(stage, summary)
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError('Unsafe existing synthetic_demo destination')
    backup = workspace / 'previous_accepted_synthetic_demo'
    had_previous = FINAL_DIR.exists()
    installed = False
    try:
        if had_previous:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        installed = True
        verify(FINAL_DIR, summary)
    except BaseException:
        try:
            if installed and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if had_previous and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as e:
            raise DemoError(f'Rollback requires manual recovery from {backup}') from e
        raise
    if had_previous:
        try:
            shutil.rmtree(backup)
        except OSError as e:
            print(f'WARNING: new results verified; backup still exists at {backup}: {e}', flush=True)


def main() -> None:
    classes = native_components()
    videos = sources()
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError('Unsafe existing final path')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='.mouth_movement_velocity_synthetic_stage_', dir=RESULTS_DIR))
    stage = workspace / 'synthetic_demo'
    stage.mkdir()
    detector = None
    try:
        print('PhysioTrack MouthMovement / Velocity Synthetic Qualitative Demonstration')
        print(f'Videos discovered: {len(videos)}')
        print('Input: validation/synthetic_media/mouth_movement_velocity_media/')
        print('Output: validation/mouth_movement_velocity/results/synthetic_demo/')
        detector = classes['Face'](device=DEVICE, conf=DETECTOR_CONFIDENCE,
                                   iou=DETECTOR_IOU, max_det=MAX_DETECTIONS, verbose=False)
        observations: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for i, source in enumerate(videos, 1):
            name = source.relative_to(INPUT_DIR).as_posix()
            expected_hash = sha(source)
            paths = names(source)
            print(f'[{i}/{len(videos)}] {name}', flush=True)
            cap, temp_input = input_capture(source, workspace, i)
            pipeline = None
            writer = None
            tmp_video = workspace / f'annotating_{i:05d}.mp4'
            try:
                width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                claimed_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                if width < 1 or height < 1 or width % 2 or height % 2:
                    raise DemoError('Source dimensions invalid or odd (MP4 codec requires even dimensions)')
                if not math.isfinite(fps) or fps <= 0:
                    raise DemoError('Source video FPS is invalid')
                pipeline, capture = make_pipeline(classes, detector, fps)
                annotated_width, annotated_height = annotated_dimensions(width, height)
                writer, codec = output_writer(tmp_video, fps, annotated_width, annotated_height)
                frame_no = 0
                local_rows: list[dict[str, Any]] = []
                temporal: dict[str, tuple[int, float]] = {}
                counters = Counter()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame is None or frame.shape != (height, width, 3):
                        raise DemoError('Corrupt input video frame / inconsistent dimensions')
                    frame_no += 1
                    capture.reset()
                    real_result = pipeline.predict(frame)
                    image, stats = process_frame(frame, real_result, capture, name, expected_hash,
                                                 frame_no, fps, temporal, local_rows)
                    writer.write(image)
                    counters.update(stats)
                    if frame_no % 60 == 0:
                        print(f'    frames={frame_no} faces={counters["faces"]} '
                              f'motion_available={counters["available"]}', flush=True)
                if frame_no <= 0:
                    raise DemoError('Source video decodes zero frames')
                if claimed_frames > 0 and abs(claimed_frames - frame_no) > 2:
                    raise DemoError(f'Incomplete video decode: declared {claimed_frames}, read {frame_no}')
                writer.release(); writer = None
                verify_mp4(tmp_video, annotated_width, annotated_height, fps, frame_no)
                dest = stage / paths['video']
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_video, dest)
                still_paths = stills_from_video(dest, paths['stillprefix'], stage, frame_no)
                save_plot(stage / paths['figure'], local_rows, name)
                if sha(source) != expected_hash:
                    raise DemoError('Source video content changed while processing')
                rec = dict(input=name, source_sha256=expected_hash, width=width, height=height,
                           annotated_width=annotated_width, annotated_height=annotated_height,
                           source_fps=fps, declared_source_frames=(claimed_frames if claimed_frames > 0 else None),
                           frames_processed=frame_no, no_face_frames=counters['no_face'],
                           face_observations=counters['faces'], mouth_available=counters['openness'],
                           mouth_motion_available=counters['available'],
                           mouth_unavailable=counters['unavailable'],
                           observed_track_ids=sorted({r['track_id'] for r in local_rows if r['track_id']}),
                           output_video=paths['video'], output_codec=codec, output_sha256=sha(dest),
                           figure=paths['figure'], figure_sha256=sha(stage / paths['figure']),
                           stills=still_paths,
                           still_sha256={f: sha(stage / f) for f in still_paths})
                records.append(rec)
                observations.extend(local_rows)
                print(f'    PASS frames={frame_no} face_rows={counters["faces"]} '
                      f'mouth_motion_rows={counters["available"]} no_face_frames={counters["no_face"]}',
                      flush=True)
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
                if pipeline is not None:
                    pipeline.close()
                tmp_video.unlink(missing_ok=True)
                if temp_input is not None:
                    temp_input.unlink(missing_ok=True)
        if not any(r['mouth_motion_available'] for r in records):
            raise DemoError('No native MouthMovement measurements were available in any source')
        write_csv(stage / ARTIFACT_CSV, observations)
        save_png(stage / ARTIFACT_SHEET, sheet(stage, records))
        source_files = {Path(__file__).resolve()}
        for cls in (classes['Face'], classes['FaceAnalysis'], classes['FaceAnalysisConfig'],
                    classes['FaceTracker'], classes['FaceLandmarks'], classes['MouthOpenness'],
                    classes['MouthMovement']):
            source_files.add(Path(inspect.getfile(cls)).resolve())
        versions = dict(python=platform.python_version(), platform=platform.system(),
                        opencv=cv2.__version__, numpy=np.__version__,
                        mediapipe=ver('mediapipe'), ultralytics=ver('ultralytics'),
                        physiotrack=ver('physiotrack'),
                        landmark_model_sha256=sha(classes['landmark_model']),
                        component_file_sha256={p.relative_to(REPO_ROOT).as_posix(): sha(p)
                                               for p in sorted(source_files)})
        summary = {
            'project': 'physiotrack-v2', 'component': 'MouthMovement / MouthVelocity',
            'type': 'synthetic_qualitative_demonstration', 'scientific_benchmark': False,
            'uses_ground_truth_motion_labels': False, 'calculates_accuracy_metrics': False,
            'input_directory': 'validation/synthetic_media/mouth_movement_velocity_media',
            'output_directory': 'validation/mouth_movement_velocity/results/synthetic_demo',
            'configuration': dict(visualization_layout='source at unchanged 1:1 scale on left; distinct readable measurements panel on right',
                                  annotation_font='OpenCV Hershey Simplex, without thick stroke or text overlay on faces',
                                  output_dimensions='(source_width + dashboard_width, source_height); source FPS unchanged',
                                  pipeline='real Face detector -> FaceTracker -> FaceAnalysis -> '
                                  'FaceLandmarks -> MouthOpenness -> MouthMovement',
                                  device=DEVICE, tracker_type=TRACKER_TYPE,
                                  detector_confidence=DETECTOR_CONFIDENCE,
                                  detector_iou=DETECTOR_IOU,
                                  fps='individual decoded input FPS (no fixed 30-FPS assumption)',
                                  geometry_tolerance=GEOMETRY_TOL,
                                  movement_tolerance=MOVEMENT_TOL,
                                  velocity_tolerance=VELOCITY_TOL,
                                  separate_pipeline_and_temporal_state_per_video=True,
                                  unrelated_components_disabled=True),
            'component_imports': classes['imports'], 'reproducibility': versions,
            'files_discovered': len(videos), 'files_processed': len(records), 'files_failed': 0,
            'observations': len(observations),
            'totals': dict(frames=sum(r['frames_processed'] for r in records),
                           face_observations=sum(r['face_observations'] for r in records),
                           mouth_available=sum(r['mouth_available'] for r in records),
                           mouth_motion_available=sum(r['mouth_motion_available'] for r in records),
                           mouth_unavailable=sum(r['mouth_unavailable'] for r in records),
                           frames_with_no_face=sum(r['no_face_frames'] for r in records)),
            'videos': records,
            'artifact_sha256': {ARTIFACT_CSV: sha(stage / ARTIFACT_CSV),
                                ARTIFACT_SHEET: sha(stage / ARTIFACT_SHEET)},
            'interpretation': ('Independent non-benchmark real-pipeline video outputs. '
                               'Values are the real native per-track PhysioTrack measurements. '
                               'Arithmetic is independently verified from the recorded landmark '
                               'coordinates, consecutive openness samples, and actual input FPS. '
                               'The chosen synthetic motion types have no measured ground truth; '
                               'this is NOT a scientific accuracy result, detection-recall result '
                               'or quantitative multi-person ID accuracy study. '
                               'Missing faces/features and potential ID switches must be interpreted '
                               'from the explicit observations, not hidden or assigned fabricated values.'),
            'status': 'COMPLETE',
        }
        write_reports(stage, summary)
        print('Validating staged CSV, reports, figures, stills and decoded annotated videos...', flush=True)
        verify(stage, summary, observations)
        publish(stage, workspace, summary)
        print(f'Committed verified results to: {FINAL_DIR}', flush=True)
        print(f'Videos={len(records)} frames={summary["totals"]["frames"]} '
              f'face_observations={summary["totals"]["face_observations"]} '
              f'mouth_motion_available={summary["totals"]["mouth_motion_available"]}', flush=True)
        print('Overall synthetic demonstration status: COMPLETE', flush=True)
    finally:
        # Never erase the only recoverable backup if final publication failed.
        if not (workspace / 'previous_accepted_synthetic_demo').exists():
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Synthetic MouthMovement/Velocity demo FAILED; prior results preserved where possible')
        print(f'{type(error).__name__}: {error}')
        traceback.print_exc()
        sys.exit(1)
