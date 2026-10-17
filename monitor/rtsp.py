"""RTSP polling, fire/smoke detection and temporal verification.

Adapted from the Edge Gateway implementation. Thread handling is unchanged in
spirit: one daemon thread per camera, a per-camera stop event, a registry of
threads, and start / stop / restart helpers.

Differences from the original:
  * Models are CCTVDevice / AlarmRule / ModelPrediction instead of Device / Alarm /
    CCTVConfiguration / CCTVAnalysisData / AlarmHistory.
  * No MQTT publishing; a verified alarm is stored on the ModelPrediction row.
  * The YOLO model is loaded lazily (so the web UI works before model.pt exists)
    and inference is serialised with a lock (one shared model, many camera threads).
  * Snapshots are stored under MEDIA_ROOT and their path is saved relative to it.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict

import cv2
from django.conf import settings
from django.db import close_old_connections
from django.utils.text import slugify

from .models import AlarmRule, CCTVDevice, ModelPrediction

logger = logging.getLogger("rtsp")

# Prefer TCP for RTSP: far fewer corrupted frames than UDP on lossy networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_THREADS: Dict[int, threading.Thread] = {}
_THREADS_MUTEX = threading.Lock()

# ---------------------------------------------------------------------------
# Stop-event registry: one threading.Event per camera device_id.
# Set the event to signal that camera's polling loop to exit cleanly.
# ---------------------------------------------------------------------------

_STOP_EVENTS: Dict[int, threading.Event] = {}
_STOP_EVENTS_MUTEX = threading.Lock()


def _get_stop_event(device_id: int) -> threading.Event:
    """Return (creating if needed) the stop event for *device_id*."""
    with _STOP_EVENTS_MUTEX:
        if device_id not in _STOP_EVENTS:
            _STOP_EVENTS[device_id] = threading.Event()
        return _STOP_EVENTS[device_id]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_temporal_states = {}

DEFAULT_TEMPORAL_FRAMES = 3

DEFAULT_POLL_INTERVAL = 60  # Default snapshot interval (seconds)

DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


def get_temporal_state(device_id: int):
    """Snapshot of the in-progress temporal sequence for a camera (used by the live UI)."""
    state = _temporal_states.get(device_id)
    return dict(state) if state else None


# ---------------------------------------------------------------------------
# Fire / smoke model (loaded on first use)
# ---------------------------------------------------------------------------

class ModelNotAvailable(RuntimeError):
    pass


_MODEL = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def get_model():
    """Load the YOLO fire/smoke model once and return it."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            path = Path(settings.FIRE_SMOKE_MODEL_PATH)
            if not path.is_file():
                raise ModelNotAvailable(f"Model weights not found at {path}")
            from ultralytics import YOLO

            _MODEL = YOLO(str(path))
            logger.info(f"Fire/smoke model loaded from {path}")
    return _MODEL


# ---------------------------------------------------------------------------
# Public API: stop / start / restart
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# start rtsp polling for all configured CCTV cameras
# ---------------------------------------------------------------------------

def start_rtsp_polling():
    """Start one polling thread for every configured CCTV camera."""

    close_old_connections()

    try:
        cameras = CCTVDevice.objects.all()

        logger.info(f"Starting RTSP polling for {cameras.count()} CCTV camera(s).")

        for camera in cameras:
            interval = camera.snapshot_interval or 30

            start_camera_polling(device_id=camera.pk, poll_interval=interval)

    except Exception as exc:
        logger.error(f"Error starting RTSP polling: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# stop rtsp polling
# ---------------------------------------------------------------------------

def stop_rtsp_polling(wait: bool = True, timeout: float = 10.0):
    """Stop all RTSP camera polling threads (mirrors start_rtsp_polling)."""
    logger.info("Stopping all RTSP camera polling threads...")

    stop_all_camera_polling(wait=wait, timeout=timeout)

    logger.info("All RTSP camera polling threads stopped.")


# ---------------------------------------------------------------------------
# stop rtsp polling for all configured CCTV cameras
# ---------------------------------------------------------------------------

def stop_all_camera_polling(wait: bool = True, timeout: float = 10.0):
    """
    Signal ALL active camera polling threads to stop.

    Args:
        wait:    If True, wait (join) for every camera polling thread.
        timeout: Per-thread join timeout in seconds.
    """

    # Signal every polling loop to stop
    with _STOP_EVENTS_MUTEX:
        for device_id, event in _STOP_EVENTS.items():
            event.set()
            logger.info(f"[device_id={device_id}] Stop signal sent.")

    # Optionally wait for all threads to terminate
    if wait:
        with _THREADS_MUTEX:
            threads_snapshot = dict(_THREADS)

        for device_id, thread in threads_snapshot.items():
            if thread.is_alive():
                thread.join(timeout=timeout)

                if thread.is_alive():
                    logger.warning(
                        f"[device_id={device_id}] Camera polling thread did not stop within {timeout}s."
                    )
                else:
                    logger.info(f"[device_id={device_id}] Camera polling thread stopped cleanly.")


# ---------------------------------------------------------------------------
# restart rtsp polling for all configured CCTV cameras
# ---------------------------------------------------------------------------

def restart_rtsp_polling():
    """Stop all polling threads, then re-read the cameras from the database and respawn."""

    logger.info("Restarting all RTSP camera polling threads...")

    stop_rtsp_polling(wait=True, timeout=10.0)

    start_rtsp_polling()

    logger.info("All RTSP camera polling threads restarted.")


# ---------------------------------------------------------------------------
# starts a thread for each CCTV camera to poll for snapshots
# ---------------------------------------------------------------------------

def start_camera_polling(device_id: int, poll_interval: int = DEFAULT_POLL_INTERVAL):
    """
    Spawn a fresh polling thread for one CCTV camera.

    If a polling thread for this camera is already alive, this call is a no-op.
    """

    # Check whether a polling thread already exists
    with _THREADS_MUTEX:
        existing = _THREADS.get(device_id)
        if existing and existing.is_alive():
            logger.info(
                f"[device_id={device_id}] Camera polling thread already running — skipping start."
            )
            return

    try:
        device = CCTVDevice.objects.get(pk=device_id)

    except CCTVDevice.DoesNotExist:
        logger.error(f"[device_id={device_id}] Device not found — cannot start camera polling.")
        return

    # Make sure the stop event is cleared before the thread starts
    stop_event = _get_stop_event(device_id)
    stop_event.clear()

    # Create the polling thread
    thread = threading.Thread(
        target=_camera_polling_loop,
        args=(device_id, poll_interval),
        name=f"rtsp-{device.name}",
        daemon=True,
    )

    # Register the thread
    with _THREADS_MUTEX:
        _THREADS[device_id] = thread

    # Start the thread
    thread.start()

    logger.info(f"[{device.name}] Camera polling started.")


def stop_camera_polling(device_id: int, wait: bool = True, timeout: float = 10.0):
    """
    Signal the polling thread for the specified CCTV camera to stop
    after its current polling cycle.

    Args:
        device_id: Primary key of the CCTV device to stop.
        wait:      If True, block until the polling thread exits (up to *timeout* seconds).
        timeout:   Maximum seconds to wait for the thread to exit.
    """

    # Signal the polling loop to stop
    event = _get_stop_event(device_id)
    event.set()

    logger.info(f"[device_id={device_id}] Camera stop signal sent.")

    # Wait for the polling thread to exit
    if wait:
        with _THREADS_MUTEX:
            thread = _THREADS.get(device_id)

        if thread and thread.is_alive():
            thread.join(timeout=timeout)

            if thread.is_alive():
                logger.warning(
                    f"[device_id={device_id}] Camera polling thread did not stop within {timeout}s."
                )
            else:
                logger.info(f"[device_id={device_id}] Camera polling thread stopped cleanly.")


def restart_camera_polling(device_id: int, poll_interval: int = DEFAULT_POLL_INTERVAL):
    """
    Stop the existing camera polling thread (if any) and spawn a fresh one.

    Blocks until the old thread exits (max 10 s) before starting the new thread,
    so two threads never poll the same CCTV camera simultaneously.
    """

    logger.info(f"[device_id={device_id}] Restarting camera polling thread...")

    stop_camera_polling(device_id, wait=True, timeout=10.0)

    start_camera_polling(device_id, poll_interval)


# ---------------------------------------------------------------------------
# Per-camera polling loop (daemon thread)
# ---------------------------------------------------------------------------

def _camera_polling_loop(device_id: int, poll_interval: int):

    logger.info(f"[device_id={device_id}] Polling loop started (interval={poll_interval}s).")

    stop_event = _get_stop_event(device_id)

    while not stop_event.is_set():

        close_old_connections()

        try:
            camera = CCTVDevice.objects.get(pk=device_id)
            poll_and_process(camera)

        except CCTVDevice.DoesNotExist:
            logger.info(f"[device_id={device_id}] Device was deleted; stopping its polling loop.")
            break

        except Exception as exc:
            logger.error(f"[device_id={device_id}] Polling cycle failed: {exc}", exc_info=True)

        stop_event.wait(timeout=poll_interval)

    logger.info(f"[device_id={device_id}] Polling loop stopped.")

    _temporal_states.pop(device_id, None)

    with _THREADS_MUTEX:
        _THREADS.pop(device_id, None)


def poll_and_process(camera):
    """
    Perform one complete polling cycle for a single CCTV camera.

    Workflow:
        1. Capture one frame from the RTSP stream.
        2. Save the snapshot to the media folder.
        3. Run fire/smoke detection.
        4. Save the prediction to ModelPrediction.
        5. Temporal verification, then process the alarm if it passed.
    """

    try:
        get_model()
    except ModelNotAvailable as exc:
        logger.error(f"[{camera.name}] {exc}. Place your weights there to start predicting.")
        return

    try:
        # ------------------------------------------------------------------
        # 1. Capture one snapshot from the RTSP stream
        # ------------------------------------------------------------------
        frame = capture_snapshot(camera)

        if frame is not None:
            camera.status = "active"
            camera.save(update_fields=["status"])
        else:
            camera.status = "inactive"
            camera.save(update_fields=["status"])
        if frame is None:
            logger.warning(f"[{camera.name}] Snapshot capture failed.")
            return

        # ------------------------------------------------------------------
        # 2. Save snapshot to media folder
        # ------------------------------------------------------------------
        timestamp_ms = _now_ms()

        snapshot_path = save_snapshot(camera=camera, frame=frame, timestamp=timestamp_ms)

        # ------------------------------------------------------------------
        # 3. Fire Detection
        # ------------------------------------------------------------------
        detection = fire_detection(frame)

        logger.info(f"[{camera.name}] YOLO detection result | detection={detection}")

        # ------------------------------------------------------------------
        # 4. Save prediction locally
        # ------------------------------------------------------------------
        analysis = save_camera_analysis(
            camera=camera,
            timestamp=timestamp_ms,
            snapshot_path=snapshot_path,
            detection=detection,
        )

        if not detection:
            logger.info(f"[{camera.name}] No fire/smoke detected.")
            # Nothing detected: a gap breaks the consecutive sequence.
            _temporal_states.pop(camera.id, None)
            return

        detection_type = detection.get("class_name", "").lower()
        confidence = detection.get("confidence", 0.0)

        if detection_type not in ("fire", "smoke"):
            logger.info(f"[{camera.name}] Ignoring unsupported detection: {detection_type}")
            return

        alarm = AlarmRule.objects.filter(
            device=camera,
            parameter=detection_type,
            is_enabled=True,
        ).first()

        if alarm is None:
            logger.warning(f"[{camera.name}] No enabled alarm rule found for {detection_type}.")
            return

        # ---------------------------------------------------------
        # Temporal verification
        # ---------------------------------------------------------

        logger.info(
            f"[{camera.name}] Calling temporal verification | "
            f"type={detection_type} | confidence={confidence}"
        )

        temporal_result = temporal_verification(camera=camera, detection=detection, alarm=alarm)

        logger.info(f"[{camera.name}] Temporal verification returned | result={temporal_result}")

        if not temporal_result["verified"]:
            logger.info(
                f"[{camera.name}] Detection not temporally verified. "
                f"Continuing to next snapshot."
            )
            return

        # ---------------------------------------------------------
        # Temporal verification succeeded
        # ---------------------------------------------------------

        logger.info(
            f"[{camera.name}] Temporal verification successful | "
            f"type={temporal_result['detection_type']}"
        )

        # ------------------------------------------------------------------
        # 5. Process alarm
        # ------------------------------------------------------------------
        process_camera_alarm(
            camera=camera,
            analysis=analysis,
            detection=detection,
            verified_detection_type=temporal_result["detection_type"],
        )

    except Exception as exc:
        logger.error(f"[{camera.name}] Camera polling failed: {exc}", exc_info=True)


def capture_snapshot(camera):
    """
    Connect to the RTSP camera, capture one frame, release the connection,
    and return the captured frame.

    Returns:
        numpy.ndarray : Captured frame on success.
        None          : If connection or frame capture fails.
    """

    capture = None

    try:
        # -------------------------------------------------------------
        # 1. Read the RTSP URL
        # -------------------------------------------------------------
        snapshot_url = camera.rtsp_url

        if not snapshot_url:
            logger.error(f"[{camera.name}] RTSP URL is not configured.")
            return None

        logger.info(f"[{camera.name}] Connecting to RTSP stream...")

        # -------------------------------------------------------------
        # 2. Open RTSP Connection
        # -------------------------------------------------------------
        capture = cv2.VideoCapture(snapshot_url)

        if not capture.isOpened():
            logger.error(f"[{camera.name}] Failed to open RTSP stream.")
            return None

        logger.info(f"[{camera.name}] RTSP connection established.")

        # -------------------------------------------------------------
        # 3. Read One Frame
        # -------------------------------------------------------------
        success, frame = capture.read()

        if not success or frame is None:
            logger.error(f"[{camera.name}] Failed to capture snapshot.")
            return None

        logger.info(f"[{camera.name}] Snapshot captured successfully.")

        # -------------------------------------------------------------
        # 4. Return Frame
        # -------------------------------------------------------------
        return frame

    except Exception as exc:
        logger.error(f"[{camera.name}] Error while capturing snapshot: {exc}", exc_info=True)
        return None

    finally:
        # -------------------------------------------------------------
        # 5. Always release the RTSP connection
        # -------------------------------------------------------------
        if capture is not None:
            capture.release()
            logger.info(f"[{camera.name}] RTSP connection released.")


def camera_media_subdir(camera) -> str:
    """Folder (relative to MEDIA_ROOT) that holds a camera's snapshots."""
    return os.path.join("cctv", f"{slugify(camera.name) or 'camera'}-{camera.pk}")


def save_snapshot(camera, frame, timestamp):
    """
    Save one CCTV snapshot under MEDIA_ROOT.

    Args:
        camera (CCTVDevice): CCTV device instance.
        frame (numpy.ndarray): Captured image.
        timestamp (int): Unix timestamp in milliseconds.

    Returns:
        str : Snapshot path relative to MEDIA_ROOT (e.g. cctv/lobby-1/lobby_1_1754034567000.jpg).
        None: If saving fails.
    """

    try:
        # -------------------------------------------------------------
        # Camera-specific directory
        # -------------------------------------------------------------
        relative_dir = camera_media_subdir(camera)
        media_dir = os.path.join(str(settings.MEDIA_ROOT), relative_dir)

        os.makedirs(media_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Generate snapshot filename: <name>_<sequence>_<timestamp>.jpg
        # -------------------------------------------------------------
        existing_files = [f for f in os.listdir(media_dir) if f.lower().endswith(".jpg")]

        sequence = len(existing_files) + 1

        filename = f"{slugify(camera.name) or 'camera'}_{sequence}_{timestamp}.jpg"

        absolute_path = os.path.join(media_dir, filename)

        # -------------------------------------------------------------
        # Save image
        # -------------------------------------------------------------
        success = cv2.imwrite(absolute_path, frame)

        if not success:
            logger.error(f"[{camera.name}] Failed to save snapshot.")
            return None

        logger.info(f"[{camera.name}] Snapshot saved -> {absolute_path}")

        return os.path.join(relative_dir, filename).replace(os.sep, "/")

    except Exception as exc:
        logger.error(f"[{camera.name}] Snapshot save failed: {exc}", exc_info=True)
        return None


def save_camera_analysis(camera, timestamp, snapshot_path, detection=None):
    """
    Save the CCTV analysis result to the local database (ModelPrediction).

    Args:
        camera: CCTVDevice instance.
        timestamp: Unix timestamp in milliseconds (kept for log parity).
        snapshot_path: Path to the captured snapshot (relative to MEDIA_ROOT).
        detection: YOLO detection dictionary or None.

    Returns:
        ModelPrediction instance if successful, otherwise None.
    """

    try:
        if detection is None:
            detection_type = "none"
            detection_confidence = None

        else:
            detection_type = detection["class_name"].lower()
            detection_confidence = detection["confidence"]

            # Ensure only supported values reach the database
            if detection_type not in {"none", "smoke", "fire"}:
                logger.warning(f"[{camera.name}] Unknown detection type: {detection_type}")
                detection_type = "none"
                detection_confidence = None

        analysis = ModelPrediction.objects.create(
            device=camera,
            snapshot_path=snapshot_path or "",
            alarm_type=detection_type,
            confidence=detection_confidence,
        )

        logger.info(
            f"[{camera.name}] ModelPrediction saved | id={analysis.id} | "
            f"detection={detection_type} | confidence={detection_confidence}"
        )

        return analysis

    except Exception as exc:
        logger.error(f"[{camera.name}] Failed to save CCTV analysis: {exc}")
        return None


def process_camera_alarm(camera, analysis, detection, verified_detection_type):
    """
    Raise the alarm for a detection that passed temporal verification.

    The alarm is recorded by flagging the ModelPrediction row (alarm_raised=True).
    This is the single place to plug in notifications (email, webhook, MQTT, ...).
    """

    detection_type = (verified_detection_type or "").lower()

    if detection_type not in {"fire", "smoke"}:
        logger.warning(f"[{camera.name}] Unsupported detection type: {detection_type}")
        return

    try:
        confidence = float(detection["confidence"])
    except (KeyError, TypeError, ValueError):
        logger.warning(f"[{camera.name}] Invalid confidence value in detection: {detection}")
        return

    if analysis is None:
        logger.error(f"[{camera.name}] Prediction row missing; cannot record {detection_type} alarm.")
        return

    analysis.alarm_raised = True
    analysis.save(update_fields=["alarm_raised"])

    logger.warning(
        f"[{camera.name}] {detection_type.upper()} ALARM RAISED | "
        f"confidence={confidence:.3f} | prediction_id={analysis.id}"
    )


def fire_detection(frame):
    """
    Run YOLO fire/smoke detection on a CCTV frame.

    Returns:
        dict | None:
            Detected class information, or None when no
            fire/smoke is detected.
    """

    model = get_model()

    with _INFERENCE_LOCK:  # one shared model, many camera threads
        results = model.predict(
            source=frame,
            conf=settings.FIRE_SMOKE_MODEL_CONF,
            iou=settings.FIRE_SMOKE_MODEL_IOU,
            imgsz=settings.FIRE_SMOKE_MODEL_IMGSZ,
            verbose=False,
        )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # For now, use the highest-confidence detection.
        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))

        class_id = int(best_box.cls[0])
        confidence = float(best_box.conf[0])
        class_name = model.names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
        }

    return None


def temporal_verification(camera, detection, alarm):
    """
    Verify whether the current fire/smoke detection persists across
    the configured number of consecutive captured snapshots.

    A detection is considered positive only when:
        1. It is Fire or Smoke.
        2. Its confidence is greater than or equal to the alarm threshold.

    The same detection type must appear in consecutive snapshots.

    Returns:
        {
            "verified": True/False,
            "detection_type": "fire"/"smoke"/None
        }
    """

    # ---------------------------------------------------------
    # 1. Get temporal frame configuration
    # ---------------------------------------------------------

    temporal_frames = getattr(alarm, "temporal_frames", DEFAULT_TEMPORAL_FRAMES)

    # Defensive fallback
    if not isinstance(temporal_frames, int) or temporal_frames <= 0:
        temporal_frames = DEFAULT_TEMPORAL_FRAMES

    # ---------------------------------------------------------
    # 2. Get confidence threshold
    # ---------------------------------------------------------

    threshold = alarm.threshold

    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # ---------------------------------------------------------
    # 3. Validate detection
    # ---------------------------------------------------------

    if not detection:
        detection_type = None
        confidence = 0.0
    else:
        detection_type = detection.get("class_name")
        confidence = detection.get("confidence", 0.0)

    # Normalize detection type
    if detection_type:
        detection_type = detection_type.lower()

    # Only Fire and Smoke are valid temporal detections
    if detection_type not in ("fire", "smoke"):
        detection_type = None

    # Check confidence threshold
    if detection_type is None or confidence < threshold:

        # Current snapshot is not a valid positive detection.
        # Reset the temporal sequence for this camera.
        _temporal_states.pop(camera.id, None)

        logger.info(
            f"[{camera.name}] Temporal verification reset | "
            f"detection={detection_type} | "
            f"confidence={confidence:.3f} | "
            f"threshold={threshold}"
        )

        return {"verified": False, "detection_type": None}

    # ---------------------------------------------------------
    # 4. Get current temporal state
    # ---------------------------------------------------------

    state = _temporal_states.get(camera.id)

    # ---------------------------------------------------------
    # 5. Start a new sequence
    # ---------------------------------------------------------

    if state is None:

        _temporal_states[camera.id] = {"detection_type": detection_type, "count": 1}

        logger.info(
            f"[{camera.name}] Temporal verification started | "
            f"type={detection_type} | count=1/{temporal_frames}"
        )

        if temporal_frames == 1:
            _temporal_states.pop(camera.id, None)

        return {
            "verified": temporal_frames == 1,
            "detection_type": detection_type if temporal_frames == 1 else None,
        }

    # ---------------------------------------------------------
    # 6. Check whether detection type is consistent
    # ---------------------------------------------------------

    if state["detection_type"] != detection_type:

        # Fire -> Smoke or Smoke -> Fire.
        # Start a new sequence using the current detection.
        _temporal_states[camera.id] = {"detection_type": detection_type, "count": 1}

        logger.info(
            f"[{camera.name}] Temporal verification sequence changed | "
            f"previous={state['detection_type']} | "
            f"current={detection_type} | "
            f"count=1/{temporal_frames}"
        )

        return {"verified": False, "detection_type": None}

    # ---------------------------------------------------------
    # 7. Same detection type -> increment count
    # ---------------------------------------------------------

    state["count"] += 1

    logger.info(
        f"[{camera.name}] Temporal verification | "
        f"type={detection_type} | count={state['count']}/{temporal_frames}"
    )

    # ---------------------------------------------------------
    # 8. Check whether temporal verification succeeded
    # ---------------------------------------------------------

    if state["count"] >= temporal_frames:

        logger.info(
            f"[{camera.name}] Temporal verification PASSED | "
            f"type={detection_type} | frames={state['count']}"
        )

        # Reset state so the same continuous event does not
        # immediately trigger another verification.
        _temporal_states.pop(camera.id, None)

        return {"verified": True, "detection_type": detection_type}

    # ---------------------------------------------------------
    # 9. Verification still in progress
    # ---------------------------------------------------------

    return {"verified": False, "detection_type": None}
