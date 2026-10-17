# FireWatch

A Django application that monitors CCTV cameras over RTSP and detects **fire** and **smoke** with a YOLO model.
Add cameras from the web UI, set a confidence threshold and a temporal-verification length for each of fire and
smoke, and watch live predictions and history for every camera.

## Where to put your model

Copy your trained weights to:

```
models/model.pt
```

That is the only file you need to add. (To use a different location, set the `FIRE_SMOKE_MODEL_PATH`
environment variable.) The model must have classes named `fire` and `smoke` (case-insensitive).
Until the file exists the UI still works and shows a notice, but no predictions run.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

If `YOLO("models/model.pt")` fails to load a YOLOv10 checkpoint, upgrade `ultralytics`
(`pip install -U ultralytics`) or install the fork the weights were trained with.

## Try it without a real camera

Use [MediaMTX](https://github.com/bluenviron/mediamtx) and FFmpeg to turn a video into an RTSP stream:

```bash
mediamtx                                   # starts an RTSP server on :8554
ffmpeg -re -stream_loop -1 -i fire_clip.mp4 -c:v libx264 -preset ultrafast \
       -tune zerolatency -an -f rtsp -rtsp_transport tcp rtsp://localhost:8554/camera1
```

In the UI choose **Add device** and enter:

| Field | Example |
|---|---|
| CCTV name | Demo camera |
| RTSP URL | `rtsp://localhost:8554/camera1` |
| Snapshot interval | `5` |

Click the camera in the list to open its live prediction card.

## Using the app

- **Add device**: name, RTSP URL and snapshot interval (seconds). A fire rule and a smoke rule are created
  automatically, and polling for that camera starts immediately.
- **Configure alarm rules**: per camera, one rule for fire and one for smoke. Each has a confidence threshold,
  the number of consecutive snapshots that must agree (temporal verification), and an on/off switch.
- **Live prediction card**: latest snapshot, current state (clear, verifying, alarm), a meter showing how many of
  the required snapshots have agreed so far, and the most recent predictions. It refreshes every 3 seconds.
- **History**: every stored prediction with confidence, whether an alarm was raised, and the snapshot.
  Use "Detections only" to hide clear snapshots.

## How detection and alarms work

Each camera has its own daemon thread. On every cycle it:

1. opens the RTSP stream, grabs one frame and releases the connection,
2. saves the snapshot under `media/cctv/<camera>/`,
3. runs the YOLO model and keeps the highest-confidence fire or smoke detection,
4. stores a `ModelPrediction` row,
5. runs **temporal verification**: the detection must be fire or smoke, at or above that rule's threshold, for the
   same class on N consecutive snapshots. A clear snapshot, a low-confidence one, or a change of class resets the
   count,
6. only when verification passes is the alarm raised (`alarm_raised=True` on that prediction).

Detection delay is roughly *frames to verify x snapshot interval*. With a 5 s interval and 3 frames, an alarm
appears after about 15 s.

`monitor/rtsp.py` holds all of this. `process_camera_alarm()` is the single place to plug in notifications
(email, webhook, MQTT).

## Data model

| Model | Purpose |
|---|---|
| `CCTVDevice` | name, RTSP URL, snapshot interval, status |
| `AlarmRule` | per device and per type (fire or smoke): threshold, frames to verify, enabled |
| `ModelPrediction` | per snapshot: device, alarm type (fire, smoke, none), confidence, snapshot path, alarm raised, time |

Data is stored in SQLite (`db.sqlite3`).

## Project layout

```
config/            Django settings and URLs
monitor/
  models.py        CCTVDevice, AlarmRule, ModelPrediction
  rtsp.py          polling threads, YOLO inference, temporal verification
  views.py         dashboard, rules page, JSON API
  forms.py         validation for devices and rules
  templates/ static/
models/model.pt    your weights (not committed)
media/             snapshots (not committed)
```

## Running tests

```bash
python manage.py test
```

## Deployment notes

`runserver` starts the camera threads automatically. Under another server (for example gunicorn), set
`FIREWATCH_START_POLLING=1` and run a **single worker process**, otherwise every worker would poll every camera.
Also set `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS`, and serve `media/` from your web server.
The app has no login; put it behind your own authentication before exposing it on a network.
