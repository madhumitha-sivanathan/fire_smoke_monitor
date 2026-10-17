import shutil
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from django.test import TestCase, override_settings
from django.urls import reverse

from . import rtsp
from .forms import AlarmRuleForm, DeviceForm
from .models import AlarmRule, CCTVDevice, ModelPrediction

TMP_MEDIA = tempfile.mkdtemp(prefix="firewatch-media-")


def detection(name="Fire", conf=0.9):
    return {"class_id": 0, "class_name": name, "confidence": conf}


class ModelTests(TestCase):
    def test_creating_a_device_creates_fire_and_smoke_rules(self):
        d = CCTVDevice.objects.create(name="Lobby", rtsp_url="rtsp://10.0.0.5/live")
        self.assertEqual(set(d.alarm_rules.values_list("parameter", flat=True)), {"fire", "smoke"})

    def test_rtsp_password_is_masked(self):
        d = CCTVDevice(name="x", rtsp_url="rtsp://admin:s3cret@10.0.0.5:554/live")
        self.assertEqual(d.masked_rtsp_url, "rtsp://admin:***@10.0.0.5:554/live")
        self.assertNotIn("s3cret", d.masked_rtsp_url)


class FormTests(TestCase):
    def test_rejects_non_rtsp_url(self):
        f = DeviceForm({"name": "A", "rtsp_url": "http://example.com", "snapshot_interval": 10})
        self.assertFalse(f.is_valid())
        self.assertIn("rtsp_url", f.errors)

    def test_accepts_valid_device(self):
        f = DeviceForm({"name": "A", "rtsp_url": "rtsp://10.0.0.5:554/live", "snapshot_interval": 10})
        self.assertTrue(f.is_valid(), f.errors)

    def test_threshold_must_be_at_or_above_model_floor(self):
        d = CCTVDevice.objects.create(name="A", rtsp_url="rtsp://h/x")
        rule = d.alarm_rules.get(parameter="fire")
        bad = AlarmRuleForm({"threshold": 0.05, "temporal_frames": 3, "is_enabled": True}, instance=rule)
        self.assertFalse(bad.is_valid())
        good = AlarmRuleForm({"threshold": 0.6, "temporal_frames": 5, "is_enabled": True}, instance=rule)
        self.assertTrue(good.is_valid(), good.errors)


class TemporalVerificationTests(TestCase):
    def setUp(self):
        rtsp._temporal_states.clear()
        self.cam = CCTVDevice.objects.create(name="Cam", rtsp_url="rtsp://h/x")
        self.rule = self.cam.alarm_rules.get(parameter="fire")
        self.rule.threshold = 0.5
        self.rule.temporal_frames = 3
        self.rule.save()

    def verify(self, det):
        return rtsp.temporal_verification(self.cam, det, self.rule)["verified"]

    def test_passes_after_configured_consecutive_frames(self):
        self.assertFalse(self.verify(detection(conf=0.8)))
        self.assertFalse(self.verify(detection(conf=0.8)))
        self.assertTrue(self.verify(detection(conf=0.8)))

    def test_frame_count_is_configurable(self):
        self.rule.temporal_frames = 5
        results = [self.verify(detection(conf=0.8)) for _ in range(5)]
        self.assertEqual(results, [False, False, False, False, True])

    def test_single_frame_rule_fires_immediately(self):
        self.rule.temporal_frames = 1
        self.assertTrue(self.verify(detection(conf=0.8)))

    def test_below_threshold_resets_sequence(self):
        self.verify(detection(conf=0.8))
        self.verify(detection(conf=0.8))
        self.assertFalse(self.verify(detection(conf=0.3)))  # reset
        self.assertFalse(self.verify(detection(conf=0.8)))
        self.assertFalse(self.verify(detection(conf=0.8)))
        self.assertTrue(self.verify(detection(conf=0.8)))

    def test_type_change_restarts_sequence(self):
        self.verify(detection("Fire", 0.8))
        self.verify(detection("Fire", 0.8))
        self.assertFalse(self.verify(detection("Smoke", 0.8)))
        self.assertEqual(rtsp.get_temporal_state(self.cam.id), {"detection_type": "smoke", "count": 1})

    def test_threshold_boundary_counts(self):
        self.rule.temporal_frames = 1
        self.assertTrue(self.verify(detection(conf=0.5)))


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class PollingTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        rtsp._temporal_states.clear()
        self.cam = CCTVDevice.objects.create(name="Dock 1", rtsp_url="rtsp://h/x")
        AlarmRule.objects.filter(device=self.cam).update(temporal_frames=2, threshold=0.5)
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def poll(self, det, frame="default"):
        frame = self.frame if frame == "default" else frame
        with mock.patch.object(rtsp, "get_model"), \
             mock.patch.object(rtsp, "capture_snapshot", return_value=frame), \
             mock.patch.object(rtsp, "fire_detection", return_value=det):
            rtsp.poll_and_process(CCTVDevice.objects.get(pk=self.cam.pk))

    def test_alarm_only_after_temporal_verification(self):
        self.poll(detection("Fire", 0.9))
        self.assertFalse(ModelPrediction.objects.filter(alarm_raised=True).exists())
        self.poll(detection("Fire", 0.9))
        raised = ModelPrediction.objects.filter(alarm_raised=True)
        self.assertEqual(raised.count(), 1)
        self.assertEqual(raised.first().alarm_type, "fire")

    def test_every_snapshot_is_stored_with_a_media_relative_path(self):
        self.poll(None)
        p = ModelPrediction.objects.get()
        self.assertEqual(p.alarm_type, "none")
        self.assertTrue(p.snapshot_path.startswith("cctv/dock-1-"))
        self.assertTrue((Path(TMP_MEDIA) / p.snapshot_path).is_file())

    def test_clear_frame_breaks_the_consecutive_sequence(self):
        self.poll(detection("Smoke", 0.9))
        self.poll(None)
        self.poll(detection("Smoke", 0.9))
        self.assertFalse(ModelPrediction.objects.filter(alarm_raised=True).exists())

    def test_disabled_rule_never_alarms(self):
        AlarmRule.objects.filter(device=self.cam, parameter="fire").update(is_enabled=False)
        for _ in range(3):
            self.poll(detection("Fire", 0.95))
        self.assertFalse(ModelPrediction.objects.filter(alarm_raised=True).exists())

    def test_unreachable_camera_marks_inactive_and_stores_nothing(self):
        self.poll(None, frame=None)
        self.cam.refresh_from_db()
        self.assertEqual(self.cam.status, "inactive")
        self.assertEqual(ModelPrediction.objects.count(), 0)

    def test_missing_model_skips_cycle_without_crashing(self):
        with mock.patch.object(rtsp, "get_model", side_effect=rtsp.ModelNotAvailable("missing")), \
             mock.patch.object(rtsp, "capture_snapshot") as cap:
            rtsp.poll_and_process(self.cam)
        cap.assert_not_called()

    def test_capture_snapshot_reads_a_frame(self):
        path = str(Path(TMP_MEDIA) / "clip.avi")
        Path(TMP_MEDIA).mkdir(exist_ok=True)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        for _ in range(5):
            writer.write(self.frame)
        writer.release()
        self.cam.rtsp_url = path  # OpenCV opens files the same way as RTSP URLs
        frame = rtsp.capture_snapshot(self.cam)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (48, 64, 3))


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class ViewTests(TestCase):
    def test_dashboard_lists_devices_and_warns_when_model_missing(self):
        CCTVDevice.objects.create(name="Gate", rtsp_url="rtsp://h/x")
        with override_settings(FIRE_SMOKE_MODEL_PATH=Path(TMP_MEDIA) / "nope.pt"):
            html = self.client.get(reverse("dashboard")).content.decode()
        self.assertIn("Gate", html)
        self.assertIn("Add device", html)
        self.assertIn("Model file not found", html)

    def test_add_device_starts_polling_and_creates_rules(self):
        with mock.patch("monitor.views.rtsp.start_camera_polling") as start:
            res = self.client.post(
                reverse("device_create"),
                {"name": "Yard", "rtsp_url": "rtsp://10.1.1.1/s", "snapshot_interval": 15},
            )
        d = CCTVDevice.objects.get(name="Yard")
        self.assertRedirects(res, f"/?device={d.pk}", fetch_redirect_response=False)
        start.assert_called_once_with(d.pk, 15)
        self.assertEqual(d.alarm_rules.count(), 2)

    def test_invalid_device_rerenders_with_errors(self):
        res = self.client.post(reverse("device_create"), {"name": "", "rtsp_url": "nope", "snapshot_interval": 0})
        self.assertEqual(res.status_code, 400)
        self.assertContains(res, "Enter a full RTSP address", status_code=400)
        self.assertEqual(CCTVDevice.objects.count(), 0)

    def test_configure_rules_saves_both_rules(self):
        d = CCTVDevice.objects.create(name="Gate", rtsp_url="rtsp://h/x")
        res = self.client.post(
            reverse("configure_rules", args=[d.pk]),
            {
                "fire-threshold": "0.7", "fire-temporal_frames": "4", "fire-is_enabled": "on",
                "smoke-threshold": "0.45", "smoke-temporal_frames": "6",
            },
        )
        self.assertEqual(res.status_code, 302)
        fire, smoke = d.alarm_rules.get(parameter="fire"), d.alarm_rules.get(parameter="smoke")
        self.assertEqual((fire.threshold, fire.temporal_frames, fire.is_enabled), (0.7, 4, True))
        self.assertEqual((smoke.threshold, smoke.temporal_frames, smoke.is_enabled), (0.45, 6, False))

    def test_delete_removes_device_and_snapshots(self):
        d = CCTVDevice.objects.create(name="Gate", rtsp_url="rtsp://h/x")
        folder = Path(TMP_MEDIA) / rtsp.camera_media_subdir(d)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.jpg").write_bytes(b"x")
        with mock.patch("monitor.views.rtsp.stop_camera_polling") as stop:
            self.client.post(reverse("device_delete", args=[d.pk]))
        stop.assert_called_once()
        self.assertFalse(CCTVDevice.objects.exists())
        self.assertFalse(folder.exists())

    def test_live_and_history_api(self):
        d = CCTVDevice.objects.create(name="Gate", rtsp_url="rtsp://u:pw@h/x")
        for i in range(30):
            ModelPrediction.objects.create(
                device=d, alarm_type="fire" if i % 2 else "none",
                confidence=0.8 if i % 2 else None, snapshot_path=f"cctv/g/{i}.jpg",
            )
        live = self.client.get(reverse("api_live", args=[d.pk])).json()
        self.assertEqual(len(live["recent"]), 12)
        self.assertEqual(live["device"]["rtsp_url"], "rtsp://u:***@h/x")
        self.assertEqual(set(live["rules"]), {"fire", "smoke"})

        page1 = self.client.get(reverse("api_predictions", args=[d.pk]) + "?limit=10").json()
        self.assertEqual(len(page1["predictions"]), 10)
        self.assertTrue(all(p["alarm_type"] == "fire" for p in page1["predictions"]))
        page2 = self.client.get(
            reverse("api_predictions", args=[d.pk]) + f"?limit=10&before={page1['next_before']}"
        ).json()
        self.assertEqual(len(page2["predictions"]), 5)
        self.assertIsNone(page2["next_before"])
        everything = self.client.get(reverse("api_predictions", args=[d.pk]) + "?detections_only=0&limit=100").json()
        self.assertEqual(len(everything["predictions"]), 30)
