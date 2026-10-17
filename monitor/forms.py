from urllib.parse import urlsplit

from django import forms
from django.conf import settings

from .models import AlarmRule, CCTVDevice


class DeviceForm(forms.ModelForm):
    class Meta:
        model = CCTVDevice
        fields = ["name", "rtsp_url", "snapshot_interval"]
        labels = {
            "name": "CCTV name",
            "rtsp_url": "RTSP URL",
            "snapshot_interval": "Snapshot interval (seconds)",
        }
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Warehouse entrance", "autocomplete": "off"}),
            "rtsp_url": forms.TextInput(
                attrs={"placeholder": "rtsp://user:password@192.168.1.20:554/stream1", "autocomplete": "off"}
            ),
            "snapshot_interval": forms.NumberInput(attrs={"min": 1}),
        }

    def clean_name(self):
        return self.cleaned_data["name"].strip()

    def clean_rtsp_url(self):
        url = self.cleaned_data["rtsp_url"].strip()
        parts = urlsplit(url)
        if parts.scheme not in ("rtsp", "rtsps") or not parts.hostname:
            raise forms.ValidationError("Enter a full RTSP address, for example rtsp://192.168.1.20:554/stream1.")
        return url

    def clean_snapshot_interval(self):
        value = self.cleaned_data["snapshot_interval"]
        if value < 1:
            raise forms.ValidationError("The interval must be at least 1 second.")
        return value


class AlarmRuleForm(forms.ModelForm):
    class Meta:
        model = AlarmRule
        fields = ["threshold", "temporal_frames", "is_enabled"]
        labels = {
            "threshold": "Confidence threshold",
            "temporal_frames": "Frames to verify",
            "is_enabled": "Rule enabled",
        }
        widgets = {
            "threshold": forms.NumberInput(attrs={"step": "0.01", "min": 0, "max": 1}),
            "temporal_frames": forms.NumberInput(attrs={"min": 1, "max": 100}),
        }

    def clean_threshold(self):
        value = self.cleaned_data["threshold"]
        floor = settings.FIRE_SMOKE_MODEL_CONF
        if not floor <= value <= 1:
            raise forms.ValidationError(
                f"Enter a score between {floor:.2f} and 1. The model never reports detections below {floor:.2f}."
            )
        return value

    def clean_temporal_frames(self):
        value = self.cleaned_data["temporal_frames"]
        if not 1 <= value <= 100:
            raise forms.ValidationError("Enter a number of frames between 1 and 100.")
        return value
