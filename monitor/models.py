from urllib.parse import urlsplit, urlunsplit

from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class CCTVDevice(models.Model):
    STATUS_CHOICES = [("active", "Active"), ("inactive", "Inactive"), ("unknown", "Unknown")]

    name = models.CharField(max_length=100, unique=True)
    rtsp_url = models.CharField(max_length=500)
    snapshot_interval = models.PositiveIntegerField(
        default=30, help_text="Seconds between snapshots."
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="unknown")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    @property
    def masked_rtsp_url(self) -> str:
        """RTSP URLs often embed credentials; never show the password in the UI."""
        parts = urlsplit(self.rtsp_url)
        if parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            netloc = f"{parts.username}:***@{host}" if parts.username else host
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return self.rtsp_url


class AlarmRule(models.Model):
    """One rule per alarm type per device (fire and smoke)."""

    PARAMETER_CHOICES = [("fire", "Fire"), ("smoke", "Smoke")]

    device = models.ForeignKey(CCTVDevice, on_delete=models.CASCADE, related_name="alarm_rules")
    parameter = models.CharField(max_length=10, choices=PARAMETER_CHOICES)
    threshold = models.FloatField(default=0.5, help_text="Minimum confidence score (0 to 1).")
    temporal_frames = models.PositiveSmallIntegerField(
        default=3, help_text="Consecutive snapshots that must agree before an alarm is raised."
    )
    is_enabled = models.BooleanField(default=True)

    class Meta:
        ordering = ["parameter"]
        constraints = [
            models.UniqueConstraint(fields=["device", "parameter"], name="unique_rule_per_device_parameter")
        ]

    def __str__(self) -> str:
        return f"{self.device.name}: {self.parameter} >= {self.threshold}"


class ModelPrediction(models.Model):
    """One live inference (one snapshot) from the fire/smoke model."""

    TYPE_CHOICES = [("fire", "Fire"), ("smoke", "Smoke"), ("none", "None")]

    device = models.ForeignKey(CCTVDevice, on_delete=models.CASCADE, related_name="predictions")
    alarm_type = models.CharField(max_length=10, choices=TYPE_CHOICES, default="none")
    confidence = models.FloatField(null=True, blank=True)
    snapshot_path = models.CharField(max_length=500, blank=True, help_text="Path relative to MEDIA_ROOT.")
    alarm_raised = models.BooleanField(default=False, help_text="True when temporal verification passed.")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["device", "-id"])]

    def __str__(self) -> str:
        return f"{self.device.name} {self.alarm_type} {self.confidence}"


@receiver(post_save, sender=CCTVDevice)
def ensure_alarm_rules(sender, instance, created, **kwargs):
    """Every device always has exactly one fire rule and one smoke rule."""
    if created:
        for parameter, _ in AlarmRule.PARAMETER_CHOICES:
            AlarmRule.objects.get_or_create(device=instance, parameter=parameter)
