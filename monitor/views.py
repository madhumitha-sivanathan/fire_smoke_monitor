import shutil
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from . import rtsp
from .forms import AlarmRuleForm, DeviceForm
from .models import AlarmRule, CCTVDevice, ModelPrediction


def _model_ready() -> bool:
    return Path(settings.FIRE_SMOKE_MODEL_PATH).is_file()


def _render_dashboard(request, add_form, open_add_modal=False, status=200):
    selected = request.GET.get("device", "")
    return render(
        request,
        "monitor/dashboard.html",
        {
            "devices": CCTVDevice.objects.all(),
            "add_form": add_form,
            "open_add_modal": open_add_modal,
            "selected_id": selected if selected.isdigit() else "",
            "model_ready": _model_ready(),
            "model_path": settings.FIRE_SMOKE_MODEL_PATH,
        },
        status=status,
    )


def dashboard(request):
    return _render_dashboard(request, DeviceForm())


@require_POST
def device_create(request):
    form = DeviceForm(request.POST)
    if not form.is_valid():
        return _render_dashboard(request, form, open_add_modal=True, status=400)

    device = form.save()  # a post_save signal creates the fire and smoke rules
    rtsp.start_camera_polling(device.pk, device.snapshot_interval)
    messages.success(request, f"Added {device.name}. Review its alarm rules to set thresholds.")
    return redirect(f"/?device={device.pk}")


@require_POST
def device_delete(request, pk):
    device = get_object_or_404(CCTVDevice, pk=pk)
    name = device.name
    media_dir = Path(settings.MEDIA_ROOT) / rtsp.camera_media_subdir(device)

    rtsp.stop_camera_polling(device.pk, wait=False)
    device.delete()
    shutil.rmtree(media_dir, ignore_errors=True)

    messages.success(request, f"Removed {name} and its snapshots.")
    return redirect("dashboard")


def configure_rules(request, pk):
    device = get_object_or_404(CCTVDevice, pk=pk)
    rules = {r.parameter: r for r in device.alarm_rules.all()}
    # Defensive: guarantee both rules exist even for devices created outside the UI.
    for parameter, _ in AlarmRule.PARAMETER_CHOICES:
        if parameter not in rules:
            rules[parameter] = AlarmRule.objects.create(device=device, parameter=parameter)

    forms = {
        parameter: AlarmRuleForm(
            request.POST or None, instance=rules[parameter], prefix=parameter
        )
        for parameter in ("fire", "smoke")
    }

    if request.method == "POST":
        if all(f.is_valid() for f in forms.values()):
            for f in forms.values():
                f.save()
            messages.success(request, f"Alarm rules saved for {device.name}.")
            return redirect(f"/?device={device.pk}")
        status = 400
    else:
        status = 200

    return render(
        request,
        "monitor/rules.html",
        {"device": device, "forms": forms, "model_floor": settings.FIRE_SMOKE_MODEL_CONF},
        status=status,
    )


# ---------------------------------------------------------------------------
# JSON API used by the dashboard
# ---------------------------------------------------------------------------

def _prediction_json(p: ModelPrediction) -> dict:
    return {
        "id": p.id,
        "alarm_type": p.alarm_type,
        "confidence": p.confidence,
        "alarm_raised": p.alarm_raised,
        "snapshot_url": f"{settings.MEDIA_URL}{p.snapshot_path}" if p.snapshot_path else None,
        "created_at": p.created_at.isoformat(),
    }


@require_GET
def api_devices(request):
    return JsonResponse(
        {
            "devices": [
                {"id": d.id, "name": d.name, "status": d.status, "snapshot_interval": d.snapshot_interval}
                for d in CCTVDevice.objects.all()
            ]
        }
    )


@require_GET
def api_live(request, pk):
    device = get_object_or_404(CCTVDevice, pk=pk)
    rules = {r.parameter: r for r in device.alarm_rules.all()}
    recent = list(device.predictions.all()[:12])
    latest = recent[0] if recent else None

    verification = None
    state = rtsp.get_temporal_state(device.pk)
    if state and state["detection_type"] in rules:
        verification = {
            "type": state["detection_type"],
            "count": state["count"],
            "required": rules[state["detection_type"]].temporal_frames,
        }

    return JsonResponse(
        {
            "device": {
                "id": device.id,
                "name": device.name,
                "status": device.status,
                "snapshot_interval": device.snapshot_interval,
                "rtsp_url": device.masked_rtsp_url,
            },
            "model_ready": _model_ready(),
            "latest": _prediction_json(latest) if latest else None,
            "recent": [_prediction_json(p) for p in recent],
            "verification": verification,
            "rules": {
                key: {
                    "threshold": r.threshold,
                    "temporal_frames": r.temporal_frames,
                    "is_enabled": r.is_enabled,
                }
                for key, r in rules.items()
            },
        }
    )


@require_GET
def api_predictions(request, pk):
    """Prediction history, newest first, with cursor paging via ?before=<id>."""
    device = get_object_or_404(CCTVDevice, pk=pk)
    qs = device.predictions.all()
    if request.GET.get("detections_only", "1") == "1":
        qs = qs.exclude(alarm_type="none")
    before = request.GET.get("before", "")
    if before.isdigit():
        qs = qs.filter(id__lt=int(before))

    try:
        limit = min(max(int(request.GET.get("limit", 25)), 1), 100)
    except ValueError:
        limit = 25

    rows = list(qs[: limit + 1])
    has_more = len(rows) > limit
    rows = rows[:limit]
    return JsonResponse(
        {
            "predictions": [_prediction_json(p) for p in rows],
            "next_before": rows[-1].id if has_more and rows else None,
        }
    )
