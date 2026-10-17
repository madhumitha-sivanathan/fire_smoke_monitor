from django.contrib import admin

from .models import AlarmRule, CCTVDevice, ModelPrediction

admin.site.register(CCTVDevice)
admin.site.register(AlarmRule)
admin.site.register(ModelPrediction)
