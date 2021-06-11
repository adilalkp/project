from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models import (CharField, DateField, DecimalField, EmailField,
                              ForeignKey, ImageField, PositiveIntegerField,
                              SmallIntegerField, URLField, UUIDField,
                              OneToOneField, PositiveSmallIntegerField, DateTimeField, FileField)
import datetime
from django.utils import timezone

# Create your models here.
class Video(models.Model):
    owner = CharField(blank=True,max_length=100)
    videofile = FileField(upload_to='videosfolder', null=True, blank=True)
    job_code = CharField(blank=True, max_length=100)

class Job(models.Model):
    owner = CharField(blank=True, max_length=100)
    job_name = CharField(blank=True, max_length=100)
    job_code = CharField(blank=True, max_length=100)
    status = CharField(blank=True, max_length=100)
    created_on = DateTimeField(default=timezone.now(), blank=True)
    completed_on = DateTimeField(blank=True, null=True) # check if both completed and created are same, 
                                                                    #if same, not completed...if completed, 
                                                                    #update the complted_on from views


class VehicleRecord(models.Model):
    image = ImageField(null=True, blank=True)
    job_code = CharField(blank=True, max_length=100)
    license_plate = CharField(blank=True, null=True, max_length=100)
    colour = CharField(blank=True, null=True, max_length=100)
    vehicle_type = CharField(blank=True, null=True, max_length=100)
    vehicle_model = CharField(blank=True, null=True, max_length=100)
    vehicle_logo = CharField(blank=True, null=True, max_length=100)


