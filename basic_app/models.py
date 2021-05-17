from django.db import models
from django.db.models import (CharField, DateField, DecimalField, EmailField,
                              ForeignKey, ImageField, PositiveIntegerField,
                              SmallIntegerField, URLField, UUIDField,
                              OneToOneField, PositiveSmallIntegerField, DateTimeField)
# Create your models here.
class Image(models.Model):
    imagefile = ImageField(upload_to='imagesfolder', null=True, blank=True)
    