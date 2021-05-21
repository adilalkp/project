from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models import (CharField, DateField, DecimalField, EmailField,
                              ForeignKey, ImageField, PositiveIntegerField,
                              SmallIntegerField, URLField, UUIDField,
                              OneToOneField, PositiveSmallIntegerField, DateTimeField)
# Create your models here.
class Image(models.Model):
    owner = CharField(blank=True,max_length=100)
    imagefile = ImageField(upload_to='imagesfolder', null=True, blank=True)
    
