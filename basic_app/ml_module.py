from .models import Video, Job, VehicleRecord
from pathlib import Path
import uuid
BASE_DIR = Path(__file__).resolve().parent.parent
from django.utils import timezone
import shutil
from celery import shared_task

#mail
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template


#ml imports
import cv2
import numpy as np
import os
import utils as ut
from os.path import splitext, basename
import glob
import cv2
from basic_app.ml.local_utils import detect_lp
from tensorflow.keras.models import model_from_json
import glob
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils
import easyocr


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        return model
    except Exception as e:
        print(e)

def get_plate(wpod_net,image_path, Dmax=608, Dmin=280):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def filt(thresh,ocr_result,regthresh):
  rect=thresh.shape[0]*thresh.shape[1]
  plate=[]
  for result in ocr_result:
    ln=np.sum(np.subtract(result[0][1],result[0][0]))
    h=np.sum(np.subtract(result[0][2],result[0][1]))
    if ln*h/rect>regthresh:
      plate.append(result[1])
  return plate



def capture_frames(username, job_code):
    video_path = str(BASE_DIR / 'media/videosfolder/{}-{}.mp4'.format(username, job_code)) #since cv2.VideoCapture doesn't 
                                                                            #accept POSIX path, we convert it into string
    cam = cv2.VideoCapture(video_path)
    currframe = 0
    try:
        if not os.path.exists('framesimages/{}'.format(username)):
            os.makedirs('framesimages/{}'.format(username))
        if not os.path.exists('framesimages/{}/{}'.format(username,job_code)):
            os.makedirs('framesimages/{}/{}'.format(username,job_code))
    except OSError as e:
        print (e)
    while(True):
        ret,frame = cam.read()
        if ret:
            cam.set(cv2.CAP_PROP_POS_MSEC,(currframe*1000))
            name = './framesimages/{}/{}/frame{}.jpg'.format(username, job_code, str(currframe)) 
            cv2.imwrite(name, frame)
            currframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()



def read_plates(username, job_code):
    capture_frames(username, job_code)
    wpod_net_path = BASE_DIR / "basic_app/ml/wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    reader=easyocr.Reader(['en'])
    plates_list=[]
    image_paths = glob.glob("framesimages/{}/{}/*.jpg".format(username, job_code))
    for i in range(len(image_paths)):
        try:
            LpImg,_ = get_plate(wpod_net, image_paths[i])
            impath=image_paths[i]
            for k in range(len(LpImg)):
                regthresh=0.7
                plate = cv2.convertScaleAbs(LpImg[k], alpha=(255.0))
                V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
                T = threshold_local(V, 29, offset=15, method="gaussian")
                thresh = (V > T).astype("uint8") * 255
                thresh = cv2.bitwise_not(thresh)
                thresh = imutils.resize(thresh, width=400)
                ocr_result=reader.readtext(thresh)
                c=filt(thresh,ocr_result,regthresh)
                cnts=''
                for i in c:
                    cnts=cnts+i
                chkplt = ""
                for character in cnts:
                    if character.isalnum():
                        chkplt += character

                if chkplt not in plates_list:
                    if chkplt == '':
                        pass
                    else:
                        plates_list.append(chkplt)
                    #accomodate colour code



        except Exception as e:
            print(e)
            pass
    for i in plates_list:
        print(".{}".format(i))
    return plates_list


@shared_task
def run_job(username, job_code, domain, user_email):
    #first read plates
    plates_list = read_plates(username, job_code)
    #call other ml modules

    #update job in Job
    # run some loop to iterate through sample cases, here it is list of license plates
    for plate in plates_list:
         record = VehicleRecord(job_code=job_code, license_plate=plate, colour="nil", vehicle_type="nil", vehicle_model="nil", vehicle_logo="nil")
         record.save()
    job_obj = Job.objects.filter(job_code=job_code).first()
    job_obj.status = "completed"
    job_obj.completed_on = timezone.now()
    job_obj.save()

    #send mail to user
    mail_subject = 'Vehicle Recognition Report'
    t = get_template('gmail_template.html')
    Context = {
        'domain': domain,
        'subheading': "Your vehi-scanner report is ready",
        'content': plates_list,
        'salutation': "Hi " + str(username) + " ,",
    }

    to_email = user_email
    message = EmailMultiAlternatives(subject=mail_subject, to=[to_email])
    message.attach_alternative(t.render(Context), "text/html")
    message.send()

    #clear ml waste(frames, video..)

    job_frames_location = BASE_DIR / "framesimages/{}/{}/".format(username,job_code)
    shutil.rmtree(job_frames_location)
    video_obj = Video.objects.filter(job_code=job_code).first()
    video_obj.delete()

    
