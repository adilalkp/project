from datetime import time
from basic_app.models import Video, Job, VehicleRecord
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
from django.core.files import File

# Commented out IPython magic to ensure Python compatibility.
# Import the necessary packages
import cv2
import os
import pathlib
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from webcolors import rgb_to_name
import webcolors
import easyocr
from basic_app.ml.local_utils import detect_lp
import utils as ut
from os.path import splitext, basename
from keras.models import model_from_json
import matplotlib
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec

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

def loadtypemodel():
  json_file = open(BASE_DIR / 'ml/type_recognition.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights(BASE_DIR / "ml/type_recognition_weight.h5")
  labels = LabelEncoder()
  labels.classes_ = np.load(BASE_DIR / 'ml/type_classes.npy')
  return model,labels

def frame_extraction(path, username, job_code):
  #Convert input video into frames(1 frame every second)
    capture = cv2.VideoCapture(path)
    count = 0
    current=0
    try:
        if not os.path.exists('framesimages/{}'.format(username)):
            os.makedirs('framesimages/{}'.format(username))
        if not os.path.exists('framesimages/{}/{}'.format(username,job_code)):
            os.makedirs('framesimages/{}/{}'.format(username,job_code))
    except OSError as e:
        print (e)
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            name = './framesimages/{}/{}/frame{}.jpg'.format(username, job_code, str(current))
            cv2.imwrite(name, frame)
            current=current+1
            count += 20
            capture.set(1, count)
        else:
            capture.release()
            break



# Use Tensorflow Object detection api for Vehicle detection from frames
def vehicle_detect(username, job_code):
  utils_ops.tf = tf.compat.v1
  tf.gfile = tf.io.gfile
  def load_model(model_name):
    model_file = model_name + '.tar.gz'
    model_dir = BASE_DIR / '{}/saved_model'.format(model_name)
    model = tf.saved_model.load(str(model_dir))
    return model
  labelpath = 'object_detection/data/mscoco_label_map.pbtxt'
  category_index = label_map_util.create_category_index_from_labelmap(labelpath, use_display_name=True)
  imgpath= glob.glob("framesimages/{}/{}/frame*.jpg".format(username, job_code))

  # Model used for detection is ssd+mobilenet
  model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
  detection_model = load_model('ssd_mobilenet_v1_coco_2017_11_17')
  def findboundingboxes(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                  for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict
  def extractvehicleregion(model, image_path,j, username, job_code):
    image_np = np.array(Image.open(image_path))
    output_dict = findboundingboxes(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    imgs=Image.fromarray(image_np)
    imginput = cv2.imread(image_path)
    img= cv2.cvtColor(imginput, cv2.COLOR_BGR2RGB)
    cv2.line(img, (0, 600),(1500,600),5)
    (w,h)=imgs.size
    checkclass=[2,3,4,6,8]
    for i in range(len(output_dict['detection_classes'])):
      if(output_dict['detection_classes'][i] in checkclass and output_dict['detection_scores'][i]>0.55):
        c=np.array(Image.open(image_path))
        x1=int(output_dict['detection_boxes'][i][1]*w)
        y1=int(output_dict['detection_boxes'][i][0]*h)
        x2=int(output_dict['detection_boxes'][i][3]*w)
        y2=int(output_dict['detection_boxes'][i][2]*h)
        y=max(y1,y2)
        if(y>(0.8*h) and y<(0.98*h)):
          #section for extracting timestamp
          path_string = image_path[-7:]
          if path_string[0] == 'e':
            timestamp = path_string[1:3]
          else:
            timestamp = path_string[2]
          #end section
          d=c[y1:y2, x1:x2]
          name = './framesimages/{}/{}/extracted{}.jpg'.format(username, job_code, timestamp)
          j=j+1
          cv2.imwrite(name, cv2.cvtColor(d, cv2.COLOR_RGB2BGR))
    return j
  j=0
  for image_path in imgpath:
    ans=extractvehicleregion(detection_model, image_path,j, username, job_code)
    j=ans



def vehicle_license_plate(impath,wpod_net,reader,loaded_model,labels):
    def get_plate(wpod_net, image_path, Dmax=608, Dmin=280):
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

    def ocrrecognition(img,reader):
      regthresh=0.7
      plate = cv2.convertScaleAbs(img, alpha=(255.0))
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
      finalplate =''
      for character in cnts:
          if character.isalnum():
              finalplate += character.upper()
      return finalplate

    def predict_from_model(image,model,labels):
      image = cv2.resize(image,(80,80))
      image = np.stack((image,)*3, axis=-1)
      prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
      return prediction

    def sort_function(d,e):
        n = len(e)
        for i in range(n-1):
            for j in range(0, n-i-1):
                if e[j] > e[j+1] :
                        e[j], e[j+1] = e[j+1], e[j]
                        d[j], d[j+1] = d[j+1], d[j]
        return d

    def sort_contours(cnts):
        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
          key=lambda b:b[1][i], reverse=reverse))
        return cnts

    def segm(img):
          plateimage = cv2.convertScaleAbs(img, alpha=(255.0))
          grayimage = cv2.cvtColor(plateimage, cv2.COLOR_BGR2GRAY)
          blurimg = cv2.GaussianBlur(grayimage,(5,5),0)
          binaryimg = cv2.threshold(blurimg, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
          kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
          thresh = cv2.morphologyEx(binaryimg, cv2.MORPH_DILATE, kernel3)
          cont, _  = cv2.findContours(binaryimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          testroi = plateimage.copy()
          crop_characters = []
          digit_w, digit_h = 30, 60
          for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5:
                if h/plateimage.shape[0]>=0.5:
                    cv2.rectangle(testroi, (x, y), (x + w, y + h), (0, 255,0), 2)
                    curr_num = thresh[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
            final_string = ''
            for j,character in enumerate(crop_characters):
                title = np.array2string(predict_from_model(character,loaded_model,labels))
                final_string+=title.strip("'[]")
          return final_string

    def order_points(pts):
      rect = np.zeros((4, 2), dtype = "float32")
      s = pts.sum(axis = 1)
      rect[0] = pts[np.argmin(s)]
      rect[2] = pts[np.argmax(s)]
      diff = np.diff(pts, axis = 1)
      rect[1] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]
      return rect

    def four_point_transform(image, pts):
      rect = order_points(pts)
      (tl, tr, br, bl) = rect
      widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
      widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
      maxWidth = max(int(widthA), int(widthB))
      heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
      heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
      maxHeight = max(int(heightA), int(heightB))
      dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
      M = cv2.getPerspectiveTransform(rect, dst)
      warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
      return warped

    def Characterdetect(image,region):
      plateimg=four_point_transform(image,region)
      V = cv2.split(cv2.cvtColor(plateimg, cv2.COLOR_BGR2HSV))[2]
      T = threshold_local(V, 29, offset=15, method="gaussian")
      thresh = (V > T).astype("uint8") * 255
      thresh = cv2.bitwise_not(thresh)
      plateimg= imutils.resize(plateimg, width=400)
      thresh = imutils.resize(thresh, width=400)
      labels = measure.label(thresh, background=0)
      charCandidates = np.zeros(thresh.shape, dtype="uint8")
      crop_characters=[]
      x_cor=[]
      i=0
      for label in np.unique(labels):
        if label == 0:
          continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(cnts) > 0:
          c=max(cnts,key=cv2.contourArea)
          (x, y, w, h) = cv2.boundingRect(c)
          aspectRatio = w / float(h)
          solidity = cv2.contourArea(c) / float(w * h)
          heightRatio = h / float(plateimg.shape[0])
          if aspectRatio<1.0 and solidity>0.15 and heightRatio>0.4 and heightRatio<0.95:
            cv2.rectangle(plateimg, (x, y), (x + w, y + h), (0, 255,0), 2)
            curr_num = thresh[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(30, 60))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.insert(i,curr_num)
            x_cor.insert(i,x)
            i=i+1
      charCandidates = segmentation.clear_border(charCandidates)
      return (charCandidates,crop_characters,x_cor)

    def segm2(impath,pts):
        crop_c=[]
        x_c=[]
        char_c,crop_c,x_c=Characterdetect(cv2.imread(impath),pts)
        d=sort_function(crop_c,x_c)
        final_string = ''
        for i,character in enumerate(d):
            title = np.array2string(predict_from_model(character,loaded_model,labels))
            final_string+=title.strip("'[]")
        return final_string

    try:
      plates=[]
      LpImg,cor = get_plate(wpod_net, impath)
      pts=[]
      x_coordinates=cor[0][0]
      y_coordinates=cor[0][1]
      for i in range(4):
          pts.append((int(x_coordinates[i]),int(y_coordinates[i])))
      plates.append(segm(LpImg[0]))
      plates.append(segm2(impath,np.array(pts)))
      plates.append(ocrrecognition(LpImg[0],reader))
      if len(plates[0])>6 and plates[0].startswith('KL'):
        return plates[0]
      elif len(plates[1])>6:
          return plates[1]
      elif len(plates[2])>6:
          return plates[2]
      else:
          return 0

    except Exception as e:
      print(e)
      return 0


#Module to identify the color of the detected vehicle
def vehicle_color(impath):
  def RGB2HEX(color):
      return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


  def get_image(image_path):
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image


  def closest_colour(requested_colour):
      min_colours = {}
      for key, name in webcolors.css3_hex_to_names.items():
          r_c, g_c, b_c = webcolors.hex_to_rgb(key)
          rd = (r_c - requested_colour[0]) ** 2
          gd = (g_c - requested_colour[1]) ** 2
          bd = (b_c - requested_colour[2]) ** 2
          min_colours[(rd + gd + bd)] = name
      return min_colours[min(min_colours.keys())]

  def get_colour_name(requested_colour):
      try:
          closest_name = rgb_to_name(requested_colour)
      except ValueError:
          closest_name = closest_colour(requested_colour)
      return closest_name

  def get_colors(image, number_of_colors, show_chart):

      modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
      modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

      clf = KMeans(n_clusters = number_of_colors)
      labels = clf.fit_predict(modified_image)

      counts = Counter(labels)
      # sort to ensure correct color percentage
      counts = dict(sorted(counts.items()))

      center_colors = clf.cluster_centers_
      # We get ordered colors by iterating through the keys
      ordered_colors = [center_colors[i] for i in counts.keys()]
      hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
      rgb_colors = [ordered_colors[i] for i in counts.keys()]
      #if (show_chart):
          #plt.figure(figsize = (8, 6))
        # plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
      return rgb_colors
  colours_detected = get_colors(get_image(impath), 2, True)
  colour_list = []
  for i in range(3):
    colour_list.append(int(colours_detected[0][i]))
  cp=tuple(colour_list)
  colour_recognized = get_colour_name(cp)
  return colour_recognized

# The model to recognise the type of the detected vehicle. The model used is mobilenet V2.
def vehicle_type(impath,model,labels):
  def predict_from_model(image,model,labels):
      image = cv2.resize(image,(80,80))
      image = np.stack((image,)*3, axis=-1)
      prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
      return prediction

  img = Image.open(impath)
  img_gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
  pred=predict_from_model(img_gray,model,labels)
  return pred[0]

def attribute_extract(path, username, job_code):
  finalans=[]
  frame_extraction(path, username, job_code)
  vehicle_detect(username, job_code)
  wpod_net_path = BASE_DIR / "ml/wpod-net.json"
  wpod_net = load_model(wpod_net_path)
  json_file = open(BASE_DIR / 'ml/model_char_recognitionorgnew.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  recognition_model = model_from_json(loaded_model_json)
  recognition_model.load_weights(BASE_DIR / "ml/model_char_recognitionorgnew.h5")
  characterlabels = LabelEncoder()
  characterlabels.classes_ = np.load(BASE_DIR / 'ml/license_character_classes.npy')
  model,labels=loadtypemodel()
  reader=easyocr.Reader(['en'])
  extracted_image_paths = glob.glob("framesimages/{}/{}/extracted*.jpg".format(username, job_code))
  for i in range(len(extracted_image_paths)):
      ans=[]
      ans.append(extracted_image_paths[i])
      ans.append(vehicle_license_plate(extracted_image_paths[i],wpod_net,reader,recognition_model, characterlabels))
      ans.append(vehicle_color(extracted_image_paths[i]))
      ans.append(vehicle_type(extracted_image_paths[i],model,labels))
      if ans[1]==0 and ans[3]=='Car':
          continue
      else:
      #timestamp section
          path_string = extracted_image_paths[i][-7:]
          if path_string[0] == 'e':
            timestamp = path_string[2]
          else:
            timestamp = path_string[1:3]
          #section end
          if int(timestamp) < 60:
            if int(timestamp)<10:
              formatted_timestamp = "00:00:0{}".format(timestamp)
            else:
              formatted_timestamp = "00:00:{}".format(timestamp)
          elif int(timestamp) < 3600:
              mod = timestamp%60
              mins = timestamp/60
              if int(mins)<10:
                if int(mod)<10:
                  formatted_timestamp = "00:0{}:0{}".format(mins, mod)
                else:
                  formatted_timestamp = "00:0{}:{}".format(mod)
              else:
                if int(mod)<10:
                  formatted_timestamp = "00:0{}:0{}".format(mins, mod)
                else:
                  formatted_timestamp = "00:0{}:{}".format(mod)
          else:
            pass #hours no needed


          #this field is used for timestamp
          ans.append(formatted_timestamp)
          record = VehicleRecord(job_code=job_code, license_plate=ans[1], colour=ans[2], vehicle_type=ans[3], vehicle_model=formatted_timestamp, vehicle_logo="nil")
          imageopen = open("{}".format(extracted_image_paths[i]), "rb")
          imagefile = File(imageopen)
          record.image.save("{}-{}-{}.jpg".format(username, job_code, i), imagefile, save=True)
          finalans.append(ans)
  return finalans



@shared_task
def run_job(username, job_code, domain, user_email):
    path = str(BASE_DIR / '../media/videosfolder/{}-{}.mp4'.format(username, job_code))
    attributes_list = attribute_extract(path, username, job_code)

    # for attrs in attributes_list:
    #     record = VehicleRecord(job_code=job_code, license_plate=attrs[1], colour=attrs[2], vehicle_type=attrs[3], vehicle_model="nil", vehicle_logo="nil")
    #     record.save()
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
        # 'content': ['asdas', 'asdasd', '1qwd3', '2wfdcq3w', 'd13fcww'],
        'content': attributes_list,
        'salutation': "Hi " + str(username) + " ,",
    }

    to_email = user_email
    message = EmailMultiAlternatives(subject=mail_subject, to=[to_email])
    message.attach_alternative(t.render(Context), "text/html")
    message.send()
