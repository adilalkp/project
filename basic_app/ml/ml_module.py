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
import cv2
import easyocr
import os
import numpy as np
import matplotlib.pyplot as plt
from basic_app.ml.local_utils import detect_lp
import utils as ut
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import matplotlib
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils
import os
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
# %matplotlib inline

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
    cap = cv2.VideoCapture(path)
    count = 0
    cur=0
    try:
        if not os.path.exists('framesimages/{}'.format(username)):
            os.makedirs('framesimages/{}'.format(username))
        if not os.path.exists('framesimages/{}/{}'.format(username,job_code)):
            os.makedirs('framesimages/{}/{}'.format(username,job_code))
    except OSError as e:
        print (e)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("ret")
            name = './framesimages/{}/{}/frame{}.jpg'.format(username, job_code, str(cur))
            cv2.imwrite(name, frame)
            cur=cur+1
            count += 30 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break
    


# Use Tensorflow Object detection api for Vehicle detection from frames
def vehicle_detect(username, job_code):
  # patch tf1 into `utils.ops`
  utils_ops.tf = tf.compat.v1

  # Patch the location of gfile
  tf.gfile = tf.io.gfile
  def load_model(model_name):
    model_file = model_name + '.tar.gz'
    model_dir = BASE_DIR / '{}/saved_model'.format(model_name)
    model = tf.saved_model.load(str(model_dir))
    return model

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  TEST_IMAGE_PATHS= glob.glob("framesimages/{}/{}/frame*.jpg".format(username, job_code))

  # Model used for detection is ssd+mobilenet
  model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
  detection_model = load_model('ssd_mobilenet_v1_coco_2017_11_17')
  #print(detection_model.signatures['serving_default'].inputs)
  #detection_model.signatures['serving_default'].output_dtypes
  #detection_model.signatures['serving_default'].output_shapes
  def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
      
    return output_dict
  def show_inference(model, image_path,j, username, job_code):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    #print(output_dict)

    #display(Image.fromarray(image_np))
    imgs=Image.fromarray(image_np)
    #cv2.line(str(image_path), (0, 600),(1080,600),(100, 255, 255),5)
    imginput = cv2.imread(image_path)
    img= cv2.cvtColor(imginput, cv2.COLOR_BGR2RGB)
    cv2.line(img, (0, 600),(1500,600),5)
    #display(Image.fromarray(img))

  
    (w,h)=imgs.size
    
    checkclass=[2,3,4,6,8]
    for i in range(len(output_dict['detection_classes'])):
      if(output_dict['detection_classes'][i] in checkclass and output_dict['detection_scores'][i]>0.5): 
        
        c=np.array(Image.open(image_path))
        x1=int(output_dict['detection_boxes'][i][1]*w)
        y1=int(output_dict['detection_boxes'][i][0]*h)
        x2=int(output_dict['detection_boxes'][i][3]*w)
        y2=int(output_dict['detection_boxes'][i][2]*h)
        x=(x1+x2)/2
        y=(y1+y2)/2
        y=max(y1,y2)
        if(y>650 and y<750):
          d=c[y1:y2, x1:x2]
          name = './framesimages/{}/{}/extracted{}.jpg'.format(username, job_code, str(j))
          j=j+1
          cv2.imwrite(name, cv2.cvtColor(d, cv2.COLOR_RGB2BGR))
          
        #print(x1,x2,y1,y2) 
    return j 
  j=0
  for image_path in TEST_IMAGE_PATHS:
    ans=show_inference(detection_model, image_path,j, username, job_code)
    j=ans

#Module to detect license plate and recognise the license plate characters from it. The model used for license plate 
#detection is wpod-net and the recognition is done using ocr (pytorch)
def vehicle_license_plate(impath,wpod_net,reader):  
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
  try:
    LpImg,_ = get_plate(wpod_net, impath)
    regthresh=0.7
    plate = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
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
    if len(finalplate)==0:
      return 0
    else:
      return finalplate
  except:
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
  model,labels=loadtypemodel()
  reader=easyocr.Reader(['en'])
  extracted_image_paths = glob.glob("framesimages/{}/{}/extracted*.jpg".format(username, job_code))
  for i in range(len(extracted_image_paths)):
      ans=[]
      ans.append(extracted_image_paths[i])
      ans.append(vehicle_license_plate(extracted_image_paths[i],wpod_net,reader))
      ans.append(vehicle_color(extracted_image_paths[i]))
      ans.append(vehicle_type(extracted_image_paths[i],model,labels))
      finalans.append(ans)
  return finalans

def run_job(username, job_code, domain, email):
    path = str(BASE_DIR / '../media/videosfolder/{}-{}.mp4'.format(username, job_code))
    attributes_list = attribute_extract(path, username, job_code)

    for attrs in attributes_list:
        record = VehicleRecord(job_code=job_code, license_plate=attrs[1], colour=attrs[2], vehicle_type=attrs[3], vehicle_model="nil", vehicle_logo="nil")
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
        # 'content': ['asdas', 'asdasd', '1qwd3', '2wfdcq3w', 'd13fcww'],
        'content': attributes_list,
        'salutation': "Hi " + str(username) + " ,",
    }

    to_email = user_email
    message = EmailMultiAlternatives(subject=mail_subject, to=[to_email])
    message.attach_alternative(t.render(Context), "text/html")
    message.send()





































# from .models import Video, Job, VehicleRecord
# from pathlib import Path
# import uuid
# BASE_DIR = Path(__file__).resolve().parent.parent
# from django.utils import timezone
# import shutil
# from celery import shared_task

# #mail
# from django.contrib.sites.shortcuts import get_current_site
# from django.core.mail import EmailMultiAlternatives
# from django.template.loader import get_template


# #ml imports
# import cv2
# import numpy as np
# import os
# import utils as ut
# from os.path import splitext, basename
# import glob
# import cv2
# from basic_app.ml.local_utils import detect_lp
# from tensorflow.keras.models import model_from_json
# import glob
# from sklearn.preprocessing import LabelEncoder
# from collections import namedtuple
# from skimage.filters import threshold_local
# from skimage import segmentation
# from skimage import measure
# from imutils import perspective
# import imutils
# import easyocr


# def load_model(path):
#     try:
#         path = splitext(path)[0]
#         with open('%s.json' % path, 'r') as json_file:
#             model_json = json_file.read()
#         model = model_from_json(model_json, custom_objects={})
#         model.load_weights('%s.h5' % path)
#         print("loaded model")
#         return model
#     except Exception as e:
#         print("couldnt load model")
#         print(e)

# def get_plate(wpod_net,image_path, Dmax=608, Dmin=280):
#     vehicle = preprocess_image(image_path)
#     ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
#     side = int(ratio * Dmin)
#     bound_dim = min(side, Dmax)
#     _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
#     print("get plate ran successfully")
#     return LpImg, cor

# def preprocess_image(image_path,resize=False):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img / 255
#     if resize:
#         img = cv2.resize(img, (224,224))
#     print("process imaes ran successfully. Image path recievede - {}".format(image_path))
#     return img


# def filt(thresh,ocr_result,regthresh):
#   rect=thresh.shape[0]*thresh.shape[1]
#   plate=[]
#   for result in ocr_result:
#     ln=np.sum(np.subtract(result[0][1],result[0][0]))
#     h=np.sum(np.subtract(result[0][2],result[0][1]))
#     if ln*h/rect>regthresh:
#       plate.append(result[1])
#   print("filter ran successfully")
#   return plate



# def capture_frames(username, job_code):
#     video_path = str(BASE_DIR / 'media/videosfolder/{}-{}.mp4'.format(username, job_code)) #since cv2.VideoCapture doesn't 
#                                                                             #accept POSIX path, we convert it into string
#     cam = cv2.VideoCapture(video_path)
#     currframe = 0
#     try:
#         if not os.path.exists('framesimages/{}'.format(username)):
#             os.makedirs('framesimages/{}'.format(username))
#         if not os.path.exists('framesimages/{}/{}'.format(username,job_code)):
#             os.makedirs('framesimages/{}/{}'.format(username,job_code))
#     except OSError as e:
#         print (e)
#     while(True):
#         ret,frame = cam.read()
#         if ret:
#             cam.set(cv2.CAP_PROP_POS_MSEC,(currframe*1000))
#             name = './framesimages/{}/{}/frame{}.jpg'.format(username, job_code, str(currframe)) 
#             cv2.imwrite(name, frame)
#             currframe += 1
#         else:
#             break
#     cam.release()
#     cv2.destroyAllWindows()
#     print("capture frames ran successfully")



# def read_plates(username, job_code):
#     capture_frames(username, job_code)
#     wpod_net_path = BASE_DIR / "basic_app/ml/wpod-net.json"
#     print("read plates 1")
#     wpod_net = load_model(wpod_net_path)
#     reader=easyocr.Reader(['en'])
#     plates_list=[]
#     image_paths = glob.glob("framesimages/{}/{}/*.jpg".format(username, job_code))
#     for i in range(len(image_paths)):
#         try:
#             LpImg,_ = get_plate(wpod_net, image_paths[i])
#             impath=image_paths[i]
#             for k in range(len(LpImg)):
#                 regthresh=0.7
#                 plate = cv2.convertScaleAbs(LpImg[k], alpha=(255.0))
#                 V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
#                 T = threshold_local(V, 29, offset=15, method="gaussian")
#                 thresh = (V > T).astype("uint8") * 255
#                 thresh = cv2.bitwise_not(thresh)
#                 thresh = imutils.resize(thresh, width=400)
#                 ocr_result=reader.readtext(thresh)
#                 c=filt(thresh,ocr_result,regthresh)
#                 cnts=''
#                 for i in c:
#                     cnts=cnts+i
#                 chkplt = ""
#                 for character in cnts:
#                     if character.isalnum():
#                         chkplt += character

#                 if chkplt not in plates_list:
#                     if chkplt == '':
#                         pass
#                     else:
#                         plates_list.append(chkplt)
#                     #accomodate colour code
#         except Exception as e:
#             print(e)
#             pass
#     for i in plates_list:
#         print(".{}".format(i))
#     return plates_list


# @shared_task
# def run_job(username, job_code, domain, user_email):
#     #first read plates
#     plates_list = read_plates(username, job_code)
#     #call other ml modules

#     #update job in Job
#     # run some loop to iterate through sample cases, here it is list of license plates
#     for plate in plates_list:
#          record = VehicleRecord(job_code=job_code, license_plate=plate, colour="nil", vehicle_type="nil", vehicle_model="nil", vehicle_logo="nil")
#          record.save()
#     job_obj = Job.objects.filter(job_code=job_code).first()
#     job_obj.status = "completed"
#     job_obj.completed_on = timezone.now()
#     job_obj.save()

#     #send mail to user
#     mail_subject = 'Vehicle Recognition Report'
#     t = get_template('gmail_template.html')
#     Context = {
#         'domain': domain,
#         'subheading': "Your vehi-scanner report is ready",
#         # 'content': ['asdas', 'asdasd', '1qwd3', '2wfdcq3w', 'd13fcww'],
#         'content': plates_list,
#         'salutation': "Hi " + str(username) + " ,",
#     }

#     to_email = user_email
#     message = EmailMultiAlternatives(subject=mail_subject, to=[to_email])
#     message.attach_alternative(t.render(Context), "text/html")
#     message.send()

#     #clear ml waste(frames, video..)
#     # if os.path.exists('framesimages/{}/{}'.format(username,job_code)):
#     #     job_frames_location = BASE_DIR / "framesimages/{}/{}/".format(username,job_code)
#     #     shutil.rmtree(job_frames_location)
#     # video_obj = Video.objects.filter(job_code=job_code).first()
#     # video_obj.delete()

    