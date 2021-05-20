from django.shortcuts import render, redirect
from django.contrib.auth import (authenticate, login, logout)
from django.contrib import messages
from time import sleep

# from numpy.lib.arraysetops import ediff1d
from .forms import LoginForm
from .models import Image
from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


#ml imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from basic_app.ml.local_utils import detect_lp
import utils as ut
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import matplotlib
from sklearn.preprocessing import LabelEncoder





#ml functions
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


def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
def get_plate(image_path, wpod_net, Dmax=608, Dmin=280):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
def get_plate(image_path, wpod_net, Dmax=608, Dmin=280):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts







#views
def index(request):
    form = LoginForm()
    if request.method=='GET':
        return render(request, 'index.html', {'form':form})
    else:
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('basic_app:dashboard')
        messages.error(request, "Invalid username or password")
        return redirect('basic_app:index')

def logout_view(request):
    logout(request)
    return redirect('basic_app:index')


def dashboard(request):
    if request.method=="GET":
        POST = 0
        return render(request, 'dashboard.html', {'post':POST})
    else:
        POST = 1 # for get post differentiation in template for conditional rendering
        images = request.FILES.getlist('images')
        for image in images:
            imgobj = Image.objects.create(imagefile=image)        
        #ml portion starts





        plates_list = []
        wpod_net_path = BASE_DIR / "basic_app/ml/wpod-net.json"
        wpod_net = load_model(wpod_net_path)
        json_file = open( BASE_DIR / 'basic_app/ml/model_char_recognitionorgnew.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(BASE_DIR / "basic_app/ml/model_char_recognitionorgnew.h5")
        labels = LabelEncoder()
        labels.classes_ = np.load(BASE_DIR / 'basic_app/ml/license_character_classes.npy')
        image_paths = glob.glob("media/imagesfolder/*")
        program_crashed = 0
        for i in range(len(image_paths)):
            try:
                LpImg,_ = get_plate(image_paths[i], wpod_net)
                for k in range(len(LpImg)):
                    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
                    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray,(7,7),0)
                    binary = cv2.threshold(blur, 180, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
                    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    test_roi = plate_image.copy()
                    crop_characters = []
                    digit_w, digit_h = 30, 60
                    for c in sort_contours(cont):
                        (x, y, w, h) = cv2.boundingRect(c)
                        ratio = h/w
                        if 1<=ratio<=3.5:
                            if h/plate_image.shape[0]>=0.5:
                                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                                curr_num = thre_mor[y:y+h,x:x+w]
                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                crop_characters.append(curr_num)
                        final_string = ''
                        for j,character in enumerate(crop_characters):
                            title = np.array2string(predict_from_model(character,loaded_model,labels))
                            final_string+=title.strip("'[]")
                    
                    if(len(final_string)==0):
                        plates_list.append("Unrecognizable") 
                    else:          
                        plates_list.append(final_string)
            except Exception as e:
                print(e)
                program_crashed = 1

        #ml portion ends
        Image.objects.all().delete()
        return render(request, 'dashboard.html', {'plates':plates_list, 'crashed':program_crashed, 'post':POST})





        




