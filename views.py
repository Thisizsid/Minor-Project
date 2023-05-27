from django.shortcuts import render,redirect
from .forms import CreateUserForm
from django.contrib.auth import authenticate,login,logout
from subprocess import call
from home.models import Login
from django.contrib import messages
from django.http import HttpResponse
from .models import *
#import simplejson as json
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading
# views.py
from django.http import JsonResponse
import cv2
import time
import math
### importing required libraries
import torch
# import pytesseract
import re
import numpy as np
import easyocr
import math
global class1
global class2
class1={}
class2={}
import datetime




from .models import Speed

# Create your views here.
def index(request):
    return render(request,'index.html')
    
def team(request):
    return render(request,"team.html")

def design(request):
    speeds = Speed.objects.all() # fetching all the speeds from the database
    context = {'speeds': speeds}
    return render(request,"design.html",context)

#def login_in(request):
    if request.method=="POST":
        name= request.POST.get('name')
        email= request.POST.get('email')
        phone= request.POST.get('phone')
        login=Logins(name=name,email=email,phone=phone)
        login.save()
        messages.success(request, 'Successfully sent!!.')
    return render(request,"login.html")

def registerPage(request):
    form= CreateUserForm ( )

    if request.method == "POST":
        form= CreateUserForm(request.POST )
        if form.is_valid():
            form.save()
            user= form.cleaned_data.get("username") # for retreiving the username from form registration
            messages.success(request, 'Account was created for '+ user)
            return redirect('login1')

    context ={'form':form}
    return render(request,'register.html',context)

def loginPage(request):

    if request.method == "POST":
        username=request.POST.get("username")
        password=request.POST.get("password")

        user= authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            messages.success(request, 'Welcome '+username)
            return redirect('home')
        else :
            messages.info(request,"Username OR Password is incorrect")
            

    context ={}
    return render(request,'login1.html',context)
    
def logoutUser(request):
    logout(request)
    return redirect('login1')


#def push(request):
    p=Login(number_plate="000",speed=50,s_speed="30")
    p.save()
    messages.info(request, 'Successfully Pushed!!.')
    return render(request,"index.html")


##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['de','en'], gpu= True) ### initiating easyocr
OCR_TH = 0.5
### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")
    dictionaries={}
    time3=time.time()
    ### looping through the detections
    for i in range(n):
        value = []
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])
    
            coords = [x1,y1,x2,y2]

            z1=int((x1+x2)/2)
            z2=int((y1+y2)/2)
            cords=[z1,z2]
            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            number = str(plate_num)
            dictionaries[number]=cords


            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])
    estimateSpeed(dictionaries,time3)       



    return frame

def estimateSpeed(dictonaries,timess):
    global s_time
    global class1,class2
    class2=dictonaries
    for key2 in class2:
        for key1 in class1:
            if key1==key2:
                location1=class1[key1]
                location2=class2[key2]
                # the logic to estimate speed
                d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
                # ppm = location2[2] /  carWidhs
                ppm=35
                d_meters = d_pixels / ppm
                e_time=timess
                #fpps=int(cap.get(cv2.CAP_PROP_FPS))
                fpss=1.0/(e_time-s_time)
                # current_time = datetime.datetime.now()
                #print("The current time is:", current_time)
                speed = int(d_meters * fpss *3.6)
                if speed>8 and speed<25:
                    speed_data = Speed(vehicle_number=key2, speed=speed,s_speed='8') # creating an instance of the Speed model
                    speed_data.save() # saving the instance to the database
                    print("The speed of numberplate "+key2+" is "+str(speed)+" km/hr ")
    s_time=timess
    class1=class2
    
#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image


    ocr_result = reader.readtext(nplate, paragraph = False)



    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    print(text)

    if len(text) ==1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate





### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):
    global s_time
    s_time=time.time()
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('../yolov5-master', 'custom', source ='local', path='../best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format




    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame,classes = classes)
        

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.

                break

    ### --------------- for detection on video --------------------
    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1

        
        print(f"[INFO] Cleaning up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()





#cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)


def vehicle_speed(request):
### -------------------  calling the main function-------------------------------
    main(vid_path="../test_images/7.mp4",vid_out="vid_1.mp4") ### for custom video
    # main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam

    # main(img_path="./test_images/cars119.jpg") ## for image
            

    #speeds = Speed.objects.all() # fetching all the speeds from the database
    #context = {'speeds': speeds}
    #return render(request, 'vehicle_speed.html', context)
    return render(request, 'index.html')


"""
@gzip.gzip_page

def Home(request):
    try:
        #cam = VideoCamera()
        cam=cv2.VideoCapture('carsvideoo.mp4')
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'app1.html')

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
"""

