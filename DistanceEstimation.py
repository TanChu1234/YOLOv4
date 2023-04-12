import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 33 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    # classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[0], score)
        # # draw rectangle on and label on object
        
        
        # # getting the data 
        # # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            data_list.append([class_names[0], box[2], (box[0], box[1]-2)])
        # elif classid ==67:
        #     data_list.append([class_names[0], box[2], (box[0], box[1]-2)])
        # # if you want inclulde more classes then you have to simply add more [elif] statements here
        # # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
# ref_mobile = cv.imread('ReferenceImages/image4.png')

# mobile_data = object_detector(ref_mobile)
# mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)

person_width_in_rf = person_data[0][1]

# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

# focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)



cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    count_people = 0
    arr = []
    for d in data:
        if d[0] =='person':
            count_people += 1
            distance = 2.54*distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(frame, f'Dis: {round(distance,2)} cm', (x+5,y+13), FONTS, 0.48, GREEN, 2)
            arr.append(distance)
    if len(arr) == 0:
        print("Không có ai trong khung hình")
    # elif len(arr) == 1:
    #     print(f"Có 1 người trong khung hình cách camera {arr[0]} cm")
    else:
        dis_cm = " "
        for i in range(0, len(arr)):
            dis_cm += (str(round(arr[i], 2)) + " cm" + ", ")
        print(f"Có {len(arr)} người trong khung hình cách camera lần lượt là {dis_cm}" )
        # elif d[0] =='cell phone':
        #     distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
        #     x, y = d[2]

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()


