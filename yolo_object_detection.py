import cv2
import numpy as np
import glob
import random
from scipy import ndimage
from matplotlib import pyplot as plt


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_best.weights", "yolov3_training.cfg")

# Name custom object
classes = ["glass insulator","white insulator","red insulator"]

# Images path
images_path = glob.glob(r"data_test\Failure-of-Line-Insulators(6).jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)

# loop through all the images
for img_path in images_path:
    # Loading image
    img_old = cv2.imread(img_path)
    img = cv2.resize(img_old, None, fx=0.7, fy=0.7)
    height, width, channels = img.shape
    
    #Chuyển hệ màu BGR sang RGB (hàm plt dùng RGB, cv2 dùng BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #FILE NAME IMAGE
    name_img = img_path[46:-4] 

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    n=0
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
  
                confidences.append(float(confidence))
                class_ids.append(class_id)

                #LabelImg txt data
                
                center_x1=center_x/img.shape[1]
                center_y1=center_y/img.shape[0]
                
                w1=w/img.shape[1]
                h1=h/img.shape[0]
                
                X=np.empty(shape=[n,5])
                D=np.append(X,[[15 ,x,y,w,h]],axis=0)
    

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    img_new = img.copy()
    d=2 		#do day khung detect

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]	
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
         
            if(label=='glass insulator'):		#glass = màu xanh lá
            	cv2.rectangle(img_new, (x, y), (x + w, y + h), [0,255,0],d)
            	cv2.putText(img_new, label, (x, y - 10), font, 2, [0,255,0], d)
            if(label=='white insulator'):		#white = màu xanh dương
            	cv2.rectangle(img_new, (x, y), (x + w, y + h), [255,0,0],d)
            	cv2.putText(img_new, label, (x, y - 10), font, 2, [255,0,0], d)
            if(label=='red insulator'):			#red = màu đỏ
            	cv2.rectangle(img_new, (x, y), (x + w, y + h), [0,0,255],d)
            	cv2.putText(img_new, label, (x, y - 10), font, 2, [0,0,255], d)





h1= int(h/10)
w1=int(w/10)
img_crop=img[y-h1:y+h+h1, x-w1:x+w+w1]

#img_crop = cv2.resize(img_crop,(img_crop.shape[1]*5,img_crop.shape[0]*5))


#Xoay ngang ảnh nếu ảnh là ảnh dọc
if(h>w):
     img_crop =ndimage.rotate(img_crop, 90)

#Display dùng cv2 
cv2.imshow("Image", img_new)
key = cv2.waitKey(1)

cv2.imshow("Image_crop", img_crop)
key = cv2.waitKey(0)
cv2.destroyAllWindows()


'''
#Display dùng plt, hiển thị toàn bộ 1 lần trên 1 khung
# Lưu ý nhớ chuyển hệ màu trước khi dùng

titles = ['Anh goc','Detect',"Crop"]
images = [img, img_new, img_crop]

i=0
o=len(images)
for i in range(o):
    plt.subplot(5,5,i+1)
    plt.imshow(images[i])
    #plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

'''