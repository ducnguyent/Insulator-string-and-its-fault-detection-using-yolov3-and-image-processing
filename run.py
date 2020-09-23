import cv2
import numpy as np

# Load Yolo
def yolo(weights, cfg, names, image, color):
    net = cv2.dnn.readNet(weights, cfg)
    classes = []
    with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Loading image
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    #img = cv2.resize(img, (720, 480))
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
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
    print(class_ids)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x , y + 50), font, 2, color, 3)

    return img, class_ids, boxes

#nhap thong so cua yolo
weights = "yolov3_training_best.weights"
cfg = "yolov3_training.cfg"
names = "yolo.names"
image = "glass/13.jpg"

weights1 = "yolov3_training_best_f.weights"
cfg1 = "yolov3_custom_train_f.cfg"
names1 = "data/yolo_f.names"
image1 = "1.jpg"
img_ori = cv2.imread(image, 1)
img, id, box = yolo(weights, cfg, names, image, (0, 255, 0))

#if id[0] == 2:
#    img, __ = yolo(weights1, cfg1, names1, img1, (0, 0, 255))
x, y, w, h = box[0]
print(box[0])
img0 = img_ori[0:y+h, x:x+w]
cv2.imshow("Image", img)
#cv2.imshow("i", img)
#cv2.imwrite('../venv/Lib/site-packages/mrcnn/glass/13.1.jpg', img1)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

def glassfault(img):
    #img = cv2.resize(img, (720,480))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def sub (a, b):
        c = int(a)-int(b)
        if c >= 0:
            return c
        if c < 0:
            return -1 * c

    x,y,z = img.shape
    print(img.shape)
    def improc(img):
        kernel1 = np.ones((9, 9), np.uint8)
        kernel2 = np.ones((25, 25), np.uint8)
        kernel3 = np.ones((29, 29), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
        img = cv2.dilate(img, kernel2, iterations=1)
        #img = cv2.erode(img,kernel2, iterations=1)
        return img
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
    def Spatial(img):
        #[m, n] = shape(img)
        Xdis = []
        for i in range(img.shape[1]):
            a = (sum(img[:, i]))
            Xdis.append(a)
        return Xdis
    def fault(img, Xdis):
        a1 = []
        a = []
        b = -1
        c = -1
        d = -1
        y = []
        for i in range(1, img.shape[1]-1):
            if Xdis[i] == 0 and Xdis[i] < Xdis[i-1]:
                a.append(i)
                b += 1
                c += 1
            elif Xdis[i] == 0 and Xdis[i] < Xdis[i+1]:
                a.append(i)
                b += 1
                if c == b-1:
                    a1.append(a[c])
                    a1.append(a[c+1])
                    d += 1
                c += 1
        m = np.argmax(Xdis)
        n = 0
        for j in range(1, img.shape[0]):
            n += 1
            if img[j, m] > img[j-1, m]:
                y.append(j)
            elif img[j, m] < img[j-1, m]:
                y.append(j)
        return a1[2:], d, y
    for i in range(x):
        for j in range(y):
            B, G, R = img[i,j]
            bool1 = (25 <= R <= 165)
            bool2 = (40 <= B <= 115)
            bool3 = (6 <= sub(G, R) <= 45)

            if bool1 and bool2 and bool3 :
                img1[i, j] = 255
            else:
                img1[i, j] = 0

    cv2.medianBlur(img1, 7, img1)

    lines = cv2.HoughLinesP(img1, 1, np.pi/180, 100, minLineLength=100, maxLineGap=100)
    len=[]
    theta1=[]

    for line in lines:
        x1,y1,x2,y2=line[0]
        #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        len1 = np.sqrt((y2-y1)**2+(x2-x1)**2)
        a = ((y2-y1)/(x2-x1))
        theta=np.arctan(a)
        len.append(len1)
        theta1.append(theta)

    pos = np.argmax(len)
    angle = -theta1[pos]*180/np.pi
    print(angle)

    img = rotate_bound(img, angle)
    img2 = rotate_bound(img1, angle)
    img1 = img2
    img2 = improc(img2)

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i, j]>0:
                img2[i, j]=255
            else:
                img2[i, j]=0

    cv2.imshow('img1',img1)
    cv2.imshow('img2',img2)
    #cv2.waitKey(0)
    #cv2.imwrite('glass/13.2.jpg', img1)

    Xdis = Spatial(img2)
    font = cv2.FONT_HERSHEY_PLAIN
    x, d, y = fault(img2, Xdis)
    print(x, y)
    for i in range(d):
        name = "fault"
        cv2.rectangle(img, (x[2*i]-10, y[0]+2), (x[2*i+1]+10, y[1]-2), (0, 0, 255), 3)
        cv2.putText(img, name, (x[2*i]-10, y[0]-2), font, 2, (0, 0, 255), 3)
    cv2.imshow("img", img)

    plt.plot(Xdis)
    plt.show()

glassfault(img)
