import cv2
from tracker import *
from matplotlib import pyplot as plt

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("outputs/random.gif") #select the gif to track

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

while True:
    ret, frame = cap.read()
    try:
        height, width, _ = frame.shape
    except:
        break
    roi = frame

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    boxes_ids,x_coord,y_coord = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

cap.release()
cv2.destroyAllWindows()
y_coord = y_coord[::-1]
plt.plot(x_coord[0],y_coord[0],'x',color='red')
plt.plot(x_coord,y_coord,color='blue')
plt.plot(x_coord[-1],y_coord[-1],'x',color='red')
plt.xlim(0,500)
plt.xlabel("X-Coordinate")
plt.ylabel("Y-Coordinate")
plt.title("Final Trajectory of Lunar Lander - Random Policy")
plt.savefig("random.png")
plt.show()
