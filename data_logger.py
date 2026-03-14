import cv2
import time
import csv

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

camera = cv2.VideoCapture(0)

log_file = open("log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["time", "people_detected"])

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(gray, winStride=(4,4))

    count = len(boxes)

    current_time = time.strftime("%H:%M:%S")
    writer.writerow([current_time, count])

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(frame, f'People: {count}', (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("CCTV Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
log_file.close()
cv2.destroyAllWindows()