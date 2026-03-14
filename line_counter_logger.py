import cv2
import csv
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

camera = cv2.VideoCapture(0)

customer_count = 0
line_position = 250
counted = False

log_file = open("../data/line_crossings.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["time", "event", "total_count"])

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4))

    frame_width = frame.shape[1]
    cv2.line(frame, (0, line_position), (frame_width, line_position), (255, 0, 0), 2)

    person_detected = False

    for (x, y, w, h) in boxes:
        person_detected = True
        center_y = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x + w // 2, center_y), 5, (0, 0, 255), -1)

        if center_y > line_position and not counted:
            customer_count += 1
            counted = True

            current_time = time.strftime("%H:%M:%S")
            writer.writerow([current_time, "line_crossed", customer_count])

    if not person_detected:
        counted = False

    cv2.putText(frame, f'Customers Entered: {customer_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Line Crossing Logger", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
log_file.close()
cv2.destroyAllWindows()