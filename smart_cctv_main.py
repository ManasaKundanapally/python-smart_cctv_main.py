import cv2
import csv
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

camera = cv2.VideoCapture(0)

customer_count = 0
line_position = 250
counted = False
start_time = None

log_file = open("../data/smart_cctv_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["time", "people_detected", "customers_entered", "time_in_frame"])

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4))

    frame_width = frame.shape[1]
    cv2.line(frame, (0, line_position), (frame_width, line_position), (255, 0, 0), 2)

    count = len(boxes)
    person_detected = False

    if count > 0:
        if start_time is None:
            start_time = time.time()
        elapsed = int(time.time() - start_time)
    else:
        start_time = None
        elapsed = 0

    for (x, y, w, h) in boxes:
        person_detected = True
        center_y = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x + w // 2, center_y), 5, (0, 0, 255), -1)

        if center_y > line_position and not counted:
            customer_count += 1
            counted = True

    if not person_detected:
        counted = False

    current_time = time.strftime("%H:%M:%S")
    writer.writerow([current_time, count, customer_count, elapsed])

    cv2.putText(frame, f'People: {count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f'Entered: {customer_count}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f'Time: {elapsed}s', (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

    cv2.imshow("Smart CCTV Behaviour Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
log_file.close()
cv2.destroyAllWindows()