import cv2
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
test_video = BASE_DIR / 'Videos' / '5198949-uhd_4096_2160_25fps.mp4'

cap = cv2.VideoCapture(test_video)

yolo_model = YOLO('../yolov8n.pt')

if not cap.isOpened():
    print('Камера жок экен!')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))
classes = [2, 4, 15, 16]

if frame_fps == 0:
  frame_fps = 30.00

new_date_format = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = None

while True:
    ret, frame = cap.read()
    start_time = time.time()

    if not ret:
        print('Кадр жок экен!')
        break

    frame = cv2.flip(frame, 1)

    predict = yolo_model(frame, stream=True, conf=0.5, classes=classes)

    for i in predict:
        for n in i.boxes:
            cls = int(n.cls[0])
            conf = float(n.conf[0])
            label = yolo_model.names[cls]


            new_conf = round(100 * conf)
            x, y, w, h = map(int, n.xyxy[0])
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}:{new_conf}-%', (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(frame, 'AI-32', (550, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (100, 100, 87), 2)

    f_time = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, f'{f_time}', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 87), 1)

    if out is not None:
        out.write(frame)
        cv2.putText(frame, '● REC', (1150, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.putText(frame, f'FPS: {round(fps)}', (0, 120),
                cv2.FONT_ITALIC, 1, (255, 0, 0), 1)
    start_time = end_time

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('v'):
        if out is None:
            out = cv2.VideoWriter(
                f'Videos/output_{new_date_format}.avi', fourcc, frame_fps, (frame_width, frame_height))
    elif key == ord('s'):
        image_name = f'Screens/image_{new_date_format}.png'
        cv2.imwrite(image_name, frame)

cap.release()
cv2.destroyAllWindows()



