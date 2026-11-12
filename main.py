import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

cam = cv2.VideoCapture(0)
if not cam.isOpened():
  print('Мындай камера жок экен!')
  exit()

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cam.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
  frame_fps = 30.00

new_date_format = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = None

while True:
  ret, frame = cam.read()
  if not ret:
    print('Кадр жок экен!')
    break

  frame = cv2.flip(frame, 1)
  # frame = cv2.resize(frame, (640, 480))

  cv2.putText(
      frame, 'AI-32', (550, 40), cv2.FONT_ITALIC, 1.5, (100, 60, 50), 2)

  f_time = datetime.now().strftime('%H:%M:%S')
  cv2.putText(
      frame, f'{f_time}', (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 87), 1)

  if out is not None:
    out.write(frame)
    cv2.putText(frame, '● REC', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  cv2.imshow('Test Video: ', frame)

  key = cv2.waitKey(1) & 0xFF

  if key == ord('q'):
    break

  elif key == ord('v'):
    # 'v' басылганда файл бир эле жолу ачылат
    if out is None:
      out = cv2.VideoWriter(
          f'Videos/output_{new_date_format}.avi', fourcc, frame_fps, (frame_width, frame_height),
      )
  elif key == ord('s'):
      image_name = f'Screens/image_{new_date_format}.png'
      cv2.imwrite(image_name, frame)

if out is not None:
  out.release()
  print('Видео сохранено!')

cam.release()
cv2.destroyAllWindows()



# ушундан пример алып жазгам


