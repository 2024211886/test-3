from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头无法打开！请检查摄像头连接或权限")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面")
        break

    results = model(frame, classes=[63,67])

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(f"检测到{model.names[int(box.cls[0])]}，中心点坐标：({center_x}, {center_y})")
            cv2.putText(frame, f"({center_x},{center_y})", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('YOLOv8 Smart Shelf Monitor', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
