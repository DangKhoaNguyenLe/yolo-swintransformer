from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_face(frame):

    results = model(frame)

    faces = []

    for r in results:

        for box in r.boxes.xyxy:

            x1,y1,x2,y2 = map(int,box)

            faces.append((x1,y1,x2,y2))

    return faces