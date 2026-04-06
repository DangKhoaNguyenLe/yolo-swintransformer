import cv2
import pickle

from face_detection.detect_face import detect_face
from face_recognition.swin_embedding import get_embedding
from utils.similarity import compare

database = pickle.load(open("database/embeddings.pkl","rb"))

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    faces = detect_face(frame)

    for (x1,y1,x2,y2) in faces:

        face = frame[y1:y2,x1:x2]

        emb = get_embedding(face)

        name = "Unknown"

        best = 0

        for person in database:

            for db_emb in database[person]:

                score = compare(emb,db_emb)

                if score > best:

                    best = score
                    name = person

        if best < 0.6:

            name = "Unknown"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,name,(x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1)==27:

        break

cap.release()

cv2.destroyAllWindows()