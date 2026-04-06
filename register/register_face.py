import cv2
import os

name = input("Nhập tên của bạn: ")

path = f"dataset/persons/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    cv2.imshow("Đăng ký",frame)

    key = cv2.waitKey(1)

    if key == ord('s'):

        cv2.imwrite(f"{path}/{count}.jpg",frame)

        print("saved",count)

        count += 1

    if key == 27:

        break

cap.release()

cv2.destroyAllWindows()