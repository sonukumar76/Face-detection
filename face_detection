import cv2

# Haar Cascade XML file ka path sahi rakho
face_cascade = cv2.CascadeClassifier("project/face detection/haarcascade_frontalface_default (1).xml")

# Correct function: VideoCapture (V capital hai)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Correct spelling: detectMultiScale
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("video live", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
