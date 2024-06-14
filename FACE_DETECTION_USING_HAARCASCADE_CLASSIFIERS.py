import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

    if len(faces) > 0:
        # Find the face with the maximum area
        max_area = 0
        max_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                max_face = (x, y, w, h)

        # Draw a rectangle around the face with the maximum area
        if max_face is not None:
            (x, y, w, h) = max_face
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Resize the image for display
    resized_image = cv2.resize(img, (640, 480))

    # Display the resulting frame
    cv2.imshow("Detected-face", resized_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
