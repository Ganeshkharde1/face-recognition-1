import cv2

# Load pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('model.yml')

# Open video capture device
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Crop face region
        face = gray[y:y+h, x:x+w]
        
        # Resize face to match the training data size
        face = cv2.resize(face, (100, 100))
        
        # Predict label for the face using the face recognition model
        label, confidence = model.predict(face)
        
        # Draw a rectangle around the face and display label and confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {label}, Confidence: {confidence}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
