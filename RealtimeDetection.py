import cv2
from keras.models import model_from_json
import numpy as np

def read_model_from_json(file_path):
    with open(file_path, "r") as file:
        model_json = file.read()
    return model_from_json(model_json)

def extract_image_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def main():
    model = read_model_from_json("emotiondetector.json")
    model.load_weights("emotiondetector.h5")
    
    haar_cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_file)

    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        try: 
            for (x, y, w, h) in faces:
                face_image = gray_frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                resized_face = cv2.resize(face_image, (48, 48))
                features = extract_image_features(resized_face)
                prediction = model.predict(features)
                predicted_label = labels[prediction.argmax()]
                cv2.putText(frame, predicted_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
                
            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
