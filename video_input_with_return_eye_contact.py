import cv2
import numpy as np
from keras.models import load_model
import datetime

def process_video(video_path):
    # Load the pre-trained model
    model = load_model('CVTeam/best_model.h5')

    # Emotion and eye contact dictionaries for labeling predictions
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    # Load the Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Initialize video capture with the MP4 file
    cap = cv2.VideoCapture(video_path)
    print('Gathering FPS data...')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'{fps} FPS gathered!')

    frame_skip = 10  # Adjust based on your needs for performance vs. accuracy
    frame_count = 0
    eye_contact_losses = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        eye_detected_in_this_frame = False

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eye_detected_in_this_frame = True
                break  # If eyes are detected in any face, break the loop

        if not eye_detected_in_this_frame:
            # No eyes detected, infer as loss of eye contact for this frame
            timestamp = frame_count / fps
            print(f'Loss of eye contact at {timestamp} seconds')
            eye_contact_losses.append(timestamp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return eye_contact_losses

eye_contact_loss_timestamps = process_video('CVTeam/shortened_recording.mp4')
print('Timestamps where eye contact was lost:', eye_contact_loss_timestamps)
