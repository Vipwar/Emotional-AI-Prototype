import cv2
from deepface import DeepFace
import mediapipe as mp
import math

# Khởi tạo Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Hàm tính khoảng cách giữa 2 điểm
def euclidean_dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # DeepFace Emotion Recognition
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except:
        emotion = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Vẽ landmark
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Nhận diện nháy mắt (khoảng cách giữa mí trên và mí dưới)
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            left_eye_dist = euclidean_dist(left_eye_top, left_eye_bottom)

            if left_eye_dist < 0.015:
                cv2.putText(frame, "Left eye blink", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Nhận diện cười (khoảng cách giữa 2 khoé miệng)
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]
            mouth_dist = euclidean_dist(mouth_left, mouth_right)

            if mouth_dist > 0.07:
                cv2.putText(frame, "Smile Detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Emotion + Landmarks + Motion Detect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
