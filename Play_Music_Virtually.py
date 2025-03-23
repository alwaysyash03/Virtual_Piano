import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize pygame and mixer
pygame.init()
pygame.mixer.init()

# Load piano sounds
keys = {
    'C': pygame.mixer.Sound('sounds/C.wav'),
    'D': pygame.mixer.Sound('sounds/D.wav'),
    'E': pygame.mixer.Sound('sounds/E.wav'),
    'F': pygame.mixer.Sound('sounds/F.wav'),
    'G': pygame.mixer.Sound('sounds/G.wav'),
   # 'A': pygame.mixer.Sound('sounds/A.wav'),
   # 'B': pygame.mixer.Sound('sounds/B.wav')
}
# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to play sound based on key
def play_sound(key):
    if key in keys:
        keys[key].play()

# Define piano key areas on the screen
key_areas = {
    'C': (50, 100, 150, 300),
    'D': (150, 200, 150, 300),
    'E': (250, 300, 150, 300),
    'F': (350, 400, 150, 300),
    'G': (450, 500, 150, 300),
    'A': (550, 600, 150, 300),
    'B': (650, 700, 150, 300)
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if index finger tip is in any key area
            for key, (x1, x2, y1, y2) in key_areas.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    play_sound(key)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Virtual Piano', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
