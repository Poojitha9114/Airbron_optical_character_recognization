import cv2
import numpy as np
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing settings
brush_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
eraser_color = (0, 0, 0)
brush_thickness = 7
eraser_thickness = 50
color_index = 0

# Webcam setup
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Error: Cannot access webcam")
    exit()

canvas = np.zeros_like(frame)
prev_x, prev_y = 0, 0
drawing = False
color_change_cooldown = 0

# Finger IDs for landmark comparison
tip_ids = [4, 8, 12, 16, 20]

def fingers_up(hand):
    fingers = []

    # Thumb
    if hand.landmark[tip_ids[0]].x < hand.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand.landmark[tip_ids[id]].y < hand.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    if canvas is None:
        canvas = np.zeros_like(frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get fingers state
        fingers = fingers_up(hand_landmarks)

        # Coordinates
        index_x = int(hand_landmarks.landmark[8].x * w)
        index_y = int(hand_landmarks.landmark[8].y * h)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        # Detect pinch gesture for color change
        if get_distance(thumb_tip, index_tip) < 0.03:
            if color_change_cooldown == 0:
                color_index = (color_index + 1) % len(brush_colors)
                color_change_cooldown = 20  # 20 frames cooldown

        # Reset cooldown
        if color_change_cooldown > 0:
            color_change_cooldown -= 1

        # Drawing gesture (only index finger up)
        if fingers == [0, 1, 0, 0, 0]:
            drawing = True
            color = brush_colors[color_index]
            thickness = brush_thickness

        # Erasing gesture (all fingers up)
        elif fingers == [1, 1, 1, 1, 1]:
            drawing = True
            color = eraser_color
            thickness = eraser_thickness

        else:
            drawing = False
            prev_x, prev_y = 0, 0

        if drawing:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_x, index_y
            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), color, thickness)
            prev_x, prev_y = index_x, index_y
        else:
            prev_x, prev_y = 0, 0

    # Combine canvas with webcam
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, canvas_fg)

    # UI
    for i, col in enumerate(brush_colors):
        cv2.rectangle(frame, (20 + i * 70, 10), (70 + i * 70, 60), col, -1)
        if i == color_index:
            cv2.rectangle(frame, (20 + i * 70, 10), (70 + i * 70, 60), (255, 255, 255), 2)

    # Eraser
    cv2.rectangle(frame, (240, 10), (310, 60), (50, 50, 50), -1)
    cv2.putText(frame, 'E', (255, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Help text
    cv2.putText(frame, 'Draw: Index up | Erase: All up | Pinch: Change Color', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Airborne Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
