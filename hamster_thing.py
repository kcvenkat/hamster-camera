import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import math

#landmark initialization
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'hand_landmarker.task'
)
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'face_landmarker.task'
)

hand_options = vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = 'hand_landmarker.task'),
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 2
)
face_options = vision.FaceLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = 'face_landmarker.task'),
    running_mode = vision.RunningMode.IMAGE
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

FACE_POINTS = {
    "lip_top": 13,
    "lip_bottom": 14,
    "lip_left_corner": 61,
    "lip_right_corner": 291,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_eyebrow_outer": 70,
    "left_eyebrow_middle": 52,
    "left_eyebrow_inner": 107,
    "right_eyebrow_outer": 276,
    "right_eyebrow_middle": 282,
    "right_eyebrow_inner": 336,

}
#math to detect gestures
def distance(lm1, lm2, w, h):
    ax, ay = lm1.x * w, lm1.y * h
    bx, by = lm2.x * w, lm2.y * h
    return math.hypot(bx - ax, by - ay)

#fingertips together detection
def fingers_together(hand, w, h, threshold = 90):
    tips = [4, 8, 12, 16, 20]

    distances = []

    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            d = distance(hand[tips[i]], hand[tips[j]], w, h)
            distances.append(d)

    avg = sum(distances) / len(distances)  

    return avg <= threshold


def is_left_index_finger_downward_slope(hand, w, h):
    if len(hand) <= 8:
        return False

    base = hand[6]
    mid = hand[7]
    tip = hand[8]

    base_y = base.y * h
    mid_y = mid.y * h
    tip_y = tip.y * h

    return (mid_y - base_y) > 8 and (tip_y - mid_y) > 8


def is_right_index_finger_downward_slope(hand, w, h):
    if len(hand) <= 8:
        return False

    base = hand[6]
    mid = hand[7]
    tip = hand[8]

    base_y = base.y * h
    mid_y = mid.y * h
    tip_y = tip.y * h

    return (mid_y - base_y) > 8 and (tip_y - mid_y) > 8


def is_left_middle_finger_downward_slope(hand, w, h):
    if len(hand) <= 10:
        return False

    base = hand[10]
    mid = hand[11]
    tip = hand[12]

    base_y = base.y * h
    mid_y = mid.y * h
    tip_y = tip.y * h

    return (mid_y - base_y) > 8 and (tip_y - mid_y) > 8


def is_right_middle_finger_downward_slope(hand, w, h):
    if len(hand) <= 10:
        return False

    base = hand[10]
    mid = hand[11]
    tip = hand[12]

    base_y = base.y * h
    mid_y = mid.y * h
    tip_y = tip.y * h

    return (mid_y - base_y) > 8 and (tip_y - mid_y) > 8


def is_heart(hand_results, w, h, threshold=80):
    if len(hand_results.hand_landmarks) < 2:
        return False

    left_hand = None
    right_hand = None
    for hand_idx, hand in enumerate(hand_results.hand_landmarks):
        if hand_idx < len(hand_results.handedness):
            handedness_list = hand_results.handedness[hand_idx]
            if handedness_list and len(handedness_list) > 0:
                side = handedness_list[0].category_name.lower()
                if side == "left":
                    left_hand = hand
                elif side == "right":
                    right_hand = hand

    if left_hand is None or right_hand is None:
        return False

    left_index_tip = left_hand[8]
    right_index_tip = right_hand[8]
    left_middle_tip = left_hand[12]
    right_middle_tip = right_hand[12]

    index_distance = distance(left_index_tip, right_index_tip, w, h)
    middle_distance = distance(left_middle_tip, right_middle_tip, w, h)

    return index_distance < threshold and middle_distance < threshold


def get_face_axis(face, w, h):
    left_eye = face[33]
    right_eye = face[263]

    dx = (right_eye.x - left_eye.x) * w
    dy = (right_eye.y - left_eye.y) * h
    length = math.hypot(dx, dy)

    right_vec = (dx / length, dy / length)
    up_vec = (-dy / length, dx / length )

    return left_eye, right_vec, up_vec


def project_onto_face(lm, origin, right_vec, up_vec, w, h):
    dx = (lm.x - origin.x) * w
    dy = (lm.y - origin.y) * h
    u = dx * right_vec[0] + dy * right_vec[1]
    v = dx * up_vec[0]    + dy * up_vec[1]

    return u, v

def is_smirk(face, w, h):
    origin, right_vec, up_vec = get_face_axis(face, w, h)

    _, left_corner_y = project_onto_face(face[61], origin, right_vec, up_vec, w, h)
    _, right_corner_y = project_onto_face(face[291], origin, right_vec, up_vec, w, h)
    _, top_lip_y = project_onto_face(face[13], origin, right_vec, up_vec, w, h)

    left_drop = left_corner_y - top_lip_y
    right_drop = right_corner_y - top_lip_y

    asymmetry = abs(left_drop - right_drop)

    return asymmetry >= 7

#camera capture and loop
cap = cv2.VideoCapture(1)

#hamster images
hamster_images = {
    "open": cv2.imread('images/happy_hamster.png'),
    "smile": cv2.imread('images/smile_hamster.png'),
    "neutral": cv2.imread('images/neutral_hamster.png'),
    "scream": cv2.imread('images/scream_hamster.png'),
    "purse": cv2.imread('images/purse_hamster.png'),
    "smirk": cv2.imread('images/smirk_hamster.png'),
    "heart": cv2.imread('images/heart_hamster.png')
}

for img in hamster_images.values():
    if img is None:
        raise ValueError("One or more hamster images failed to load.")

while True:
    ret,frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb)

    hand_results = hand_landmarker.detect(mp_image)
    

    h, w, _ = frame.shape
    h, w, _ = frame.shape

    if not hand_results.hand_landmarks:
        is_heart_ok = False
        purse_hand_ok = False
    else:
        purse_hand_ok = any(
            fingers_together(hand, w, h) for hand in hand_results.hand_landmarks
        )
        is_heart_ok = is_heart(hand_results, w, h)

    print(f"is_heart={is_heart_ok}")

    for hand in hand_results.hand_landmarks:
        for idx, lm in enumerate(hand):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in connections:
            start = hand[connection.start]
            end = hand[connection.end]
            x1, y1 = int(start.x*w), int(start.y*h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    face_results = face_landmarker.detect(mp_image)

    for face in face_results.face_landmarks:
        if is_heart_ok:
            expression = "heart"
        else:
            mouth_open = distance(face[13], face[14], w, h)
            mouth_width = distance(face[61], face[291], w, h)

            left_corner_y = face[61].y * h
            right_corner_y = face[291].y * h
            lip_top_y = face[13].y * h
            lip_bottom_y = face[14].y * h

            left_to_top = abs(left_corner_y - lip_top_y)
            right_to_top = abs(right_corner_y - lip_top_y)

            left_to_bottom = abs(left_corner_y - lip_bottom_y)
            right_to_bottom = abs(right_corner_y - lip_bottom_y)
            if left_to_top > left_to_bottom and right_to_top > right_to_bottom and mouth_open > 40:
                expression = "scream"
            elif is_smirk(face, w, h):
                expression = "smirk"
            elif left_to_top < left_to_bottom and right_to_top < right_to_bottom and mouth_open > 40:
                expression = "open"
            elif mouth_width > 140:
                expression = "smile"
            elif mouth_width <= 110 and purse_hand_ok:
                expression = "purse"
            else:
                expression = "neutral"

        overlay = cv2.resize(hamster_images[expression], (300, 300))
        cv2.imshow('Hamster', overlay)
        
        for i in FACE_POINTS.values():
            lm = face[i]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        #Landmark labeling in case face landmarks need to be changed
        # for i, lm in enumerate(face):
        #     x, y = int(lm.x * w), int(lm.y * h)
        #     cv2.putText(frame, str(i), (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4, (0, 255, 255), 1)

    cv2.imshow('Hand and Face Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()