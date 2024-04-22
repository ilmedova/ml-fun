import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

import emoji
from PIL import Image, ImageDraw, ImageFont

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]

        out = ""

        if pred == "happy":
            out = emoji.emojize(':grinning_face_with_big_eyes:', language='alias')
        elif pred == "lucky":
            out = emoji.emojize(':thumbs_up:', language='alias')
        elif pred == "unlucky":
            out = emoji.emojize(':thumbs_down:', language='alias')
        elif pred == "crying":
            out = emoji.emojize(':loudly_crying_face:', language='alias')
        else:
            out = emoji.emojize(':neutral_face:', language='alias')

        emoji_size = 100  # Adjust the size of the emoji as needed

        # Create a larger emoji image
        emoji_image = Image.new('RGB', (emoji_size, emoji_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(emoji_image)

        # Load the font and set the text
        font_size = emoji_size // 2  # Adjust font size based on emoji size
        font = ImageFont.truetype("OpenSansEmoji.ttf", font_size)
        draw.text((emoji_size // 4, emoji_size // 4), out, font=font, fill=(0, 0, 0))

        # Convert the emoji image to a numpy array and then to BGR format
        emoji_cv2 = cv2.cvtColor(np.array(emoji_image), cv2.COLOR_RGB2BGR)

        # Adjust the position where the emoji is pasted onto the frame
        emoji_height, emoji_width, _ = emoji_cv2.shape
        frm[10:10 + emoji_height, 10:10 + emoji_width] = emoji_cv2

    #drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS, landmark_drawing_spec=drawing.DrawingSpec((0,255,50), 1, 1), connection_drawing_spec=drawing.DrawingSpec((0,200,20), 1, 1))
    #drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    #drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break