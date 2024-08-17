import math
import cv2 as cv
import mediapipe as mp
from time import sleep
import pyautogui # type: ignore

capture = cv.VideoCa

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

keyboardLayout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'BK'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', 'SP'] ,
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'EN']
];

keyLayoutHW = []

keyHeight, keyWidth = 80, 80

for i in range(0, 3):
    for id, j in enumerate(keyboardLayout[i]):
        keyLayoutHW.append([j, keyHeight*id + 20, keyWidth*i + 20])

def drawKeyborad(frame, key, r = 0, g = 0, b = 0):
    cv.rectangle(frame, (key[1], key[2]), (key[1]+60, key[2]+60), (r, g, b), -1)
    cv.putText(frame, key[0], (key[1]+20, key[2]+40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cx, cy, c12x, c12y = 0,0,0,0

while True:
    success, frame = capture.read()
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (1000, 750)) 
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    if result.multi_hand_landmarks:
        for oneHandLm in result.multi_hand_landmarks:
            
            mpDraw.draw_landmarks(frame, oneHandLm, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(oneHandLm.landmark):
                fh, fw, fc = frame.shape

                if id == 12:
                    c12x, c12y = int(lm.x * fw), int(lm.y * fh)

                if id == 8:
                    for key in keyLayoutHW:
                        cx, cy = int(lm.x * fw), int(lm.y * fh)

                        distance8n12 = int(math.sqrt(math.pow(cx - c12x, 2) + math.pow(cy-c12y, 2)))

                        if key[1] <= cx  <= key[1] + keyWidth and key[2] <= cy <= key[2] + keyHeight:
                            if distance8n12 < 50:
                                if key[0] == "BK":
                                    pyautogui.press('backspace')
                                elif key[0] == "EN":
                                    pyautogui.press('enter')
                                elif key[0] == "SP":
                                    pyautogui.press('space')
                                else:
                                    pyautogui.press(key[0])
                                sleep(0.2)

                            drawKeyborad(frame, key, g=155)
                        else:
                            drawKeyborad(frame, key, g=255)
                else:
                    drawKeyborad(frame, key, g=255)
    else:
        for key in keyLayoutHW:
            cv.rectangle(frame, (key[1], key[2]), (key[1]+60, key[2]+60), (0, 255, 0), -1)
            cv.putText(frame, key[0], (key[1]+20, key[2]+40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # drawKeys(frame)

    cv.imshow("Image", frame)
    cv.waitKey(1)
