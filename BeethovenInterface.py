import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import time
import pyautogui
import os
import torch.nn as nn
import torch


CLOSED_OPENED_TIME = 1

#To not crush in corners
pyautogui.FAILSAFE = False

class BeethovenTracker():
    def __init__(self):
        #FurEliseNN
        self.eye_model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 8, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 4, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(484, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 1),
        nn.Sigmoid())
        model_path = os.path.join(os.path.dirname(__file__), 'FurEliseNN9717.tar')
        self.eye_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.eye_model.eval()

        #Variables
        self.running = True
        self.latest_result = None
        self.eyes_closed_time = 0
        self.eyes_opened_time = 0
        self.ready_to_blink = False
        self.last_left_eye_center = None
        self.last_right_eye_center = None
        self.normal = (0,0)
        self.left_closed_time = 0
        self.left_closed_at_first_time = True
        self.last_left_eye_closed_time = 0  
        self.PAD = 20
        
        #For right eye
        self.right_closed_time = 0
        self.right_closed_at_first_time = True
        self.last_right_eye_closed_time = 0
        self.can_do_sepcial = True

        #Mediapipe settings
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.get_result)
        
    def get_result(self, result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def click_from_blink(self):
        pyautogui.click()

    def right_click_from_blink(self):
        pyautogui.rightClick()
        
    def run_tracker(self):
        with mp.tasks.vision.FaceLandmarker.create_from_options(self.options) as landmarker:
            if self.running:
                self.capture = cv2.VideoCapture(0)
            while self.running:
                    current_time = time.time()
                    ret, frame = self.capture.read()
                    if not ret:
                        break;

                    numpy_frame_from_opencv = frame
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    frame_timestamp_ms = int(time.time() * 1000)
                    landmarker.detect_async(mp_image, frame_timestamp_ms) 

                    #Blink capture
                    if self.latest_result and self.latest_result.face_landmarks:
                        for face_landmarks in self.latest_result.face_landmarks:

                            self.last_left_eye_center = face_landmarks[473]
                            self.last_right_eye_center = face_landmarks[468]

                            h, w, _ = frame.shape

                            #print("")
                            #Left eye
                            lp1 = 388
                            lp2 = 384
                            lp3 = 390
                            lp4 = 381
                            l_pts = [lp1, lp2, lp3, lp4]
                            xs = [int(face_landmarks[i].x * w) for i in l_pts]
                            ys = [int(face_landmarks[i].y * h) for i in l_pts]
                            x1, x2 = max(0, min(xs) - self.PAD), min(w, max(xs) + self.PAD)
                            y1, y2 = max(0, min(ys) - self.PAD), min(h, max(ys) + self.PAD)
                            eye_crop = frame[y1:y2, x1:x2]
                            if eye_crop.size == 0:
                                continue
                            eye_crop = cv2.resize(eye_crop, (90, 90))
                            eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                            tensor = torch.from_numpy(eye_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            with torch.no_grad():
                                predictiont = self.eye_model(tensor).item()
                            is_left_closed = predictiont < 0.6
                            #print(f"Prob: {predictiont:.3f} |L Closed: {is_left_closed}")
                            
                            #Right eye
                            rp1 = 157
                            rp2 = 161
                            rp3 = 154
                            rp4 = 163
                            r_pts = [rp1, rp2, rp3, rp4]
                            xs = [int(face_landmarks[i].x * w) for i in r_pts]
                            ys = [int(face_landmarks[i].y * h) for i in r_pts]
                            x1, x2 = max(0, min(xs) - self.PAD), min(w, max(xs) + self.PAD)
                            y1, y2 = max(0, min(ys) - self.PAD), min(h, max(ys) + self.PAD)
                            eye_crop = frame[y1:y2, x1:x2]
                            if eye_crop.size == 0:
                                continue
                            eye_crop = cv2.resize(eye_crop, (90, 90))
                            eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                            tensor = torch.from_numpy(eye_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            with torch.no_grad():
                                predictiont = self.eye_model(tensor).item()
                            is_right_closed = predictiont < 0.6
                            #print(f"Prob: {predictiont:.3f} |R Closed: {is_right_closed}")

                            #Boths closed
                            if (self.ready_to_blink == False and (is_left_closed) and (is_right_closed)):
                                self.eyes_closed_time = current_time
                                self.ready_to_blink = True
                            #Both opened after closed
                            elif (self.ready_to_blink and (not is_left_closed) and (not is_right_closed)):
                                self.eyes_opened_time = current_time
                                if abs(self.eyes_closed_time - self.eyes_opened_time) > CLOSED_OPENED_TIME:
                                    if abs(self.last_left_eye_closed_time - current_time) > CLOSED_OPENED_TIME*0.2 and abs(self.last_right_eye_closed_time - current_time) > CLOSED_OPENED_TIME*0.2:
                                        self.click_from_blink()
                                self.ready_to_blink = False
                            #Left closed right opened
                            elif (is_left_closed and not is_right_closed):
                                self.last_left_eye_closed_time = current_time
                                if self.left_closed_at_first_time:
                                    self.left_closed_at_first_time = False
                                    self.left_closed_time = current_time
                                    continue
                                if abs(self.left_closed_time - current_time) > CLOSED_OPENED_TIME:

                                    offset_x = self.last_right_eye_center.x - self.normal[0]
                                    offset_y = self.last_right_eye_center.y - self.normal[1]
                                    DEAD_ZONE = 0.0
                                    SPEED = 5000
                                    if (abs(offset_x) > DEAD_ZONE) or (abs(offset_y) > DEAD_ZONE):
                                        lr = int(-offset_x * SPEED)
                                        ud = int(offset_y * SPEED)
                                        pyautogui.moveRel(lr, ud)
                            #Left opened right closed
                            elif (not is_left_closed) and (is_right_closed):
                                self.last_right_eye_closed_time = current_time
                                if self.right_closed_at_first_time:
                                    self.right_closed_at_first_time = False
                                    self.right_closed_time = current_time
                                    continue
                                if abs(self.right_closed_time - current_time) > 1:
                                    #Right mouse
                                    if self.can_do_sepcial:
                                        self.right_click_from_blink()
                                        self.can_do_sepcial = False
                                    pass
                            #Take base position
                            elif (not is_left_closed) and (not is_right_closed):
                                self.ready_to_blink = False
                                self.left_closed_at_first_time = True
                                self.right_closed_at_first_time = True
                                self.can_do_sepcial = True
                                self.normal = (self.last_right_eye_center.x, self.last_right_eye_center.y)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_tracker = BeethovenTracker()
    my_tracker.run_tracker()
