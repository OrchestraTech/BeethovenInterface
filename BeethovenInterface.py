import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import pyautogui
import winsound
import os


CLOSED_OPENED_TIME = 1

#Избежание крашей при нахождении курсора в углах
pyautogui.FAILSAFE = False

class BeathovenTracker():
    def __init__(self):
        
        #Чудо техники EAR
        self.ear_time = time.time()
        self.left_open_ear = 0.3
        self.right_open_ear = 0.3
        self.is_ready_to_take_open_ear = True
        self.left_open_ear_data = []
        self.right_open_ear_data = []
        self.is_ready_to_compute_ear = 0
        #Чудо техники EAR

        #Переменные
        self.we_are_starting = True
        self.running = True#
        self.latest_result = None#
        self.eyes_closed_time = 0#
        self.eyes_opened_time = 0#
        self.ready_to_blink = False#
        self.last_left_eye_center = None#
        self.last_right_eye_center = None#
        self.normal = (0,0)
        self.left_closed_time = 0
        self.left_closed_at_first_time = True
        self.last_left_eye_closed_time = 0       
        
        #Количество стадий закрытости правого глаза
        self.right_studies_count = 0
        self.right_closed_time = 0
        self.right_closed_at_first_time = True
        self.last_right_eye_closed_time = 0
        self.can_do_sepcial = True

        #Настройки для гугловской модели
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        #Настройка гугловской модели
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
                    if abs(current_time - self.ear_time) > 20 and self.is_ready_to_take_open_ear:
                        self.is_ready_to_take_open_ear = False
                        self.is_ready_to_compute_ear += 1
                        winsound.Beep(1000,500)
                    ret, frame = self.capture.read()
                    if not ret:
                        break;

                    numpy_frame_from_opencv = frame
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    frame_timestamp_ms = int(time.time() * 1000)
                    landmarker.detect_async(mp_image, frame_timestamp_ms) 

                    if self.we_are_starting:
                        self.we_are_starting = False
                        print("We are starting")

                    #Ловим моргание
                    if self.latest_result and self.latest_result.face_landmarks:
                        for face_landmarks in self.latest_result.face_landmarks:

                            #Берем левый и правый зрачки
                            self.last_left_eye_center = face_landmarks[473]
                            self.last_right_eye_center = face_landmarks[468]

                            #Левый глаз
                            lp1 = face_landmarks[362]
                            lp2 = face_landmarks[385]
                            lp3 = face_landmarks[387]
                            lp4 = face_landmarks[263]
                            lp5 = face_landmarks[373]
                            lp6 = face_landmarks[380]
                            A = abs(lp2.y - lp6.y)
                            B = abs(lp3.y - lp5.y)
                            C = abs(lp1.x - lp4.x)
                            self.left_ear = (A + B)/(2*C)
                            
                            
                            
                            #Правый глаз
                            rp1 = face_landmarks[33]
                            rp2 = face_landmarks[160]
                            rp3 = face_landmarks[158]
                            rp4 = face_landmarks[133]
                            rp5 = face_landmarks[153]
                            rp6 = face_landmarks[144]
                            A = abs(rp2.y - rp6.y)
                            B = abs(rp3.y - rp5.y)
                            C = abs(rp1.x - rp4.x)
                            self.right_ear = (A + B)/(2*C)
            
                            if self.is_ready_to_take_open_ear:
                                self.left_open_ear_data.append(self.left_ear)
                                self.right_open_ear_data.append(self.right_ear)

                            if (self.is_ready_to_compute_ear == 1):
                                #self.left_open_ear = sum(self.left_open_ear_data)/len(self.left_open_ear_data)
                                #self.right_open_ear = sum(self.right_open_ear_data)/len(self.right_open_ear_data)
                                self.left_open_ear = max(self.left_open_ear_data)
                                self.right_open_ear = max(self.right_open_ear_data)
                                self.is_ready_to_compute_ear += 1
                                print(self.left_open_ear)
                                print(self.right_open_ear)
                            
                            if self.is_ready_to_compute_ear != 2:
                                continue

                            is_left_closed = True if self.left_ear < self.left_open_ear * 0.5 else False
                            is_right_closed = True if self.right_ear < self.right_open_ear * 0.5 else False
                            #Оба закрыты
                            if (self.ready_to_blink == False and (is_left_closed) and (is_right_closed)):
                                print("BOTH CLOSED")
                                print(f"LEFT DISTANCE: {self.left_ear}")
                                print(f"RIGHT DISTANCE: {self.right_ear}")
                                self.eyes_closed_time = current_time
                                self.ready_to_blink = True
                            #Оба открыты после закрытости
                            elif (self.ready_to_blink and (not is_left_closed) and (not is_right_closed)):
                                print("BOTH OPENED AFTER CLOSE")
                                self.eyes_opened_time = current_time
                                if abs(self.eyes_closed_time - self.eyes_opened_time) > CLOSED_OPENED_TIME:
                                    if abs(self.last_left_eye_closed_time - current_time) > CLOSED_OPENED_TIME*0.2 and abs(self.last_right_eye_closed_time - current_time) > CLOSED_OPENED_TIME*0.2:
                                        self.click_from_blink()
                                self.ready_to_blink = False
                            #Левый открыт, правый закрыт
                            elif (is_left_closed and not is_right_closed):
                                print(f"LEFT CLOSED RIGHT OPENED {self.left_ear}")
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
                                        print(f"OFFSET X: {offset_x}")
                                        print(f"OFFSET Y: {offset_y}")
                                        lr = int(-offset_x * SPEED)
                                        ud = int(offset_y * SPEED)
                                        pyautogui.moveRel(lr, ud)
                            #Правый открыт, левый закрыт
                            elif (not is_left_closed) and (is_right_closed):
                                print(f"|||LEFT OPENED RIGHT CLOSED {self.right_ear}")
                                self.last_right_eye_closed_time = current_time
                                if self.right_closed_at_first_time:
                                    self.right_closed_at_first_time = False
                                    self.right_closed_time = current_time
                                    continue
                                if abs(self.right_closed_time - current_time) > 1:
                                    #Нажатие правой кнопкой мыши
                                    if self.can_do_sepcial:
                                        self.right_click_from_blink()
                                        self.can_do_sepcial = False
                                    print("FIRST")
                                    pass
                            #Берем нормаль
                            elif (not is_left_closed) and (not is_right_closed):
                                print(f"BOTH OPENED LEFT DISTANCE: {self.left_ear} RIGHT DISTANCE: {self.right_ear}")
                                self.ready_to_blink = False
                                self.left_closed_at_first_time = True
                                self.right_closed_at_first_time = True
                                self.can_do_sepcial = True
                                self.normal = (self.last_right_eye_center.x, self.last_right_eye_center.y)
                                #print(f"NORMAL: {self.normal}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_tracker = BeathovenTracker()
    my_tracker.run_tracker()
