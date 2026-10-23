BeethovenInterface

Mediapipe Based Eye To Mouse Interface

Control your mouse cursor and clicks by blinking, using a webcam and MediaPipe Face Landmarker.

How it works

The program tracks the position of eyelids and pupils. Based on the Eye Aspect Ratio (EAR), it determines whether the eye is closed or open.

Both eyes closed (for more than 1 sec) -left mouse click.
Left eye closed (for more than 1 sec) - cursor movement tracking the right pupil.
Right eye closed (for more than 1 sec) — right mouse click.

Installation and Usage

Install dependencies:

```bash
pip install mediapipe opencv-python pyautogui numpy
```

Download the face_landmarker.task model from the official MediaPipe website and place it in the same folder as the script.

Run:

```bash
python beathoven.py
```

A sound signal will play on startup. Look at the camera for 20 seconds - the calibration of the open-eye position is in progress.

Limitations

Works only on Windows.
Requires a webcam.
