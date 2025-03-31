import mss
import numpy as np
import keyboard
import time
from ultralytics import YOLO
import pydirectinput
import cv2

model = YOLO(r"C:\Disk D\foldar nou\best.pt")

is_paused = False
CONFIDENCE_THRESHOLD = 0.6

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return frame

def process_frame(frame):
    results = model(frame, imgsz=512)
    detected_notes = []

    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                if conf > CONFIDENCE_THRESHOLD:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    detected_notes.append((center_x, center_y))

    detected_notes.sort(key=lambda note: (note[1], note[0]))

    return detected_notes

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    print("⏸️ Paused" if is_paused else "▶️ Resumed")

def instant_mouse_move(x, y):
    pydirectinput.moveTo(x, y)

def main():
    global is_paused
    last_f1_press = 0

    while True:
        if keyboard.is_pressed("F1") and time.time() - last_f1_press > 0.5:
            toggle_pause()
            last_f1_press = time.time()

        if is_paused:
            continue

        frame = capture_screen()
        detected_notes = process_frame(frame)

        if detected_notes:
            x, y = detected_notes[0]
            instant_mouse_move(x, y)
            print(f"🎯 Hovering over note at X:{x}, Y:{y}")
        else:
            print("❌ No notes found", end="\r")

        if keyboard.is_pressed("q"):
            print("\n🔴 Exiting...")
            break

if __name__ == "__main__":
    main()
