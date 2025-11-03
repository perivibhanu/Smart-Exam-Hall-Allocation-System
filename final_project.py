import serial
import time
import adafruit_fingerprint
from adafruit_fingerprint import Adafruit_Fingerprint

import face_recognition
import cv2
import numpy as np
import pickle
from pyzbar.pyzbar import decode
from picamera2 import Picamera2
import pyttsx3
import threading
import queue
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import CP437_FONT

# --- NEW: Google Sheets Imports ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- LED Display Setup ---
try:
    serial_led = spi(port=0, device=0, gpio=noop())
    device = max7219(serial_led, cascaded=4, block_orientation=-90, blocks_arranged_in_reverse_order=False)
    device.contrast(50)
    print("[INFO] LED Matrix display initialized.")
    led_device_initialized = True
except Exception as e:
    print(f"[ERROR] Failed to initialize LED Matrix display: {e}. Display functionality will be disabled.")
    led_device_initialized = False

# --- LED Message Queue and Thread ---
led_message_queue = queue.Queue()

def led_display_worker():
    """Worker function for the LED display thread."""
    while True:
        try:
            message = led_message_queue.get(timeout=0.1)
            if led_device_initialized:
                show_message(device, message, fill="white", font=CP437_FONT, scroll_delay=0.08)
                device.clear()
            led_message_queue.task_done()
        except queue.Empty:
            continue

# Start the LED worker thread
led_thread = threading.Thread(target=led_display_worker, daemon=True)
led_thread.start()

# --- Audio Setup ---
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    def speak(text):
        if led_device_initialized:
            led_message_queue.put(text)
        def run_speak():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run_speak).start()
except Exception as e:
    print(f"Failed to initialize pyttsx3: {e}. Audio output will be disabled.")
    def speak(text):
        if led_device_initialized:
            led_message_queue.put(text)

# --- Fingerprint Sensor Setup ---
FINGERPRINT_SERIAL_PORT = "/dev/serial0"
FINGERPRINT_BAUD_RATE = 57600
FINGERPRINT_TIMEOUT = 1

try:
    fingerprint_uart = serial.Serial(FINGERPRINT_SERIAL_PORT, baudrate=FINGERPRINT_BAUD_RATE, timeout=FINGERPRINT_TIMEOUT)
    finger = Adafruit_Fingerprint(fingerprint_uart)
    print("[INFO] Fingerprint sensor serial connection established.")
    if finger.verify_password() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to connect to fingerprint sensor.")
    print("[INFO] Fingerprint sensor verified and ready.")
    fingerprint_sensor_initialized = True
except Exception as e:
    print(f"[ERROR] Failed to initialize fingerprint sensor: {e}. Fingerprint functionality will be disabled.")
    fingerprint_sensor_initialized = False

# --- NEW: Google Sheets Configuration ---
google_docs_spreadsheet_name = 'exam'  # Replace with your spreadsheet name
oauth_json_location = 'attendance-tracker-api-26f4cb228830.json'   # Replace with your JSON file name
worksheet = None # Initialize a global variable for the worksheet

# --- Data Mappings ---
FINGERPRINT_ID_MAP = {
    1: "Bhanu",
    2: "Bhoumik",
    3: "Karthik",
}

PERSON_QR_DATA = {
    "Bhanu": ("this hallticket is verified for Bhanu", "Room 101", "Bench 5"),
    "Bhoumik": ("this hallticket is verified for Bhoumik", "Room 102", "Bench 12"),
    "Karthik": ("this hallticket is verified for Karthik", "Room 103", "Bench 8"),
}

# --- Face Recognition Setup ---
print("[INFO] loading encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
except FileNotFoundError:
    print("[ERROR] 'encodings.pickle' not found. Run 'image_capture.py' and the training script first.")
    known_face_encodings = []
    known_face_names = []

# --- Configuration for Performance ---
PROCESS_FRAME_INTERVAL = 5
CAPTURE_RESOLUTION = (480, 360)
RESIZE_FACTOR = 0.5
TOLERANCE = 0.45

# --- Camera and Display Setup ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": CAPTURE_RESOLUTION}))
picam2.start()
time.sleep(1)

# --- Variables ---
verified_person = None
face_location = None
is_face_verified = False
face_verified_time = None
last_unknown_time = 0
frame_count = 0
fingerprint_matched_name = None

# --- NEW: Helper function to log data to spreadsheet ---
def log_to_spreadsheet(person_name, status, room=None, bench=None):
    try:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        if room and bench:
            data_row = [current_time, person_name, status, room, bench]
        else:
            data_row = [current_time, person_name, status, "", ""]
        
        worksheet.append_row(data_row)
        print(f"[INFO] Logged to Google Sheet: {data_row}")
    except Exception as e:
        print(f"[ERROR] Failed to log to Google Sheet: {e}")

# --- Helper Functions ---
def verify_fingerprint():
    """Waits for and verifies a fingerprint."""
    print("\n[INFO] Place your finger on the sensor...")
    speak("Please place your finger on the sensor.")
    while True:
        try:
            if finger.get_image() != adafruit_fingerprint.OK:
                continue
            
            print("[INFO] Image captured.")
            if finger.image_2_tz(1) != adafruit_fingerprint.OK:
                print("[ERROR] Error converting image to template.")
                continue

            search_result = finger.finger_search()
            if search_result == adafruit_fingerprint.OK:
                finger_id = finger.finger_id
                print(f"[SUCCESS] Fingerprint matched! ID: {finger_id}")
                return FINGERPRINT_ID_MAP.get(finger_id)
            elif search_result == adafruit_fingerprint.NOTFOUND:
                print("[INFO] No matching fingerprint found. Please try again.")
                speak("No match found. Please try again.")
                time.sleep(2)
            else:
                print(f"[ERROR] Fingerprint search failed with code: {search_result}")
                time.sleep(2)

        except Exception as e:
            print(f"[ERROR] An error occurred during fingerprint verification: {e}")
            break
    return None

def process_frame_for_faces(frame, target_name):
    """Processes a frame to detect and identify faces."""
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    found_name = "Unknown"
    location = None
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            found_name = known_face_names[best_match_index]
        else:
            found_name = "Unknown"

    if face_locations:
        location = face_locations[0]
        if found_name == target_name:
            return target_name, location
        else:
            return "Unknown", location
    
    return None, None

def process_frame_for_qrs(frame):
    """Processes a frame to detect and decode QR codes."""
    decoded_objects = decode(frame)
    if decoded_objects:
        obj = decoded_objects[0]
        qr_data = obj.data.decode('utf-8')
        points_np = np.array([[p.x, p.y] for p in obj.polygon], dtype=np.int32)
        cv2.polylines(frame, [points_np], True, (0, 255, 0), 2)
        return qr_data
    return None

def draw_status_text(frame, text, color=(255, 255, 255)):
    """Draws status text on the OpenCV frame."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def draw_face_box(frame, name, location):
    """Draws a box and label around the detected face."""
    if location is None:
        return frame
        
    top, right, bottom, left = [int(coord / RESIZE_FACTOR) for coord in location]
    cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
    cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    return frame

# --- Main Loop ---
try:
    # --- NEW: Google Sheets Initialization ---
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(oauth_json_location, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(google_docs_spreadsheet_name)
    worksheet = spreadsheet.sheet1
    print("[INFO] Google Sheets connection established.")

    while True:
        # Step 1: Fingerprint Verification
        if not fingerprint_matched_name:
            fingerprint_matched_name = verify_fingerprint()
            if fingerprint_matched_name:
                print(f"[INFO] Fingerprint matched. Starting face verification for {fingerprint_matched_name}.")
                speak(f"Fingerprint verified as {fingerprint_matched_name}. Please show your face to the camera.")
            else:
                continue

        # Step 2: Face and QR Verification
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        status_text = ""
        status_color = (255, 255, 255)
        
        if is_face_verified:
            status_text = f"Step 3: Face verified as {verified_person}. Scan your QR code."
            status_color = (0, 255, 255)
            qr_data = process_frame_for_qrs(frame)
            
            if qr_data:
                qr_info = PERSON_QR_DATA.get(verified_person)
                
                if qr_info and qr_data == qr_info[0]:
                    room_no = qr_info[1]
                    bench_no = qr_info[2]
                    
                    status_text = f"VERIFIED! Room: {room_no}, Bench: {bench_no}"
                    status_color = (0, 255, 0)
                    print(f"[RESULT] QR matched! Successfully Verified. {room_no}, {bench_no}")
                    speak(f"Access granted. Go to {room_no}, {bench_no}.")
                    # --- NEW: Log successful verification ---
                    log_to_spreadsheet(verified_person, "Success", room_no, bench_no)
                    time.sleep(2)
                else:
                    status_text = "NOT VERIFIED! QR code does not match."
                    status_color = (0, 0, 255)
                    print(f"[RESULT] QR mismatched! Verification Failed.")
                    speak("QR mismatched. Verification Failed.")
                    # --- NEW: Log failed QR verification ---
                    log_to_spreadsheet(verified_person, "QR Mismatch")
                
                # Reset all states
                is_face_verified = False
                verified_person = None
                face_verified_time = None
                fingerprint_matched_name = None
                time.sleep(15)
            
            elif (time.time() - face_verified_time) > 30:
                print("[TIMEOUT] 30 seconds elapsed. Resetting to fingerprint scan.")
                speak("Time out. Please start the process again.")
                status_text = "TIMEOUT! Returning to fingerprint scan."
                status_color = (0, 0, 255)
                # Reset all states
                is_face_verified = False
                verified_person = None
                face_verified_time = None
                fingerprint_matched_name = None
        else:
            status_text = f"Step 2: Please show your face to the camera, {fingerprint_matched_name}."
            
            if frame_count % PROCESS_FRAME_INTERVAL == 0:
                name, face_location = process_frame_for_faces(frame, fingerprint_matched_name)
                
                if name == "Unknown":
                    if time.time() - last_unknown_time > 5:
                        speak("Face not recognized.")
                        last_unknown_time = time.time()
                    status_text = "Unknown face detected. Please try again."
                    status_color = (0, 0, 255)
                    # --- NEW: Log unknown face detection ---
                    log_to_spreadsheet(fingerprint_matched_name, "Face Mismatch")
                elif name:
                    is_face_verified = True
                    verified_person = name
                    face_verified_time = time.time()
                    print(f"[INFO] Face verified: {verified_person}. Please scan QR code now.")
                    speak(f"Face verified. Please scan your QR code.")

        draw_status_text(frame, status_text, status_color)
        if face_location:
            draw_face_box(frame, name if not is_face_verified else verified_person, face_location)
        
        cv2.imshow('Verification System', frame)
        if cv2.waitKey(1) == ord("q"):
            break
            
        frame_count += 1
finally:
    cv2.destroyAllWindows()
    picam2.stop()
    if led_device_initialized:
        device.clear()
        device.cleanup()
    if fingerprint_sensor_initialized:
        fingerprint_uart.close()
    try:
        engine.stop()
    except:
        pass
    