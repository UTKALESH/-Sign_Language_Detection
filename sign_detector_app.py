# sign_detector_app.py
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import threading # To prevent GUI freeze during prediction
import time # For adding slight delay if needed
import os

# --- Configuration ---
MODEL_PATH = 'saved_model/sign_language_model.keras' # Use .keras extension
IMG_HEIGHT = 64 # Must match the training configuration
IMG_WIDTH = 64
# Determine input channels based on how the model was trained (check model_training.py output)
# Example: Determine from loaded model later, or set manually:
# INPUT_CHANNELS = 3 # for RGB
# INPUT_CHANNELS = 1 # for Grayscale

# Time window (6 PM to 10 PM)
START_HOUR = 18
END_HOUR = 22

# --- Load Model and Class Names ---
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

    # Infer input shape and class names if possible (more robust)
    try:
        # Attempt to infer input shape details
        model_input_shape = model.input_shape # e.g., (None, 64, 64, 3)
        IMG_HEIGHT = model_input_shape[1]
        IMG_WIDTH = model_input_shape[2]
        INPUT_CHANNELS = model_input_shape[3]
        print(f"Inferred input shape: H={IMG_HEIGHT}, W={IMG_WIDTH}, C={INPUT_CHANNELS}")

        # Try to get class names from the training directory structure
        # Assumes 'data/train' exists relative to this script or model_training.py
        train_dir = 'data/train' # Adjust path if necessary
        if os.path.exists(train_dir):
             CLASS_NAMES = sorted(os.listdir(train_dir))
             print(f"Inferred class names: {CLASS_NAMES}")
        else:
             # Fallback: Define class names manually if directory not found
             # YOU MUST EDIT THIS LIST TO MATCH YOUR TRAINED CLASSES
             CLASS_NAMES = ['A', 'B', 'C', 'L', 'O', 'V', 'Y'] # Example
             print(f"Warning: Training directory not found. Using default class names: {CLASS_NAMES}")
             print("Please ensure these match your trained model!")

        NUM_CLASSES = len(CLASS_NAMES)
        if model.output_shape[1] != NUM_CLASSES:
             raise ValueError(f"Model output units ({model.output_shape[1]}) do not match number of class names ({NUM_CLASSES})!")

    except Exception as e:
        print(f"Error inferring model details: {e}")
        print("Please ensure MODEL_PATH is correct and configuration matches training.")
        # Fallback to manual config if inference fails
        INPUT_CHANNELS = 3 # Set manually if needed (1 for grayscale, 3 for RGB)
        CLASS_NAMES = ['A', 'B', 'C', 'L', 'O', 'V', 'Y'] # Example - EDIT THIS
        print("Using fallback configuration.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists at the specified path and required libraries (like h5py) are installed.")
    model = None # Indicate model loading failed
    CLASS_NAMES = [] # No classes if model fails

# --- Global Variables ---
cap = None # OpenCV VideoCapture object
is_realtime_running = False
last_prediction = "None"
is_operational_time = False

# --- Helper Functions ---
def preprocess_image(img, target_height, target_width, channels):
    """Prepares an image for model prediction."""
    # Resize
    img_resized = cv2.resize(img, (target_width, target_height))

    # Handle color channels
    if channels == 1: # Grayscale
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        elif len(img_resized.shape) == 2:
             img_gray = img_resized # Already grayscale
        else: # Unexpected shape, handle gracefully
            print(f"Warning: Unexpected image shape {img_resized.shape}, attempting grayscale conversion.")
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) > 2 else img_resized

        img_processed = img_gray.astype('float32') / 255.0
        img_processed = np.expand_dims(img_processed, axis=-1) # Add channel dimension
    else: # RGB
        if len(img_resized.shape) == 2: # Input is grayscale, need to convert to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 1: # Input has 1 channel, convert to RGB
             img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        else:
            # Assume BGR, convert to RGB (if model trained on RGB)
            # If model trained on BGR, skip cvtColor
            # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Use if needed
            img_rgb = img_resized # Assuming model expects BGR or handled internally

        img_processed = img_rgb.astype('float32') / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch

def predict_sign(image_batch):
    """Predicts the sign from a preprocessed image batch."""
    if model is None:
        return "Model not loaded", 0.0

    try:
        prediction = model.predict(image_batch)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100 # Percentage

        if 0 <= predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            return predicted_class_name, confidence
        else:
            print(f"Error: Predicted index {predicted_class_index} out of bounds for CLASS_NAMES (len={len(CLASS_NAMES)}).")
            return "Prediction Error", 0.0

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error", 0.0

def check_operational_time():
    """Checks if current time is within the operational window."""
    global is_operational_time
    now = datetime.datetime.now().time()
    start_time = datetime.time(START_HOUR, 0)
    end_time = datetime.time(END_HOUR, 0)
    is_operational_time = start_time <= now <= end_time
    update_status_label()
    # Schedule the next check
    root.after(60000, check_operational_time) # Check every 60 seconds

def update_status_label():
    """Updates the GUI label for operational status."""
    if is_operational_time:
        status_label.config(text="Status: Operational (6 PM - 10 PM)", fg="green")
        # Enable buttons if they were disabled
        upload_button.config(state=tk.NORMAL if model else tk.DISABLED)
        realtime_button.config(state=tk.NORMAL if model else tk.DISABLED)
    else:
        status_label.config(text="Status: Inactive (Operational 6 PM - 10 PM)", fg="red")
        # Disable prediction-related buttons
        upload_button.config(state=tk.DISABLED)
        # Stop real-time feed if it's running outside hours
        if is_realtime_running:
            toggle_realtime_feed() # This will stop the feed
        realtime_button.config(state=tk.DISABLED)

# --- GUI Functions ---

def upload_image():
    """Handles image upload, preprocessing, and prediction."""
    if not is_operational_time:
        messagebox.showwarning("Inactive", "Sign detection is only operational between 6 PM and 10 PM.")
        return
    if model is None:
         messagebox.showerror("Error", "Model is not loaded. Cannot perform prediction.")
         return

    filepath = filedialog.askopenfilename(
        title="Select Sign Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not filepath:
        return

    try:
        # Load image using OpenCV (consistent with video feed)
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            messagebox.showerror("Error", f"Could not read image file: {filepath}")
            return

        # Preprocess for model
        image_batch = preprocess_image(img_cv, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)

        # --- Prediction in a separate thread to avoid GUI freeze ---
        def run_prediction():
            global last_prediction
            predicted_class, confidence = predict_sign(image_batch)
            last_prediction = f"{predicted_class} ({confidence:.1f}%)"
            prediction_label.config(text=f"Prediction: {last_prediction}")

            # Display the uploaded image (resized for display)
            img_display = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # Convert for PIL
            img_pil = Image.fromarray(img_display)
            img_pil.thumbnail((video_label.winfo_width(), video_label.winfo_height() - 50)) # Resize proportionally
            img_tk = ImageTk.PhotoImage(img_pil)

            # Update GUI (must be done in main thread)
            root.after(0, update_uploaded_image_display, img_tk)

        def update_uploaded_image_display(img_tk_arg):
             # Keep a reference to avoid garbage collection
            video_label.imgtk = img_tk_arg
            video_label.config(image=img_tk_arg, text="") # Display image, clear placeholder text


        # Start prediction thread
        prediction_thread = threading.Thread(target=run_prediction, daemon=True)
        prediction_thread.start()
        prediction_label.config(text="Prediction: Processing...")


    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")
        print(f"Error details: {e}") # Log detailed error
        prediction_label.config(text="Prediction: Error")
        video_label.config(image='', text="Error loading image") # Clear image display


def update_realtime_feed():
    """Captures frame, preprocesses, predicts, and updates GUI."""
    global last_prediction
    if not is_realtime_running or not is_operational_time or cap is None or not cap.isOpened():
        return # Stop updating if flag is off, outside hours, or camera error

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera.")
        # Optionally try to reopen camera or show error message
        # toggle_realtime_feed() # Stop the feed on error
        root.after(50, update_realtime_feed) # Try again shortly
        return

    # Flip frame horizontally (mirror effect) - optional
    frame = cv2.flip(frame, 1)

    # --- Preprocessing ---
    # Create a region of interest (ROI) box on the frame (optional)
    h, w, _ = frame.shape
    roi_x, roi_y, roi_w, roi_h = w // 2 - 100, h // 2 - 100, 200, 200
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Prepare ROI for the model
    image_batch = preprocess_image(roi, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)

    # --- Prediction ---
    predicted_class, confidence = predict_sign(image_batch)
    last_prediction = f"{predicted_class} ({confidence:.1f}%)"
    prediction_label.config(text=f"Prediction: {last_prediction}") # Update prediction label immediately

    # --- Display ---
    # Convert frame for Tkinter display
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    # Update video label
    video_label.imgtk = imgtk # Keep reference
    video_label.config(image=imgtk, text="")

    # Schedule the next frame update
    root.after(20, update_realtime_feed) # Update approx 50fps (adjust delay as needed)

def toggle_realtime_feed():
    """Starts or stops the real-time video feed."""
    global is_realtime_running, cap

    if not is_operational_time and not is_realtime_running:
         messagebox.showwarning("Inactive", "Sign detection is only operational between 6 PM and 10 PM.")
         return
    if model is None and not is_realtime_running:
         messagebox.showerror("Error", "Model is not loaded. Cannot start real-time feed.")
         return

    if is_realtime_running:
        # Stop the feed
        is_realtime_running = False
        realtime_button.config(text="Start Real-time")
        prediction_label.config(text="Prediction: Off")
        video_label.config(image='', text="Camera Feed Stopped") # Clear video display
        if cap is not None:
            cap.release()
            print("Camera released.")
            cap = None
    else:
        # Start the feed
        try:
            cap = cv2.VideoCapture(0) # Use camera 0 (default)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open webcam.")
                cap = None
                return

            print("Camera opened successfully.")
            is_realtime_running = True
            realtime_button.config(text="Stop Real-time")
            prediction_label.config(text="Prediction: Starting...")
            video_label.config(image='', text="Initializing Camera...") # Placeholder text
            # Start the update loop
            root.after(100, update_realtime_feed) # Start after a short delay

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            is_realtime_running = False
            realtime_button.config(text="Start Real-time")
            if cap:
                cap.release()
                cap = None


def on_closing():
    """Handles window closing event."""
    global is_realtime_running, cap
    print("Closing application...")
    is_realtime_running = False # Ensure feed stops
    time.sleep(0.1) # Give time for loops to potentially exit
    if cap is not None:
        cap.release()
        print("Camera released on exit.")
    root.destroy()


# --- GUI Setup ---
root = tk.Tk()
root.title("Sign Language Detector")
root.geometry("800x650") # Adjust size as needed
root.protocol("WM_DELETE_WINDOW", on_closing) # Handle closing properly


# --- Status Label ---
status_label = Label(root, text="Status: Initializing...", font=("Arial", 12), fg="orange")
status_label.pack(pady=10)

# --- Video/Image Display Area ---
# Use a Label to display video frames or uploaded images
video_label = Label(root, text="Upload an image or start real-time feed", bg="lightgrey")
video_label.pack(pady=10, padx=10, fill="both", expand=True)


# --- Prediction Label ---
prediction_label = Label(root, text="Prediction: None", font=("Arial", 16, "bold"))
prediction_label.pack(pady=10)

# --- Control Buttons Frame ---
button_frame = Frame(root)
button_frame.pack(pady=15)

upload_button = Button(button_frame, text="Upload Image", command=upload_image, width=15, height=2, state=tk.DISABLED)
upload_button.pack(side=tk.LEFT, padx=20)

realtime_button = Button(button_frame, text="Start Real-time", command=toggle_realtime_feed, width=15, height=2, state=tk.DISABLED)
realtime_button.pack(side=tk.LEFT, padx=20)

# --- Initial Checks ---
if model is None:
     status_label.config(text="Status: Model Load FAILED. Cannot operate.", fg="red")
     messagebox.showerror("Startup Error", "Failed to load the sign language model. Please check the console for errors and ensure the model file exists.")
else:
     check_operational_time() # Start the time check loop and set initial status


# --- Start GUI ---
root.mainloop()