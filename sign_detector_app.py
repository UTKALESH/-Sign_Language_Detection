
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import threading 
import time
import os


MODEL_PATH = 'saved_model/sign_language_model.keras' 
IMG_HEIGHT = 64 
IMG_WIDTH = 64


START_HOUR = 18
END_HOUR = 22


try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")


    try:
       
        model_input_shape = model.input_shape 
        IMG_HEIGHT = model_input_shape[1]
        IMG_WIDTH = model_input_shape[2]
        INPUT_CHANNELS = model_input_shape[3]
        print(f"Inferred input shape: H={IMG_HEIGHT}, W={IMG_WIDTH}, C={INPUT_CHANNELS}")


        train_dir = 'data/train' 
        if os.path.exists(train_dir):
             CLASS_NAMES = sorted(os.listdir(train_dir))
             print(f"Inferred class names: {CLASS_NAMES}")
        else:
           
             CLASS_NAMES = ['A', 'B', 'C', 'L', 'O', 'V', 'Y'] 
             print(f"Warning: Training directory not found. Using default class names: {CLASS_NAMES}")
             print("Please ensure these match your trained model!")

        NUM_CLASSES = len(CLASS_NAMES)
        if model.output_shape[1] != NUM_CLASSES:
             raise ValueError(f"Model output units ({model.output_shape[1]}) do not match number of class names ({NUM_CLASSES})!")

    except Exception as e:
        print(f"Error inferring model details: {e}")
        print("Please ensure MODEL_PATH is correct and configuration matches training.")
      
        INPUT_CHANNELS = 3 
        CLASS_NAMES = ['A', 'B', 'C', 'L', 'O', 'V', 'Y']
        print("Using fallback configuration.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists at the specified path and required libraries (like h5py) are installed.")
    model = None
    CLASS_NAMES = []

cap = None 
is_realtime_running = False
last_prediction = "None"
is_operational_time = False

def preprocess_image(img, target_height, target_width, channels):
  
    
    img_resized = cv2.resize(img, (target_width, target_height))


    if channels == 1: 
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        elif len(img_resized.shape) == 2:
             img_gray = img_resized 
        else: 
            print(f"Warning: Unexpected image shape {img_resized.shape}, attempting grayscale conversion.")
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) > 2 else img_resized

        img_processed = img_gray.astype('float32') / 255.0
        img_processed = np.expand_dims(img_processed, axis=-1) 
    else: 
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 1:
             img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        else:
          
            img_rgb = img_resized

        img_processed = img_rgb.astype('float32') / 255.0

 
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch

def predict_sign(image_batch):
    """Predicts the sign from a preprocessed image batch."""
    if model is None:
        return "Model not loaded", 0.0

    try:
        prediction = model.predict(image_batch)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100 

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
    root.after(60000, check_operational_time) 

def update_status_label():
   
    if is_operational_time:
        status_label.config(text="Status: Operational (6 PM - 10 PM)", fg="green")
      
        upload_button.config(state=tk.NORMAL if model else tk.DISABLED)
        realtime_button.config(state=tk.NORMAL if model else tk.DISABLED)
    else:
        status_label.config(text="Status: Inactive (Operational 6 PM - 10 PM)", fg="red")
        upload_button.config(state=tk.DISABLED)
        if is_realtime_running:
            toggle_realtime_feed() 
        realtime_button.config(state=tk.DISABLED)



def upload_image():
   
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
       
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            messagebox.showerror("Error", f"Could not read image file: {filepath}")
            return

    
        image_batch = preprocess_image(img_cv, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)

       
        def run_prediction():
            global last_prediction
            predicted_class, confidence = predict_sign(image_batch)
            last_prediction = f"{predicted_class} ({confidence:.1f}%)"
            prediction_label.config(text=f"Prediction: {last_prediction}")

          
            img_display = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) 
            img_pil = Image.fromarray(img_display)
            img_pil.thumbnail((video_label.winfo_width(), video_label.winfo_height() - 50)) 
            img_tk = ImageTk.PhotoImage(img_pil)

         
            root.after(0, update_uploaded_image_display, img_tk)

        def update_uploaded_image_display(img_tk_arg):
            
            video_label.imgtk = img_tk_arg
            video_label.config(image=img_tk_arg, text="") 


      
        prediction_thread = threading.Thread(target=run_prediction, daemon=True)
        prediction_thread.start()
        prediction_label.config(text="Prediction: Processing...")


    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")
        print(f"Error details: {e}") 
        prediction_label.config(text="Prediction: Error")
        video_label.config(image='', text="Error loading image") 


def update_realtime_feed():
  
    global last_prediction
    if not is_realtime_running or not is_operational_time or cap is None or not cap.isOpened():
        return 

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera.")
       
        root.after(50, update_realtime_feed)
        return

   
    frame = cv2.flip(frame, 1)


    h, w, _ = frame.shape
    roi_x, roi_y, roi_w, roi_h = w // 2 - 100, h // 2 - 100, 200, 200
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

   
    image_batch = preprocess_image(roi, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)

   
    predicted_class, confidence = predict_sign(image_batch)
    last_prediction = f"{predicted_class} ({confidence:.1f}%)"
    prediction_label.config(text=f"Prediction: {last_prediction}")

  
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

   
    video_label.imgtk = imgtk 
    video_label.config(image=imgtk, text="")

    
    root.after(20, update_realtime_feed) 

def toggle_realtime_feed():
  
    global is_realtime_running, cap

    if not is_operational_time and not is_realtime_running:
         messagebox.showwarning("Inactive", "Sign detection is only operational between 6 PM and 10 PM.")
         return
    if model is None and not is_realtime_running:
         messagebox.showerror("Error", "Model is not loaded. Cannot start real-time feed.")
         return

    if is_realtime_running:
      
        is_realtime_running = False
        realtime_button.config(text="Start Real-time")
        prediction_label.config(text="Prediction: Off")
        video_label.config(image='', text="Camera Feed Stopped") 
        if cap is not None:
            cap.release()
            print("Camera released.")
            cap = None
    else:
        
        try:
            cap = cv2.VideoCapture(0) 
            if not cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open webcam.")
                cap = None
                return

            print("Camera opened successfully.")
            is_realtime_running = True
            realtime_button.config(text="Stop Real-time")
            prediction_label.config(text="Prediction: Starting...")
            video_label.config(image='', text="Initializing Camera...") 
           
            root.after(100, update_realtime_feed)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            is_realtime_running = False
            realtime_button.config(text="Start Real-time")
            if cap:
                cap.release()
                cap = None


def on_closing():
   
    global is_realtime_running, cap
    print("Closing application...")
    is_realtime_running = False 
    time.sleep(0.1) 
    if cap is not None:
        cap.release()
        print("Camera released on exit.")
    root.destroy()


root = tk.Tk()
root.title("Sign Language Detector")
root.geometry("800x650") 
root.protocol("WM_DELETE_WINDOW", on_closing) 



status_label = Label(root, text="Status: Initializing...", font=("Arial", 12), fg="orange")
status_label.pack(pady=10)


video_label = Label(root, text="Upload an image or start real-time feed", bg="lightgrey")
video_label.pack(pady=10, padx=10, fill="both", expand=True)



prediction_label = Label(root, text="Prediction: None", font=("Arial", 16, "bold"))
prediction_label.pack(pady=10)


button_frame = Frame(root)
button_frame.pack(pady=15)

upload_button = Button(button_frame, text="Upload Image", command=upload_image, width=15, height=2, state=tk.DISABLED)
upload_button.pack(side=tk.LEFT, padx=20)

realtime_button = Button(button_frame, text="Start Real-time", command=toggle_realtime_feed, width=15, height=2, state=tk.DISABLED)
realtime_button.pack(side=tk.LEFT, padx=20)


if model is None:
     status_label.config(text="Status: Model Load FAILED. Cannot operate.", fg="red")
     messagebox.showerror("Startup Error", "Failed to load the sign language model. Please check the console for errors and ensure the model file exists.")
else:
     check_operational_time() 



root.mainloop()
