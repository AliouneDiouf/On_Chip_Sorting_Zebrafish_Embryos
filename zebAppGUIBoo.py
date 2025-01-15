# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:51:16 2024

@author: Ali
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
import serial
import time
import torch
# Initialize YOLOv8 model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(r'C:\Users\Milad\Desktop\ALIOUNE\SORTING\model\best_optimized.pt')
model.to(device=device)



# Global variables to store the path of the loaded image and counts
loaded_image_path = None
STAGE1_count = 0
DEAD_count = 0
ADVANCED_count = 0
camera_running = False  # Flag to control camera loop

# Dictionary to keep image references
icon_images = {}
def send_signal_to_arduino(signal):
    """Send a signal to the Arduino."""
    if arduino and arduino.is_open:
        try:
            # Envoyer le signal à l'Arduino
            arduino.write(signal.to_bytes(1, byteorder='big'))
            time.sleep(3)
            print(f"Signal envoyé à l'Arduino : {signal}")
            
          
        except serial.SerialException as e:
            print(f"Erreur lors de l'envoi du signal à l'Arduino : {e}")
    else:
        print("Aucun Arduino connecté.")

def reset():
    """Reset the application to its initial state."""
    global loaded_image_path, STAGE1_count, DEAD_count, ADVANCED_count
    loaded_image_path = None
    STAGE1_count = 0
    DEAD_count = 0
    ADVANCED_count = 0
    for widget in main_frame.winfo_children():
        widget.pack_forget()
    welcome_frame.pack(pady=50)

def start_live_sorting():
    """Start live sorting from the main menu."""
    global camera_running
    camera_running = True
    display_camera()

def stop_live_sorting():
    """Stop live sorting and display statistics."""
    global camera_running
    camera_running = False
    show_statistics()

def test_model():
    """Start testing the model from the main menu."""
    notebook.select(test_model_frame)

def test_pumps():
    """Start testing the pumps from the main menu."""
    notebook.select(test_pumps_frame)

loaded_image_label = None

def load_image(test_model_frame):
    """Load an image for testing the model."""
    global loaded_image_path, original_photo, canvas, loaded_image_label
    
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image File", 
                                          filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"), 
                                                     ("All Files", "*.*")))
    if filename:
        loaded_image_path = filename
        image = Image.open(filename)
        image.thumbnail((400, 400))
        original_photo = ImageTk.PhotoImage(image)
        
        # Display the uploaded image
        if loaded_image_label:
            loaded_image_label.destroy()  # Destroy the previous loaded image label if it exists

        loaded_image_label = ttk.Label(test_model_frame, image=original_photo)
        loaded_image_label.image = original_photo  # Keep a reference to the image
        loaded_image_label.pack(side="top", padx=10, pady=10)

def predict_and_display(test_model_frame):
    """Predict using YOLO and display the result image."""
    global loaded_image_path
    if loaded_image_path:
        predict_image(loaded_image_path)

def predict_image(image_path):
    """Predict the image using the model."""
    global result_image_label
    if not image_path:
        messagebox.showerror("Error", "No image loaded.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    results = model(image)
    for result in results:
        result_image = result.plot()  # Using the plot method to get the image with detections
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)
        result_image.thumbnail((400, 400))
        result_photo = ImageTk.PhotoImage(result_image)
        result_image_label.config(image=result_photo)
        result_image_label.image = result_photo  # Keep a reference to the image
        
        # Destroy the widget displaying the loaded image
        loaded_image_label.pack_forget()  # Assuming load_image_frame is the frame containing the loaded image

def display_camera():
    """Display live camera feed and perform object detection."""
    global STAGE1_count, DEAD_count, ADVANCED_count, camera_running
    
    # Try different camera IDs
    for cam_id in [0, 1, -1, 2, 3]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            break
    else:
        print("Cannot open camera")
        return

    def update_frame():
        global STAGE1_count, DEAD_count,ADVANCED_count
        if not camera_running:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.release()
            frame_label.config(image='')
            stats_label.config(text='')
            stop_live_sorting_button.grid_forget()


            return
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return
        # Prédictions sur le CPU
        with torch.no_grad():
          results = model.predict(source=frame, stream=True,project=r'C:\Users\Milad\Desktop\ALIOUNE\SORTING\runs')
        #results = model.predict(source=frame,device='1',stream=True)
        frame_STAGE1_count = 0
        frame_DEAD_count = 0
        frame_ADVANCED_count = 0
        for result in results:
            result_frame = result.plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            result_frame = Image.fromarray(result_frame)
            result_frame.thumbnail((600, 600))  # Adjusting thumbnail size
            result_frame = ImageTk.PhotoImage(result_frame)

            frame_label.config(image=result_frame)
            frame_label.image = result_frame  # Keep a reference to the image

            # Update statistics
            stats = result.names
            cls_counts = {}
            for cls in result.boxes.cls.tolist():
                cls_name = stats[int(cls)]
                if cls_name == "STAGE1":
                    frame_STAGE1_count += 1
                    send_signal_to_arduino(2)  # Envoyer 1 pour démarrer les pompes


                elif cls_name == "DEAD":
                    frame_DEAD_count += 1
                    send_signal_to_arduino(3)  # Envoyer 1 pour démarrer les pompes

                 
                elif cls_name == "ADVANCED":
                    frame_ADVANCED_count += 1
                    send_signal_to_arduino(4)
                   

                   
                if cls_name in cls_counts:
                    cls_counts[cls_name] += 1
                else:
                    cls_counts[cls_name] = 1
            STAGE1_count += frame_STAGE1_count
            DEAD_count += frame_DEAD_count
            ADVANCED_count += frame_ADVANCED_count
            stats_text = f"STAGE1: {STAGE1_count}\nDEAD: {DEAD_count}\nADVANCED: {ADVANCED_count}"
            stats_label.config(text=stats_text)

        # Call update_frame function recursively
        root.after(10, update_frame)
        # Call update_frame function for the first time
    update_frame()

def show_statistics():
    """Show statistics and a pie chart of object detections."""
    # Create a new window for statistics
    stats_window = tk.Toplevel(root)
    stats_window.title("Statistics")
    stats_window.geometry("500x500")

    # Display counts
    stats_text =f"STAGE1: {STAGE1_count}\nDEAD: {DEAD_count}\nADVANCED: {ADVANCED_count}"
    stats_label = ttk.Label(stats_window, text=stats_text, font=("Arial", 16))
    stats_label.pack(pady=20)

    # Create pie chart
    figure, ax = plt.subplots()
    labels = ['STAGE1', 'DEAD', 'ADVANCED']
    sizes = [STAGE1_count, DEAD_count, ADVANCED_count]
    colors = ['red', 'green','blue']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Embed the pie chart in the tkinter window
    canvas = FigureCanvasTkAgg(figure, stats_window)
    canvas.get_tk_widget().pack()

def tutorials():
    """Show tutorials."""
    # Create a new window for tutorials
    tutorial_window = tk.Toplevel(root)
    tutorial_window.title("Tutorials")
    tutorial_window.geometry("500x500")

    # Add tutorial text
    tutorial_text = "Welcome to the Tutorials!\n\n\nTo get started, you can:\n\n1. Test the model with a single image.\n2. Start live sorting from your camera feed.\n3. View tutorials for using the application features.\n\nEnjoy sorting!"
    tutorial_label = ttk.Label(tutorial_window, text=tutorial_text, font=("Arial", 30), foreground='black')
    tutorial_label.pack(pady=20, anchor="center")


def refresh():
    """Reset all pages of the app."""
    reset()

def exit_app():
    """Close the application."""
    if arduino and arduino.is_open:
        arduino.close()
    root.destroy()
    
def init_serial():
    try:
        return serial.Serial('COM4', 9600, timeout=1)  # Remplacez 'COM3' par le port série approprié
    except serial.SerialException as e:
        print(f"Erreur de connexion au port série: {e}")
        return None

arduino = init_serial()


def check_pumps():
    """Send a command to Arduino to check the pumps."""
    if arduino and arduino.is_open:
        try:
            arduino.write(b'\x06')  # Send the command 6 to check pumps
            print("Check Pumps command sent to Arduino.")
        except serial.SerialException as e:
            print(f"Error sending check pumps command to Arduino: {e}")
    else:
        print("No Arduino connected.")
        
def stop_test_button():
   pass      




## Initialize the main application window
root = tb.Window(themename="minty")
root.title("ZebApp")
root.geometry("1500x900")

# Load icon8
icon_paths = {
    'welcome_icon': "icons/welcome.png",
    'live_sorting_icon': "icons/live_sorting.png",
    'tutorials_icon': "icons/tutorials.png",
    'test_pumps_icon': "icons/test_pumps.png",
    'test_model_icon': "icons/test_model.png"
}
root.title("ZebApp")
root.geometry("1500x900")

# Load icon8
icon_paths = {
    'welcome_icon': "icons/welcome.png",
    'live_sorting_icon': "icons/live_sorting.png",
    'tutorials_icon': "icons/tutorials.png",
    'test_pumps_icon': "icons/test_pumps.png",
    'test_model_icon': "icons/test_model.png"
}

for key, path in icon_paths.items():
    try:
        icon_images[key] = ImageTk.PhotoImage(Image.open(path))
    except Exception as e:
        print(f"Error loading image {path}: {e}")

# Create a Notebook widget for the tabs
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Welcome Frame
welcome_frame = ttk.Frame(notebook, padding=10)
welcome_label = ttk.Label(welcome_frame, text="Hello !", font=("Roboto", 40,'bold'), foreground='black')
welcome_label.pack(pady=100, anchor="center")
# Project Description
description_text = ("ZebApp is your solution for object detection and sorting of zebrafish based on Deep Learning! This application offers a user-friendly "
                    "\ninterface and powerful features to meet your computer vision needs.\n")
             







description_label = ttk.Label(welcome_frame, text=description_text, font=("Roboto", 14,'bold'), foreground='black')


description_label.pack(pady=20, anchor="center")

if 'welcome_icon' in icon_images:
    icon_label = ttk.Label(welcome_frame, image=icon_images['welcome_icon'])

    icon_label.pack()
    


notebook.add(welcome_frame, text="Welcome")

# Live Sorting Frame
live_sorting_frame = ttk.Frame(notebook, padding=10)

# Create a frame for buttons and pack it to the left
button_frame = ttk.Frame(live_sorting_frame)
button_frame.pack(side='left', fill='y', padx=5, pady=5)  # Réduisez les padx et pady

live_sorting_label = ttk.Label(button_frame, text="Live Sorting : After verification of the System, Click on Start to begin the process !", font=("Roboto", 14, "bold"), foreground='black')
live_sorting_label.pack(pady=5, anchor="center")  # Réduisez les pady

if 'live_sorting_icon' in icon_images:
    icon_label = ttk.Label(button_frame, image=icon_images['live_sorting_icon'])
    icon_label.pack()

# Start/Stop Live Sorting buttons
start_live_sorting_button = ttk.Button(button_frame, text="Start", command=start_live_sorting, style="primary.TButton")
start_live_sorting_button.pack(pady=5)  # Réduisez les pady

stop_live_sorting_button = ttk.Button(button_frame, text="Stop Detection", command=stop_live_sorting, style="warning.TButton")
stop_live_sorting_button.pack(pady=5)  # Réduisez les pady

# Create a frame for displaying the camera feed and pack it to the right
camera_frame = ttk.Frame(live_sorting_frame)
camera_frame.pack(side='right', fill='both', expand=True, padx=5, pady=100)  # Réduisez les padx et pady

# Add a frame to create a border around the video frame
video_border_frame = ttk.Frame(camera_frame, padding=5, style="primary.TFrame")
video_border_frame.pack(pady=5)  # Réduisez les pady

frame_label = ttk.Label(video_border_frame)
frame_label.pack(pady=5)  # Réduisez les pady

stats_label = ttk.Label(camera_frame, text="", font=("Arial", 16), foreground='black')
stats_label.pack(pady=5)  # Réduisez les pady

notebook.add(live_sorting_frame, text="Live Sorting")



# Tutorials Frame
tutorials_frame = ttk.Frame(notebook, padding=10)
tutorials_label = ttk.Label(tutorials_frame, text="Welcome to the Tutorials!\n\n\nTo get started, you can:\n\n1. Test the model with a single image.\n2. Start live sorting from your camera feed.\n3. View tutorials for using the application features.\n\nEnjoy sorting!", font=("Roboto", 30, 'bold'), foreground='black')
tutorials_label.pack(pady=10, anchor="center")
if 'tutorials_icon' in icon_images:
    icon_label = ttk.Label(tutorials_frame, image=icon_images['tutorials_icon'])
    icon_label.pack()
notebook.add(tutorials_frame, text="Tutorials")

# Test the Pumps Frame
test_pumps_frame = ttk.Frame(notebook, padding=10)
test_pumps_label = ttk.Label(test_pumps_frame, text="Test the Pumps", font=("Roboto", 30, "bold"), foreground='black')
test_pumps_label.pack(pady=10, anchor="center")
if 'test_pumps_icon' in icon_images:
    icon_label = ttk.Label(test_pumps_frame, image=icon_images['test_pumps_icon'])
    icon_label.pack()

# Check Pumps Button
check_pumps_button = ttk.Button(test_pumps_frame, text="Check Pumps", command=check_pumps, style="primary.TButton")
check_pumps_button.pack(pady=10)

# Explanation Label
test_pumps_explain_label = ttk.Label(test_pumps_frame, text="Click on 'Check Pumps' to start the pump testing process.", font=("Arial", 14), foreground='black')
test_pumps_explain_label.pack(pady=10)

# Calibration Label
calibration_label = ttk.Label(test_pumps_frame, text="Calibration", font=("Roboto", 16, "bold"), foreground='black')
calibration_label.pack(pady=10)

# Test Model Frame
test_model_frame = ttk.Frame(notebook, padding=10)
test_model_explain_label = ttk.Label(test_model_frame, text="Click on 'Load image' to load an image and then Click on Test to detect eggs.", font=("Roboto", 30,'bold'), foreground='black')
test_model_explain_label.pack(pady=10)
if 'test_model_icon' in icon_images:
    icon_label = ttk.Label(test_model_frame, image=icon_images['test_model_icon'])
    icon_label.pack()

# Result Image Frame
result_image_frame = ttk.Frame(test_model_frame)
result_image_frame.pack(pady=10)

# Result Image Label
result_image_label = ttk.Label(result_image_frame)
result_image_label.pack(pady=10)

# Load Image Button
load_image_button = ttk.Button(test_model_frame, text="Load Image", command=lambda: load_image(test_model_frame), style="primary.TButton")
load_image_button.pack(side="left", padx=5, pady=10)

# Test Model Button
test_model_button = ttk.Button(test_model_frame, text="Test", command=lambda: predict_and_display(test_model_frame), style="info.TButton")
test_model_button.pack(side="left", padx=5, pady=10)

notebook.add(test_model_frame, text="Test Model")


# Test the Pumps Frame
test_pumps_frame = ttk.Frame(notebook, padding=10)
test_pumps_label = ttk.Label(test_pumps_frame, text="Test the Pumps", font=("Roboto", 30, "bold"), foreground='black')
test_pumps_label.pack(pady=10, anchor="center")

if 'test_pumps_icon' in icon_images:
    icon_label = ttk.Label(test_pumps_frame, image=icon_images['test_pumps_icon'])
    icon_label.pack()

# Check Pumps Button
check_pumps_button = ttk.Button(test_pumps_frame, text="Check Pumps", command=check_pumps, style="primary.TButton")
check_pumps_button.pack(pady=10)

# Explanation Label
test_pumps_explain_label = ttk.Label(test_pumps_frame, text="Click the buttons below to control pump speeds.", font=("Arial", 14), foreground='black')
test_pumps_explain_label.pack(pady=10)


# Ajout des contrôles pour chaque pompe
for pump_id in range(1,5):  # Pour les pompes P0, P1, P2, P3
    # Création d'un sous-frame pour chaque pompe
    pump_frame = ttk.Frame(test_pumps_frame)
    pump_frame.pack(pady=5, fill="x")

    # Ajout de l'étiquette pour la pompe
    ttk.Label(pump_frame, text=f"Pump P{pump_id}", font=("Roboto", 16, "bold"), foreground='black').grid(row=0, column=0, padx=5)

    # Ajout des boutons pour la pompe
    ttk.Button(pump_frame, text="Low Speed", command=lambda pid=pump_id: send_signal_to_arduino(pid*10), style="success.TButton").grid(row=0, column=1, padx=5)
    ttk.Button(pump_frame, text="Medium Speed", command=lambda pid=pump_id: send_signal_to_arduino((pid*10) + 1), style="success.TButton").grid(row=0, column=2, padx=5)
    ttk.Button(pump_frame, text="High Speed", command=lambda pid=pump_id: send_signal_to_arduino(pid*10 + 2), style="success.TButton").grid(row=0, column=3, padx=5)






# Button to Set these Values
stop_test_button = ttk.Button(test_pumps_frame, text="Stop_Test", command=stop_test_button)
stop_test_button.pack(pady=10)
notebook.add(test_pumps_frame, text="Test the Pumps")

# Sidebar
sidebar_frame = ttk.Frame(root, padding=5, height=150, style="primary.TFrame")
sidebar_frame.pack(side='top', fill='x')

refresh_button = ttk.Button(sidebar_frame, text="Refresh", command=refresh, style="dark.TButton")
refresh_button.pack(side='left', padx=10)

exit_button = ttk.Button(sidebar_frame, text="Exit", command=exit_app, style="danger.TButton")
exit_button.pack(side='right', padx=10)

root.mainloop()
