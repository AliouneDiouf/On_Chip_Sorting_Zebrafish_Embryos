# OnChip Sorting Of Zebrafish Embryos

This is a Python-based application designed for object detection and control tasks. It integrates YOLO model-based object detection, real-time video processing, and Arduino communication for controlling devices such as pumps.

---

## **Features**

- **Welcome Screen:** Provides an overview of the application.
- **Live Sorting:** Real-time object detection via live video feed.
- **Test Model:** Upload images to test object detection capabilities.
- **Tutorials:** Step-by-step guidance on using the app.
- **Pump Testing:** Control and test pumps connected via Arduino.

---

## **Installation**

### Prerequisites
Ensure the following software is installed:

- Python 3.8+
- pip (Python package manager)

### Install Dependencies
Run the following command to install required Python libraries:

```bash
pip install ultralytics matplotlib ttkbootstrap pillow opencv-python torch pyserial
```

---

## **Setup**
### 1. Intersection_point
First use the Intersection_point code to detect the Intersection_point of the microfluidic and ensure not to move the chip for all the process.

### 1. YOLO Model
Ensure you have a YOLO model file ready for object detection. Update the model path in the script:

```python
model = YOLO(r'C:\Users\Milad\Desktop\ALIOUNE\SORTING\model\best_optimized.pt')
```

### 2. Arduino Connection
If using an Arduino for pump control:

- Update the serial port in the script to match your Arduino's connection:

```python
return serial.Serial('COM4', 9600, timeout=1)
```

Replace `COM4` with the correct port.

### 3. Icons
Place the following icons in a folder named `icons` at the same directory level as the script:

- `welcome.png`
- `live_sorting.png`
- `tutorials.png`
- `test_pumps.png`
- `test_model.png`

---

## **Usage**

### Run the Application
Save the script as `zebapp.py` and start the app using:

```bash
python zebapp.py
```

### Explore Features
1. **Welcome Tab:** Learn about the app's features.
2. **Live Sorting Tab:** Activate live video feed for real-time object detection.
3. **Test Model Tab:** Upload an image to test object detection.
4. **Pump Testing Tab:** Manually test and control pumps via Arduino.
5. **Tutorials Tab:** Access helpful guides for using the app.

---

## **Example Workflow**

1. Launch the application:

```bash
python zebapp.py
```

2. Navigate to the **Test Model** tab to load an image and analyze object detection results.
3. Switch to the **Live Sorting** tab to start real-time object detection through a camera.
4. Test pumps under the **Test the Pumps** tab to verify Arduino-based control.

---

## **Contributing**

Contributions are welcome! Feel free to fork this repository and submit pull requests.

---


