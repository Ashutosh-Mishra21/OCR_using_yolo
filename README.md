# Performing Object Detection Using YOLO11 (YOLOv11n / YOLOv11s)

This project demonstrates **object detection** using the [Ultralytics YOLO](https://docs.ultralytics.com) models (`yolo11n.pt` and `yolo11s.pt`).
It includes:

* A **Jupyter Notebook/Google Colab** for experimenting with YOLO detection.
* **Streamlit apps** for detecting objects in images and videos.
* A **Webcam script** for real-time detection.

---

## 🚀 Features

* Detect objects in:

  * Single Image
  * Multiple Images
  * Image or Video
  * Live Webcam Stream
* Real-time bounding boxes, labels, and confidence scores.
* Class-wise detection summary.
* Streamlit apps for interactive browser usage.

---

## 📂 Project Structure

```
├── performingObjectDetectionUsingYOLO11n.ipynb   # Notebook version
├── objectDetectionSoloImg.py                     # Streamlit app: single image
├── objectDetectionMultipleImg.py                 # Streamlit app: multiple images
├── objectDetectionSoloImgVid.py                  # Streamlit app: image or video
├── useCamToFind.py                               # Live webcam detection
├── requirements.txt                              # Dependencies
├── README.md                                     # Project documentation
└── runs/                                         # YOLO outputs (auto-created)
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/arupa444/OCR-Using-YOLO.git
   cd OCR-Using-YOLO
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### 1. Run the Jupyter Notebook/Google Colab

```bash
jupyter notebook performingObjectDetectionUsingYOLO11n.ipynb
```

### 2. Detect Objects on Images

Inside the notebook:

```python
!yolo predict model=yolo11n.pt source='image.jpg'
```

### 3. Detect Objects on Video

```python
!yolo predict model=yolo11n.pt source='video.mp4'
```

### 4. Live Webcam Detection

```python
!yolo predict model=yolo11n.pt source=0
```

---

### 2. Run Streamlit Apps

#### 🔹 Single Image Detection

```bash
streamlit run objectDetectionSoloImg.py
```

#### 🔹 Multiple Image Detection

```bash
streamlit run objectDetectionMultipleImg.py
```

#### 🔹 Image or Video Detection

```bash
streamlit run objectDetectionSoloImgVid.py
```

### 3. Live Webcam Detection

```bash
python useCamToFind.py
```

* Uses your default webcam (`source=1`).
* Opens a live window with bounding boxes.
* Saves detection results in `runs/detect/`.

---

## 📦 Output

* Annotated images/videos are shown directly in the apps.
* YOLO saves all processed results inside:

  ```
  runs/detect/
  ```

  Each execution creates a new timestamped folder.

---

## 📌 Requirements

* Python 3.8+
* Jupyter Notebook/Google Colab
* Ultralytics YOLO
* Streamlit
* Pillow
* NumPy
* OpenCV
* Torch + TorchVision

Install directly with:

```bash
pip install ultralytics streamlit pillow numpy opencv-python torch torchvision
```

---

## 🛠 Example (Notebook)

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")
results.show()
```

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify.
