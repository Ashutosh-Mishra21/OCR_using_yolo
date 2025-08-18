import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

st.title("🔍 YOLO Object Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        # Save uploaded file to a temp file with same extension
        ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Run YOLO detection (save=False so we rely on results, not disk)
        results = model.predict(
            source=temp_path,
            save=False  # <--- don't save, we'll display directly
        )

        # Get annotated image from YOLO results
        result_img = results[0].plot()  # numpy array (BGR)
        result_img = Image.fromarray(result_img[..., ::-1])  # convert BGR → RGB

        # Show detection result
        st.image(result_img, caption="Detection Result", use_column_width=True)

        # ---- Show detection summary ----
        detections = results[0].boxes
        class_names = results[0].names  # dict {id: "class"}

        summary = []
        class_count = {}

        for box in detections:
            cls_id = int(box.cls[0].item())      # class index
            conf = float(box.conf[0].item())     # confidence
            cls_name = class_names[cls_id]

            summary.append(f"{cls_name} ({conf:.2f})")
            class_count[cls_name] = class_count.get(cls_name, 0) + 1

        if summary:
            st.subheader("Detections:")
            for item in summary:
                st.write("•", item)

            st.subheader("Count per class:")
            for cls, count in class_count.items():
                st.write(f"{cls}: {count}")
        else:
            st.warning("No objects detected!")

        st.success("✅ Detection complete!")
