import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import numpy as np

# Load YOLO model once
model = YOLO("yolo11s.pt")

st.title("🔍 YOLO Object Detection App")
st.write("Upload one or more images to run YOLO detection.")

# Allow multiple uploads,
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "bmp", "gif"],
    accept_multiple_files=True
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"### Image {idx}: {uploaded_file.name}")

        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save to temp file for YOLO
        ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Run YOLO detection (don't save to disk)
        results = model.predict(source=temp_path, save=False)

        # Get annotated image from YOLO results
        result_img = results[0].plot()  # numpy array (BGR)
        result_img = Image.fromarray(result_img[..., ::-1])  # convert to RGB

        # Show annotated result
        st.image(result_img, caption="Detection Result", use_column_width=True)

        # ---- Detection summary ----
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
        st.markdown("---")
