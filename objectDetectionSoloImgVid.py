import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from pathlib import Path
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

st.title("🔍 YOLO Object Detection App")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "gif", "mp4"])

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        # ---- IMAGE HANDLING ----
        image = Image.open(temp_path).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model.predict(source=temp_path, save=False)

            # Annotated image
            result_img = results[0].plot()
            result_img = Image.fromarray(result_img[..., ::-1])
            st.image(result_img, caption="Detection Result", use_column_width=True)

            # Show detections
            detections = results[0].boxes
            class_names = results[0].names
            summary, class_count = [], {}

            for box in detections:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
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


    elif ext == ".mp4":

        # ---- VIDEO HANDLING ----

        st.video(temp_path, format="video/mp4")

        if st.button("Run Detection on Video"):
            results = model.predict(source=temp_path, save=True)

            # Convert save_dir to Path before joining

            output_path = Path(results[0].save_dir) / os.path.basename(temp_path)

            st.video(str(output_path))

            st.success("✅ Video detection complete!")