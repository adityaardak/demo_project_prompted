import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Face Identifier (OpenCV + Streamlit)",
    page_icon="🙂",
    layout="centered",
)

st.title("🙂 Human Face Identifier")
st.write(
    "Upload an image, tweak detection parameters, and the app will draw a box with the label "
    "**'Human face identified'** on each detected face."
)

# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource
def load_face_cascade():
    # Uses OpenCV's built-in Haar Cascade path
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade. Check your OpenCV installation.")
    return cascade


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_and_annotate_faces(
    bgr_img: np.ndarray,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    draw_thickness: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Detect faces and draw rectangle + label."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    cascade = load_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    annotated = bgr_img.copy()

    for (x, y, w, h) in faces:
        # Rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), draw_thickness)

        # Label background box + text
        label = "Human face identified"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        # Place label above the rectangle if possible, else inside
        ty = y - 10
        if ty - th - baseline < 0:
            ty = y + th + baseline + 10

        # Filled rectangle behind text
        cv2.rectangle(
            annotated,
            (x, ty - th - baseline),
            (x + tw + 10, ty + baseline),
            (0, 255, 0),
            -1,
        )

        # Text
        cv2.putText(
            annotated,
            label,
            (x + 5, ty),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated, faces.tolist() if len(faces) else []


# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Detection Controls")

scale_factor = st.sidebar.slider(
    "scaleFactor (image pyramid step)",
    min_value=1.01,
    max_value=1.50,
    value=1.10,
    step=0.01,
    help="Lower values detect more faces but may increase false positives and slow down detection.",
)

min_neighbors = st.sidebar.slider(
    "minNeighbors (strictness)",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Higher values reduce false positives but may miss faces.",
)

min_size = st.sidebar.slider(
    "minSize (minimum face size in pixels)",
    min_value=20,
    max_value=200,
    value=40,
    step=5,
    help="Increase if you want to ignore small faces / far-away faces.",
)

draw_thickness = st.sidebar.slider(
    "Box thickness",
    min_value=1,
    max_value=8,
    value=3,
    step=1,
)

st.sidebar.divider()
st.sidebar.caption("Tip: If faces are not detected, try lowering minSize and minNeighbors, or lowering scaleFactor slightly.")


# ----------------------------
# Main UI
# ----------------------------
uploaded = st.file_uploader(
    "Upload an image (JPG/PNG/WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

col1, col2 = st.columns(2, gap="large")

if uploaded:
    # Read with PIL
    pil_img = Image.open(uploaded)

    with col1:
        st.subheader("Preview")
        st.image(pil_img, use_container_width=True)

    # Detect
    bgr = pil_to_bgr(pil_img)
    annotated_bgr, faces = detect_and_annotate_faces(
        bgr_img=bgr,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
        draw_thickness=draw_thickness,
    )
    annotated_pil = bgr_to_pil(annotated_bgr)

    with col2:
        st.subheader("Detected Faces")
        st.image(annotated_pil, use_container_width=True)

    st.markdown("---")
    if faces:
        st.success(f"✅ Faces detected: **{len(faces)}**")
        with st.expander("Show face coordinates (x, y, w, h)"):
            st.write(faces)
    else:
        st.warning("No faces detected. Try adjusting parameters in the sidebar (especially minSize / minNeighbors).")

    # Download annotated image
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download annotated image (PNG)",
        data=buf.getvalue(),
        file_name="annotated_faces.png",
        mime="image/png",
    )

else:
    st.info("Upload an image to start face detection.")
