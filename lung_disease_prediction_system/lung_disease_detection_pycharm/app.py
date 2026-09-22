# app.py
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import io

# Try import grad-cam tools (optional)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_CAM = True
except Exception:
    HAS_CAM = False

# -----------------------
# Config / Utils
# -----------------------
st.set_page_config(page_title="Lung Disease Detection", layout="centered")
st.title("🩺 Lung Disease Detection (Normal / Pneumonia / TB)")
st.markdown("Upload a chest X-ray image. Model predicts class and shows confidence. Optionally view Grad-CAM heatmap.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/infectious_model.pth"  # relative to project root
CLASSES = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# Model loading (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    # Build same architecture and load weights
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(CLASSES))
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at `{MODEL_PATH}`. Make sure the file exists.\n\nError: {e}")
    st.stop()

# -----------------------
# Prediction function
# -----------------------
def predict_image(image: Image.Image):
    """Return predicted label and confidence (0-100)."""
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASSES[pred.item()], float(conf.item() * 100), probs.cpu().numpy().squeeze()

# -----------------------
# Grad-CAM function
# -----------------------
def generate_gradcam(image: Image.Image):
    """Return visualization overlay (H x W x 3 uint8) or None if CAM not available."""
    if not HAS_CAM:
        return None
    try:
        # Target the last conv layer of ResNet18
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        img = image.convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        # get predicted class
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
        pred_class = int(torch.argmax(outputs, dim=1).item())
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_img = np.array(img.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return visualization  # uint8 HxWx3
    except Exception as ex:
        # If Grad-CAM fails, return None (we'll show a message in UI)
        st.warning(f"Grad-CAM generation failed: {ex}")
        return None

# -----------------------
# Streamlit UI
# -----------------------
uploaded = st.file_uploader("📤 Upload Chest X-ray (jpg / jpeg / png)", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload a chest X-ray image to get prediction.")
    st.write("You can use sample images in `sample_images/` for testing.")
    st.stop()

# Read image
try:
    image = Image.open(io.BytesIO(uploaded.read()))
except Exception as e:
    st.error(f"Unable to read image: {e}")
    st.stop()

# Display uploaded image
st.image(image, caption="Uploaded X-ray", use_column_width=True)

# Predict
with st.spinner("Running model inference..."):
    label, confidence, prob_array = predict_image(image)

st.markdown(f"### ✅ Prediction: **{label}**")
st.markdown(f"### 🔹 Confidence: **{confidence:.2f}%**")

# Show top-3 probabilities neatly
st.write("#### Probabilities (per class):")
for i, cls in enumerate(CLASSES):
    st.write(f"- **{cls}**: {prob_array[i]*100:.2f}%")

# Grad-CAM toggle
if HAS_CAM:
    show_cam = st.checkbox("Show Grad-CAM heatmap (explainability)")
    if show_cam:
        with st.spinner("Generating Grad-CAM..."):
            cam_vis = generate_gradcam(image)
        if cam_vis is not None:
            st.image(cam_vis, caption="Grad-CAM heatmap overlay", use_column_width=True)
        else:
            st.warning("Grad-CAM not available for this image.")
else:
    st.info("Grad-CAM not installed. Install `pytorch-grad-cam` to enable heatmaps.")

# Small footer
st.markdown("---")
st.caption("Model: ResNet18 (Transfer Learning). Trained to classify Normal / Pneumonia / Tuberculosis from chest X-rays.")
