import io
import streamlit as st
import torch
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="AI Image Detector", layout="wide")

@st.cache_resource
def load():
    p = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
    m = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
    return p, m.eval()

processor, model = load()

def ela_image(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = ImageChops.difference(img, Image.open(buf))
    mx = max(e[1] for e in diff.getextrema()) or 1
    return ImageEnhance.Brightness(diff).enhance(255.0 / mx)

def get_exif(raw):
    return {str(TAGS.get(k, k)): str(v)[:80] for k, v in raw.getexif().items()}

st.title("🔍 AI-Generated Image Detector")
st.warning("Educational project. Results are probabilistic and can be wrong. "
           "Images are processed in memory and never stored.")

tab1, tab2, tab3 = st.tabs(["Detect", "About", "Spot AI Images Yourself"])

with tab1:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        raw = Image.open(file)
        exif = get_exif(raw)
        img = raw.convert("RGB")

        with torch.no_grad():
            logits = model(**processor(images=img, return_tensors="pt")).logits
        probs = torch.softmax(logits, 1)[0]
        labels = model.config.id2label
        idx = int(probs.argmax())

        st.subheader(f"Prediction: {labels[idx]} ({probs[idx]*100:.1f}%)")
        for i, p in enumerate(probs):
            st.write(f"{labels[i]}: {p*100:.1f}%")
            st.progress(float(p))
        if probs[idx] < 0.7:
            st.info("The model is not confident about this image.")

        c1, c2 = st.columns(2)
        c1.image(img, caption="Input", use_container_width=True)
        c2.image(ela_image(img), caption="Error Level Analysis (ELA)", use_container_width=True)

        st.markdown("#### Metadata check")
        if exif:
            st.success("Camera/EXIF metadata found.")
            st.json(exif)
        else:
            st.warning("No EXIF metadata. Common in AI images, but also in screenshots "
                       "and social-media images, so it is not proof.")

with tab2:
    st.markdown("""
**Model:** Organika/sdxl-detector (Swin Transformer, Hugging Face).  
**Extra checks:** Error Level Analysis (ELA) and EXIF metadata.  
**Limitations:** works best on SDXL-style images and may fail on other generators,
compressed or low-resolution images. Not forensic proof.
""")

with tab3:
    st.markdown("""
- Hands and fingers: extra, merged or oddly bent
- Text in the image: garbled letters
- Backgrounds: warped lines, repeated patterns
- Skin and hair: overly smooth, hair blending into background
- Lighting and shadows: inconsistent direction
- Eyes and reflections: mismatched or asymmetrical
""")
