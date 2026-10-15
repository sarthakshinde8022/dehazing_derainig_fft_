
import streamlit as st

st.set_page_config(
    page_title="Clarity — Image Restoration",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Clarity")
st.subheader("Frequency-Domain Image Restoration")

st.markdown("""
Welcome to **Clarity**, an image processing mini-project
using Fast Fourier Transform (FFT) techniques.

### Available techniques

**1. Image Dehazing**

Reduce the effects of haze using frequency-domain
homomorphic filtering.

**2. Rain Streak Removal**

Suppress directional rain streaks using frequency-domain
notch filtering.

Upload an image, adjust the filter parameters, and compare
the restored result with your original image.
""")

st.divider()

st.page_link(
    "pages/2_Try_the_Demo.py",
    label="Try the Image Restoration Demo",
    icon="🧪",
)
