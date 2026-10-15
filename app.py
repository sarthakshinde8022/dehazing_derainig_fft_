
import streamlit as st

st.set_page_config(
    page_title="Clarity | Image Restoration Lab",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Clarity: Image Restoration Lab")

st.markdown(
    """
    ### Understand image restoration through Fourier-domain filtering

    Welcome! This project demonstrates two image-processing techniques:

    - **Image dehazing:** Improving the appearance of images affected by haze.
    - **Rain streak removal:** Reducing visible rain streaks in photographs.

    You can explore how each algorithm works, adjust its parameters,
    visualize its frequency-domain filter, and compare the original
    and processed images.
    """
)

st.info(
    "New to image processing? Start with the learning guide below. "
    "You can experiment with the demo without knowing the mathematics "
    "in advance."
)

st.divider()

st.header("1. What is image restoration?")

st.write(
    "Image restoration is the process of improving a degraded image. "
    "A degraded image may contain haze, rain, blur, or other unwanted "
    "visual effects. The goal is to make useful information easier "
    "to see while preserving as much real image detail as possible."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌫️ Image dehazing")
    st.write(
        "Haze makes distant objects look pale, washed out, or low in "
        "contrast. This project uses homomorphic filtering to adjust "
        "slowly varying brightness and improve local contrast."
    )
    st.caption(
        "Important: this approach does not fully model atmospheric "
        "scattering and may not recover the original colors or depth."
    )

with col2:
    st.subheader("🌧️ Rain streak removal")
    st.write(
        "Rain can appear as bright, elongated streaks across a photograph. "
        "This project uses frequency-domain notch filtering to reduce "
        "some directional patterns associated with those streaks."
    )
    st.caption(
        "Important: buildings, poles, and other genuine edges can have "
        "similar patterns, so filtering may soften real details too."
    )

st.divider()

st.header("2. The main idea: what is the Fourier domain?")

st.write(
    "An image is normally viewed as pixels in the spatial domain. "
    "The Fast Fourier Transform (FFT) represents image variations as "
    "different spatial frequencies. This gives us another way to analyze "
    "and filter an image."
)

with st.expander("📘 Learn the essential terms", expanded=True):
    st.markdown(
        """
        **Pixel**  
        The smallest individual picture element in a digital image.
        Each pixel stores a brightness value or color.

        **Spatial domain**  
        The ordinary image view: pixels arranged in rows and columns.

        **Fast Fourier Transform (FFT)**  
        An efficient mathematical algorithm that converts image data
        into a representation of its spatial frequencies.

        **Frequency domain**  
        A representation that describes how quickly image brightness
        changes across space.

        **Low spatial frequencies**  
        Slowly changing brightness and broad image structures, such as
        smooth gradients and large areas of illumination.

        **High spatial frequencies**  
        Rapid changes in brightness, commonly associated with edges,
        fine textures, and some noise.

        **Spectrum**  
        A visual representation of the strength of different frequencies.
        The bright and dark regions help show where frequency energy is
        concentrated.

        **Filter mask**  
        A mathematical map that controls which frequencies are preserved,
        attenuated, or amplified during processing.
        """
    )

st.header("3. How do the two filters work?")

with st.expander("🌫️ Homomorphic filtering — dehazing"):
    st.markdown(
        """
        Homomorphic filtering is a technique for adjusting image
        illumination and contrast using frequency-domain processing.

        **Basic workflow**
        1. Convert the image into a representation suitable for filtering.
        2. Transform the image into the frequency domain.
        3. Apply a filter that changes low- and high-frequency components.
        4. Transform the result back into an image.

        **Why?** Broad illumination variations can be reduced while
        finer details are emphasized. However, this is an approximation,
        not a complete physical model of haze.
        """
    )

with st.expander("🌧️ Notch filtering — rain streak removal"):
    st.markdown(
        """
        A notch filter suppresses selected frequency regions while
        preserving other regions as much as possible.

        **Basic workflow**
        1. Transform the rainy image into the frequency domain.
        2. Estimate or select the direction associated with the streaks.
        3. Apply a notch mask to attenuate the selected frequency band.
        4. Transform the filtered data back into an image.

        **Why?** Directional streaks can create recognizable frequency
        patterns. Unfortunately, genuine edges with similar patterns
        can also be weakened.
        """
    )

st.divider()

st.header("4. How to use the demo")

st.markdown(
    """
    **Step 1 — Choose a task:** Open the demo and select Dehazing or
    Rain Streak Removal.

    **Step 2 — Upload an image:** Use a JPG, JPEG, or PNG image.
    For a fair assessment, choose an image that actually contains haze
    or rain streaks.

    **Step 3 — Adjust the settings:** Start with the default values.
    Change one parameter at a time to observe its effect.

    **Step 4 — Compare results:** View the original and processed
    images side by side. Inspect the filter visualizations where available.

    **Step 5 — Download the result:** Save the restored image if useful.

    **Step 6 — Evaluate quality:** If you have a corresponding clean
    reference image, upload it to calculate PSNR and SSIM.
    """
)

st.warning(
    "Image restoration is not guaranteed to recover the exact original "
    "scene. A visually pleasing result may still contain artifacts or "
    "lost detail. Always compare the output with the input."
)

st.divider()

st.header("5. Try it yourself")

st.write(
    "Ready to experiment? Open the interactive lab to process your image "
    "and explore the filter settings."
)

st.page_link(
    "pages/2_Try_the_Demo.py",
    label="🧪 Open the Image Restoration Demo",
    icon="📡",
    width="stretch",
)

st.caption(
    "Academic mini-project | FFT-based dehazing and rain streak removal"
)
