import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="SignalScope",
    page_icon="🔍",
    layout="wide"
)

# ---------- DESIGN ----------
st.markdown("""
<style>

.stApp {
    background: #0b1120;
    color: white;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 20px;
    margin-bottom: 35px;
}

.description {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    max-width: 750px;
    margin: auto;
    margin-bottom: 35px;
}

.result-box {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
}

.ai-result {
    color: #f87171;
    font-size: 30px;
    font-weight: 800;
}

.warning {
    background: #1e293b;
    padding: 18px;
    border-radius: 12px;
    color: #cbd5e1;
    margin-top: 25px;
}

.section {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown(
    '<div class="title">🔍 SIGNALSCOPE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Telling Real From Synthetic in the Age of Generative Media</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description">
    Upload an image and SignalScope will assess whether it is
    likely to be a real photograph or an AI-generated image.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- UPLOAD ----------
uploaded_file = st.file_uploader(
    "📤 Upload your image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ---------- AFTER UPLOAD ----------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section">Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section">SignalScope Analysis</div>',
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🔎 Analyze Image",
            use_container_width=True
        )

        if analyze:

            # DEMO RESULT
            # Actual AI model will be connected later.

            confidence = 88

            st.markdown(
                f"""
                <div class="result-box">

                <div class="ai-result">
                ⚠️ Likely AI-generated
                </div>

                <p>
                SignalScope estimates that this image is
                likely AI-generated.
                </p>

                <h3>Confidence</h3>

                <div style="
                    background:#334155;
                    border-radius:10px;
                    height:18px;
                ">

                    <div style="
                        width:{confidence}%;
                        background:#ef4444;
                        height:18px;
                        border-radius:10px;
                    "></div>

                </div>

                <h2 style="text-align:right;">
                    {confidence}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section">💡 Why this result?</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            - Unusual local visual patterns detected
            - Fine image details show inconsistencies
            - Evidence is concentrated in specific image regions
            """)

            st.markdown(
                '<div class="section">🔥 Evidence Heatmap</div>',
                unsafe_allow_html=True
            )

            st.info(
                "Grad-CAM heatmap will be connected after the AI model is trained."
            )

            st.markdown("""
            <div class="warning">

            ⚠️ <b>Important:</b><br><br>

            This is a likelihood assessment, not proof of an image's origin.
            AI detection can be uncertain, especially for images from new
            or unseen generators.

            </div>
            """, unsafe_allow_html=True)
