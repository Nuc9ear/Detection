from __future__ import annotations

import base64
import os
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from detection_app.inference import InferenceError, prediction_service


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "assets" / "metrics"
API_URL = os.getenv("POWERGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")

CLASS_RU = {
    "powerline": {
        "vibration_damper": "Vibration damper", "festoon_insulators": "Festoon insulators",
        "traverse": "Crossarm", "nest": "Bird nest",
        "safety_sign+": "Safety sign", "bad_insulator": "Faulty insulator",
        "damaged_insulator": "Damaged insulator", "polymer_insulators": "Polymer insulator",
    },
    "fracture": {
        "fracture": "Fracture", "elbow positive": "Elbow fracture",
        "fingers positive": "Finger fracture", "forearm fracture": "Forearm fracture",
        "humerus fracture": "Humerus fracture", "humerus": "Humerus",
        "shoulder fracture": "Shoulder fracture", "wrist positive": "Wrist fracture",
    },
    "vehicle": {
        "bicycle": "Bicycle", "car": "Car", "van": "Van",
        "truck": "Truck", "tricycle": "Tricycle",
        "awning-tricycle": "Awning tricycle", "bus": "Bus", "motor": "Motorcycle",
    },
}

st.set_page_config(
    page_title="VisionGuard AI · Object Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#111111; --muted:#555555; --blue:#0645ad; --line:#dddddd; --soft:#f7f9fc; }
      html, body, [class*="css"], .stApp { font-family: ui-monospace, SFMono-Regular,
        Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important; }
      .stApp { background: #ffffff; color: var(--ink); }
      #MainMenu, footer { display:none !important; }
      [role="dialog"] .e1usqyca0:has(img[alt="Snowflake"]) { display:none !important; }
      [role="dialog"] .e1usqyca4 { grid-template-columns:repeat(2,minmax(0,1fr)) !important; }
      [role="dialog"] .e1usqyca0:nth-child(3) button:last-of-type { display:none !important; }
      .block-container { max-width: 1120px; padding-top: 2.2rem; }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { color: var(--ink) !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.2rem; letter-spacing: -.02em; }
      [data-testid="stSidebar"] label:has(input:checked) { color: var(--blue) !important; font-weight: 700; }
      .hero { padding: 1.1rem 0 1.65rem; color: var(--ink); background: #ffffff;
              border-bottom: 1px solid var(--line); margin-bottom: 1.25rem; }
      .eyebrow { color:var(--blue); letter-spacing:.05em; text-transform:uppercase;
                 font-size:.72rem; font-weight:700; }
      .hero h1 { margin:.5rem 0 .65rem; font-size:clamp(2rem,4vw,3.2rem);
                 letter-spacing:-.055em; line-height:1.05; }
      .hero p { max-width:800px; margin:0; font-size:.98rem; line-height:1.65; color:var(--muted); }
      .hero-tags { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
      .hero-tags span { color:var(--blue); border:1px solid var(--line); border-radius:999px;
                        padding:.28rem .7rem; font-size:.76rem; background:#fff; }
      .notice { padding:.8rem 1rem; border-radius:10px; background:#fffaf0;
                border:1px solid #ead9ad; color:#5d4a19; margin:.5rem 0 1rem; }
      div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
                                    padding:.8rem 1rem; border-radius:10px; box-shadow:none; }
      [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line) !important; }
      [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
        background:#fff !important; border-color:var(--line) !important; border-radius:9px !important; }
      [data-testid="stFileUploaderDropzoneInstructions"] span { font-size:0 !important; }
      [data-testid="stFileUploaderDropzoneInstructions"] span::after {
        content:"Drag and drop a file here"; font-size:.86rem; }
      [data-testid="stFileUploaderDropzoneInstructions"] small { font-size:0 !important; }
      [data-testid="stFileUploaderDropzoneInstructions"] small::after {
        content:"Up to 20 MB · JPG, PNG or WEBP"; font-size:.72rem; }
      .stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--blue) !important; }
      .stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
        border-radius:8px !important; border:1px solid var(--blue) !important;
        font-weight:700 !important; box-shadow:none !important; transition:.15s ease; }
      [data-testid="stBaseButton-primary"] { background:var(--blue) !important; color:#fff !important; }
      [data-testid="stBaseButton-primary"] p { color:#fff !important; }
      .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stLinkButton"] a:hover {
        background:#003b91 !important; color:#fff !important; border-color:#003b91 !important; }
      a { color:var(--blue) !important; }
      h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
      .footer { text-align:center; color:#777; border-top:1px solid var(--line);
                padding:1.5rem 0 .4rem; margin-top:2rem; font-size:.76rem; }
      @media (max-width: 1100px) {
        .block-container { padding-left:1.35rem; padding-right:1.35rem; }
        [data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:1.2rem !important; }
        [data-testid="stColumn"] { width:100% !important; flex:1 1 100% !important; }
        .hero h1 { font-size:clamp(2rem,7vw,2.8rem); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit owns the Deploy dialog, so this tiny observer replaces its first
# "Learn more" action with a real mail link and removes the second one.
components.html(
    """
    <script>
    (() => {
      const parentDoc = window.parent.document;
      const patchDeployDialog = () => {
        const dialog = parentDoc.querySelector('[role="dialog"]');
        if (!dialog) return;
        const cards = [...dialog.querySelectorAll('.e1usqyca0')];
        const community = cards.find(card => card.textContent.includes('Streamlit Community Cloud'));
        const other = cards.find(card => card.textContent.includes('Other platforms'));

        if (community && !community.querySelector('[data-contact-link]')) {
          const learn = [...community.querySelectorAll('button')]
            .find(button => button.textContent.trim() === 'Learn more');
          if (learn) {
            const link = parentDoc.createElement('a');
            link.href = 'mailto:egrishin.25@saif.sjtu.edu.cn';
            link.textContent = 'Contact us';
            link.setAttribute('data-contact-link', 'true');
            link.style.cssText = [
              'color:#0645ad', 'font-weight:700', 'text-decoration:none',
              'font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace',
              'padding:0.55rem 0.35rem', 'white-space:nowrap'
            ].join(';');
            learn.replaceWith(link);
          }
        }

        if (other) {
          [...other.querySelectorAll('button')]
            .filter(button => button.textContent.trim() === 'Learn more')
            .forEach(button => button.remove());
        }
      };
      new MutationObserver(patchDeployDialog).observe(parentDoc.body, {
        childList: true, subtree: true
      });
      patchDeployDialog();
    })();
    </script>
    """,
    height=0,
    width=0,
)


def api_tasks() -> list[dict[str, Any]]:
    if not API_URL:
        return [item.model_dump() for item in prediction_service.list_tasks()]
    response = requests.get(f"{API_URL}/api/v1/tasks", timeout=8)
    response.raise_for_status()
    return response.json()


def api_models(task_id: str) -> list[dict[str, Any]]:
    if not API_URL:
        return [item.model_dump() for item in prediction_service.list_models(task_id)]
    response = requests.get(f"{API_URL}/api/v1/models", params={"task_id": task_id}, timeout=8)
    response.raise_for_status()
    return response.json()


def run_prediction(
    content: bytes, filename: str, model_id: str, confidence: float, iou: float
) -> dict[str, Any]:
    if not API_URL:
        return prediction_service.predict(
            content, model_id=model_id, confidence=confidence, iou=iou
        ).model_dump()
    response = requests.post(
        f"{API_URL}/api/v1/predict",
        files={"file": (filename, content, "image/jpeg")},
        data={"model_id": model_id, "confidence": confidence, "iou": iou},
        timeout=180,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


@st.cache_data(show_spinner=False)
def decode_result(encoded: str) -> bytes:
    return base64.b64decode(encoded)


with st.sidebar:
    st.markdown("## ◉ VisionGuard AI")
    page = st.radio("Section", ["Inspection", "About"])


if page == "Inspection":
    st.markdown(
        """
        <section class="hero">
          <h1>Detection for three applied tasks</h1>
          <p>Select a domain, upload an image, and receive localized detections.
          Each task offers a fast and an accurate model profile.</p>
          <div class="hero-tags">
            <span>Power lines</span><span>X-ray</span><span>Vehicles</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        tasks = api_tasks()
    except Exception as exc:
        st.error(f"The inference server is unavailable: {exc}")
        st.stop()

    controls, workspace = st.columns([1, 2.15], gap="large")
    with controls:
        st.subheader("Analysis settings")
        task_labels = {task["title"]: task for task in tasks}
        selected_task_title = st.selectbox("Detection task", task_labels.keys())
        selected_task = task_labels[selected_task_title]
        models = api_models(selected_task["id"])
        labels = {model["title"]: model for model in models}
        model_titles = list(labels.keys())
        default_model_index = 1 if selected_task["id"] == "fracture" else 0
        selected_title = st.selectbox(
            "Model profile", model_titles, index=default_model_index
        )
        selected = labels[selected_title]
        st.caption(selected["description"])
        default_confidence = 0.10 if selected_task["id"] == "fracture" else 0.44
        confidence = st.slider(
            "Confidence threshold", 0.05, 0.95, default_confidence, 0.01,
            help="A more sensitive 0.10 threshold is used for low-contrast X-rays."
        )
        iou = st.slider("IoU for duplicate suppression", 0.10, 0.90, 0.45, 0.01)
        uploaded = st.file_uploader(
            "Image to analyze", type=["jpg", "jpeg", "png", "webp"],
            help="Up to 20 MB. Do not upload sensitive data."
        )
        analyze = st.button("Run detection", type="primary", use_container_width=True)

        st.caption(selected_task["disclaimer"])

    with workspace:
        if uploaded is None:
            st.markdown("#### Target classes")
            st.write(" · ".join(CLASS_RU[selected_task["id"]].values()))
        else:
            image = Image.open(uploaded).convert("RGB")
            if not analyze:
                st.image(image, caption="Original image", use_container_width=True)
            else:
                content = uploaded.getvalue()
                try:
                    with st.spinner("Analyzing the image…"):
                        result = run_prediction(
                            content, uploaded.name, selected["id"], confidence, iou
                        )
                except (InferenceError, RuntimeError, requests.RequestException) as exc:
                    st.error(str(exc))
                    st.stop()

                rendered = decode_result(result["annotated_image"])
                st.image(rendered, caption="Detection result", use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Objects", len(result["detections"]))
                m2.metric("Inference", f'{result["inference_ms"]:.0f} ms')
                m3.metric("Resolution", f'{result["image_width"]}×{result["image_height"]}')

                if result["detections"]:
                    counts = Counter(item["class_name"] for item in result["detections"])
                    st.markdown("#### Detected objects")
                    for class_name, count in counts.most_common():
                        translated = CLASS_RU[result["task_id"]].get(class_name, class_name)
                        st.write(f"**{translated}** — {count}")
                    st.download_button(
                        "Download annotated image",
                        data=rendered,
                        file_name=f"detected_{uploaded.name.rsplit('.', 1)[0]}.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.warning("No objects were found above the selected confidence threshold.")

elif page == "Model quality":
    st.title("Target model quality")
    st.caption("Validation set · 8 power-line object classes")
    k1, k2, k3 = st.columns(3)
    k1.metric("mAP@0.5", "0.902", "> 0.5")
    k2.metric("Best F1", "0.87", "conf = 0.439")
    k3.metric("Classes", "8")

    st.markdown(
        "Mean AP@0.5 is **0.902**. The strongest classes are `bad_insulator` "
        "(0.979), `safety_sign+` (0.977), and `nest` (0.948). The main improvement "
        "opportunities are `polymer_insulators` (0.771) and `vibration_damper` (0.792): "
        "more small-object examples and hard-negative mining may help."
    )
    tab1, tab2, tab3 = st.tabs(["Precision–Recall", "F1 and threshold", "Confusion matrix"])
    with tab1:
        st.image(str(METRICS_DIR / "BoxPR_curve.png"), use_container_width=True)
    with tab2:
        st.image(str(METRICS_DIR / "BoxF1_curve.png"), use_container_width=True)
        st.info("The default 0.44 threshold maximizes the overall validation F1 score.")
    with tab3:
        normalized = st.toggle("Normalized matrix", value=True)
        filename = "confusion_matrix_normalized.png" if normalized else "confusion_matrix.png"
        st.image(str(METRICS_DIR / filename), use_container_width=True)

else:
    st.title("About")
    st.markdown(
        """
        **VisionGuard AI** is a unified MVP for three product-track tasks:
        power-line damage, fractures in X-rays, and vehicles in aerial imagery.
        Results are assistive and must be reviewed by a qualified specialist.

        **Architecture:** Streamlit provides the user interface, FastAPI serves the
        HTTP API, and Ultralytics YOLO performs inference. Users can select a fast or
        accurate model profile for each domain.

        **Limitations:** quality depends on viewpoint, lighting, capture altitude, and
        domain shift. Production use requires field validation, drift monitoring, and
        mandatory expert review.
        """
    )
    st.markdown("#### Tasks, datasets, and classes")
    for task in api_tasks():
        with st.expander(task["title"]):
            st.write(task["description"])
            st.write(" · ".join(CLASS_RU[task["id"]].values()))
            st.link_button("Open dataset", task["dataset_url"])

st.markdown(
    '<div class="footer">VisionGuard AI · educational product-track detection MVP</div>',
    unsafe_allow_html=True,
)
