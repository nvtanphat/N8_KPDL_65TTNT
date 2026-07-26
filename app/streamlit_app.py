import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from app.config import CLASS_NAMES, CLASS_LABELS, MODELS, DISEASE_INFO
    from app.utils import load_model, predict, read_image, supports_gradcam, generate_gradcam
except ImportError:
    from config import CLASS_NAMES, CLASS_LABELS, MODELS, DISEASE_INFO
    from utils import load_model, predict, read_image, supports_gradcam, generate_gradcam

st.set_page_config(page_title="Phân Loại Bệnh Lá Đậu", layout="wide")

# CSS fix gọn giao diện trong 1 khung hình web
st.markdown("""
<style>
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
    }
    .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    img {
        max-height: 210px !important;
        object-fit: contain !important;
    }
    .stButton button {
        padding: 0.3rem 0.6rem !important;
    }
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.header("Cấu hình mô hình")
        
        model_type = st.selectbox("Chọn mô hình:", list(MODELS.keys()))
        
        cfg = MODELS[model_type]
        st.info(f"Khung phần mềm: {cfg.get('framework', 'N/A')}")
        
        compare_mode = st.checkbox("So sánh các mô hình")
        
        st.write("**Các lớp phân loại:**")
        for cls in CLASS_NAMES:
            st.write(f"• {CLASS_LABELS.get(cls, cls)}")
    
    if compare_mode:
        compare_view()
    else:
        single_view(model_type)


def single_view(model_type):
    model = load_cached_model(model_type)
    if model is None:
        st.error(f"Không tìm thấy model: {MODELS[model_type]['file']}")
        return
    
    col1, col2 = st.columns([1.2, 1])
    cfg = MODELS[model_type]
    
    with col1:
        st.subheader("Chọn hình ảnh để phân tích")
        uploaded = st.file_uploader("Kéo thả file vào đây", type=['jpg', 'jpeg', 'png'], key="single")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Ảnh upload", use_column_width=True)
        
        # Model info - bên dưới uploader
        st.subheader(f"Mô hình: {model_type}")
        st.write(f"**Kích thước:** {cfg['img_size'][0]}x{cfg['img_size'][1]}")
        
        with st.expander("Mô tả mô hình"):
            st.write(cfg.get('description', 'Không có mô tả'))
        
        with st.expander("Thông tin bộ dữ liệu"):
            st.write(cfg.get('dataset', 'Không có thông tin'))
    
    with col2:
        if not uploaded:
            st.info("Upload ảnh để bắt đầu phân tích")
        else:
            show_gradcam = False
            if supports_gradcam(model_type):
                show_gradcam = st.checkbox("Hiển thị Grad-CAM Heatmap", value=True)

            if st.button("Phân Tích", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    result = predict(model, image, model_type)
                    show_result(result)

                if show_gradcam:
                    with st.spinner("Đang tạo Grad-CAM..."):
                        overlay = generate_gradcam(model, image, model_type)
                    if overlay is not None:
                        st.subheader("Grad-CAM Heatmap")
                        st.image(overlay, caption="Vùng đỏ/vàng: Vùng AI tập trung nhận diện", use_column_width=True)


def compare_view():
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("So Sánh Tất Cả Model")
        uploaded = st.file_uploader("Kéo thả file vào đây", type=['jpg', 'jpeg', 'png'], key="compare")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Ảnh upload", use_column_width=True)
    
    with col2:
        if not uploaded:
            st.info("Upload ảnh để so sánh")
        else:
            if st.button("So Sánh Tất Cả", type="primary", use_container_width=True):
                with st.spinner("Đang so sánh..."):
                    results = {}
                    for m in MODELS.keys():
                        model = load_cached_model(m)
                        if model:
                            results[m] = predict(model, image, m)
                    
                    # Bảng so sánh
                    df = pd.DataFrame([{
                        'Model': m, 
                        'Dự đoán': CLASS_LABELS.get(r['class'], r['class']),
                        'Confidence': f"{r['confidence']:.1f}%",
                        'Framework': MODELS[m].get('framework', 'N/A')
                    } for m, r in results.items()])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    # Biểu đồ
                    plot_compare(results)


def plot_compare(results):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(results)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for i, (m, r) in enumerate(results.items()):
        vals = [r['probabilities'].get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + (i - len(results)/2 + 0.5) * w, vals, w, label=m, color=colors[i % len(colors)])
    
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in CLASS_NAMES])
    ax.legend(fontsize='small')
    ax.set_ylabel('Confidence (%)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


@st.cache_resource
def load_cached_model(model_type):
    return load_model(model_type)


def show_result(result):
    if 'segmentation_result' in result and result['segmentation_result']:
        img = result['segmentation_result'].plot()[:, :, ::-1]
        st.image(img, caption="Segmentation", use_column_width=True)
    
    st.dataframe(pd.DataFrame([{
        'Loại': CLASS_LABELS.get(result['class'], result['class']),
        'Confidence': f"{result['confidence']:.1f}%"
    }]), hide_index=True)
    
    fig, ax = plt.subplots(figsize=(5, 2.0))
    probs = list(result['probabilities'].values())
    labels = [CLASS_LABELS.get(c, c) for c in result['probabilities'].keys()]
    colors = ['#e74c3c' if p == max(probs) else '#3498db' for p in probs]
    ax.bar(labels, probs, color=colors)
    ax.set_ylabel('Confidence (%)')
    st.pyplot(fig)
    plt.close()
    
    info = DISEASE_INFO.get(result['class'])
    if info:
        st.write(f"**{info['name']}** - {info['severity']}")
        st.write(info['description'])
        st.info(info['recommendation'])


if __name__ == "__main__":
    main()
