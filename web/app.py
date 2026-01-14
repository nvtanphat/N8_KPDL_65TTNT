"""
Ung dung Web phan loai benh la dau - Streamlit
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import CLASS_NAMES, CLASS_LABELS, MODELS, DISEASE_INFO
from utils import load_model, predict, read_image

st.set_page_config(page_title="Phan Loai Benh La Dau", layout="wide")


def main():
    with st.sidebar:
        st.header("Nguoi mau Cau hinh")
        
        model_type = st.selectbox("Mo hinh Chon:", list(MODELS.keys()))
        
        cfg = MODELS[model_type]
        st.info(f"Khung phan mem: {cfg.get('framework', 'N/A')}")
        
        compare_mode = st.checkbox("So sánh các mô hình")
        
        st.write("**Lop Cac phan loai:**")
        for cls in CLASS_NAMES:
            st.write(f"• {CLASS_LABELS.get(cls, cls)}")
    
    if compare_mode:
        compare_view()
    else:
        single_view(model_type)


def single_view(model_type):
    model = load_cached_model(model_type)
    if model is None:
        st.error(f"Khong tim thay model: {MODELS[model_type]['file']}")
        return
    
    col1, col2 = st.columns([1.2, 1])
    cfg = MODELS[model_type]
    
    with col1:
        st.subheader("Chon hinh anh de phan tich")
        uploaded = st.file_uploader("Keo tha file vao day", type=['jpg', 'jpeg', 'png'], key="single")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Anh upload", use_column_width=True)
        
        # Model info - ben duoi uploader
        st.subheader(f"Mo hinh: {model_type}")
        st.write(f"**Kich thuoc:** {cfg['img_size'][0]}x{cfg['img_size'][1]}")
        st.write(f"**Nha phat trien:** {cfg.get('developer', 'N/A')}")
        
        with st.expander("Mo hinh Mo ta"):
            st.write(cfg.get('description', 'Khong co mo ta'))
        
        with st.expander("Bo du lieu Thong tin"):
            st.write(cfg.get('dataset', 'Khong co thong tin'))
    
    with col2:
        if not uploaded:
            st.info("Upload anh de bat dau phan tich")
        else:
            if st.button("Phan Tich", type="primary", use_container_width=True):
                with st.spinner("Dang xu ly..."):
                    result = predict(model, image, model_type)
                    show_result(result)


def compare_view():
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("So Sanh Tat Ca Model")
        uploaded = st.file_uploader("Keo tha file vao day", type=['jpg', 'jpeg', 'png'], key="compare")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Anh upload", use_column_width=True)
    
    with col2:
        if not uploaded:
            st.info("Upload anh de so sanh")
        else:
            if st.button("So Sanh Tat Ca", type="primary", use_container_width=True):
                with st.spinner("Dang so sanh..."):
                    results = {}
                    for m in MODELS.keys():
                        model = load_cached_model(m)
                        if model:
                            results[m] = predict(model, image, m)
                    
                    # Bang so sanh
                    df = pd.DataFrame([{
                        'Model': m, 
                        'Du doan': CLASS_LABELS.get(r['class'], r['class']),
                        'Confidence': f"{r['confidence']:.1f}%",
                        'Framework': MODELS[m].get('framework', 'N/A')
                    } for m, r in results.items()])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    # Bieu do
                    plot_compare(results)


def plot_compare(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(results)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (m, r) in enumerate(results.items()):
        vals = [r['probabilities'].get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + (i - len(results)/2 + 0.5) * w, vals, w, label=m, color=colors[i % 4])
    
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in CLASS_NAMES])
    ax.legend()
    ax.set_ylabel('Confidence (%)')
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
        'Loai': CLASS_LABELS.get(result['class'], result['class']),
        'Confidence': f"{result['confidence']:.1f}%"
    }]), hide_index=True)
    
    fig, ax = plt.subplots(figsize=(6, 3))
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
