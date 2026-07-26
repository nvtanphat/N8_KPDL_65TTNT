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
    from app.utils import load_model, predict, read_image
except ImportError:
    from config import CLASS_NAMES, CLASS_LABELS, MODELS, DISEASE_INFO
    from utils import load_model, predict, read_image

# Cấu hình trang với layout rộng và thu gọn padding để vừa 1 khung hình
st.set_page_config(
    page_title="Bean Leaf Disease AI Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Tối ưu Giao diện Nằm gọn trong 1 Màn hình (Single Viewport Layout)
CUSTOM_CSS = """
<style>
    /* Nén padding tổng thể để vừa màn hình */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Header đẹp mắt */
    .main-header {
        background: linear-gradient(135deg, #059669 0%, #10B981 50%, #047857 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }

    /* Thẻ Container Bo góc & Hiệu ứng Glassmorphic */
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    
    /* Metric Cards */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-healthy { background-color: #d1fae5; color: #065f46; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    
    /* Thu gọn bớt sidebar padding */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # Header chính dạng Banner
    st.markdown("""
    <div class="main-header">
        <div>
            <h2>🍃 Hệ Thống Chẩn Đoán & Phân Vùng Bệnh Lá Đậu (Bean Leaf AI)</h2>
            <p>Phân loại tổn thương & Phân vùng vị trí ổ bệnh realtime bằng Deep Learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("⚙️ Cấu hình Mô hình")
        
        model_type = st.selectbox("Chọn mô hình AI:", list(MODELS.keys()), index=0)
        cfg = MODELS[model_type]
        
        st.caption(f"**Khung phần mềm:** `{cfg.get('framework', 'PyTorch')}`")
        st.caption(f"**Kích thước ảnh:** `{cfg['img_size'][0]}x{cfg['img_size'][1]}`")
        
        st.divider()
        compare_mode = st.checkbox("🔍 Chế độ So sánh Tất cả Model", value=False)
        
        with st.expander("ℹ️ Chi tiết Mô hình & Tập dữ liệu"):
            st.write(cfg.get('description', ''))
            st.caption(f"Dataset: {cfg.get('dataset', '')}")
            
        st.divider()
        st.markdown("**Các lớp bệnh chẩn đoán:**")
        for cls in CLASS_NAMES:
            st.markdown(f"• **{CLASS_LABELS.get(cls, cls)}**")

    if compare_mode:
        compare_view()
    else:
        single_view(model_type)


def single_view(model_type):
    model = load_cached_model(model_type)
    if model is None:
        st.error(f"❌ Không tìm thấy file checkpoint mô hình cho {model_type}")
        return
    
    col1, col2 = st.columns([1.1, 1.2], gap="medium")
    cfg = MODELS[model_type]
    
    with col1:
        st.markdown("##### 📥 1. Chọn hoặc Tải ảnh lá đậu")
        uploaded = st.file_uploader("Kéo thả file ảnh (JPG, PNG) vào đây", type=['jpg', 'jpeg', 'png'], key="single")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Ảnh lá đậu cần chẩn đoán", use_container_width=True)
        else:
            st.info("👈 Hãy tải ảnh lá đậu lên để bắt đầu phân tích.")
    
    with col2:
        st.markdown("##### 📊 2. Kết quả Phân tích & Chẩn đoán")
        if not uploaded:
            st.warning("Vui lòng tải ảnh lên ở khung bên trái.")
        else:
            if st.button("🚀 Thực hiện Chẩn đoán Tức thì", type="primary", use_container_width=True):
                with st.spinner("Đang chạy suy luận qua mô hình AI..."):
                    result = predict(model, image, model_type)
                    show_result(result, model_type)


def compare_view():
    col1, col2 = st.columns([1.1, 1.2], gap="medium")
    
    with col1:
        st.markdown("##### 📥 Chọn ảnh để So sánh Đa Mô hình")
        uploaded = st.file_uploader("Kéo thả file ảnh vào đây", type=['jpg', 'jpeg', 'png'], key="compare")
        
        if uploaded:
            image = read_image(uploaded.read())
            st.image(image, caption="Ảnh mẫu thử nghiệm", use_container_width=True)
    
    with col2:
        st.markdown("##### ⚖️ Kết quả So sánh Đồng thời 5 Mô hình")
        if not uploaded:
            st.info("Upload ảnh để bắt đầu so sánh giữa các kiến trúc.")
        else:
            if st.button("🔍 So sánh Tất cả Mô hình", type="primary", use_container_width=True):
                with st.spinner("Đang chạy dự đoán trên tất cả 5 mô hình..."):
                    results = {}
                    for m in MODELS.keys():
                        model = load_cached_model(m)
                        if model:
                            results[m] = predict(model, image, m)
                    
                    # Bảng so sánh kết quả
                    df = pd.DataFrame([{
                        'Mô hình': m, 
                        'Kết quả': CLASS_LABELS.get(r['class'], r['class']),
                        'Độ tin cậy': f"{r['confidence']:.1f}%",
                        'Khung phần mềm': MODELS[m].get('framework', 'N/A')
                    } for m, r in results.items()])
                    
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    plot_compare_bars(results)


def plot_compare_bars(results):
    st.markdown("###### Biểu đồ Xác suất (%) giữa các Mô hình:")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(results)
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#8B5CF6', '#EF4444']

    for i, (m, r) in enumerate(results.items()):
        vals = [r['probabilities'].get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + (i - len(results)/2 + 0.5) * w, vals, w, label=m, color=colors[i % len(colors)])
    
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in CLASS_NAMES], fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('Xác suất (%)', fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


@st.cache_resource
def load_cached_model(model_type):
    return load_model(model_type)


def show_result(result, model_type):
    # Nếu là YOLO segmentation có result
    if 'segmentation_result' in result and result['segmentation_result']:
        st.markdown("###### 🎯 Khoanh vùng Ổ Bệnh (Instance Segmentation):")
        img_seg = result['segmentation_result'].plot()[:, :, ::-1]
        st.image(img_seg, caption="Mặt nạ phân vùng tổn thương (YOLOv8)", use_container_width=True)
    
    pred_class = result['class']
    conf = result['confidence']
    class_label = CLASS_LABELS.get(pred_class, pred_class)
    
    # Hiển thị badge kết quả chẩn đoán chính
    badge_class = "badge-healthy" if pred_class == "healthy" else ("badge-danger" if pred_class == "angular_leaf_spot" else "badge-warning")
    
    st.markdown(f"""
    <div class="card-box">
        <h4 style="margin:0 0 0.5rem 0; color:#1F2937;">Kết quả: 
            <span class="metric-badge {badge_class}">{class_label}</span>
        </h4>
        <p style="margin:0; font-size:1.1rem; color:#4B5563;">Độ tin cậy chẩn đoán: <b>{conf:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Thanh xác suất từng lớp (Progress bars gọn gàng)
    st.markdown("###### Phân phối Xác suất Chẩn đoán:")
    for cls in CLASS_NAMES:
        prob = result['probabilities'].get(cls, 0.0)
        label = CLASS_LABELS.get(cls, cls)
        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.caption(f"**{label}**")
        with col_b:
            st.progress(min(max(int(prob), 0), 100), text=f"{prob:.1f}%")
            
    # Khuyến nghị y tế nông nghiệp
    info = DISEASE_INFO.get(pred_class)
    if info:
        with st.expander("💡 Tóm tắt Bệnh lý & Khuyến nghị Điều trị Nông nghiệp", expanded=True):
            st.markdown(f"**Mô tả:** {info['description']}")
            if info['symptoms']:
                st.markdown("**Triệu chứng điển hình:** " + ", ".join(info['symptoms'][:3]))
            st.success(f"**Gợi ý xử lý:** {info['recommendation']}")


if __name__ == "__main__":
    main()
