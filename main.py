import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.5", layout="centered")
st.title("📏 プロ仕様：高精度段数カウンター")

uploaded_file = st.sidebar.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])
camera_file = st.camera_input("またはカメラで撮影")
file = uploaded_file if uploaded_file else camera_file

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    
    # 1. 鮮明化処理（コントラストを極限まで高めて影を浮き彫りにする）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # 2. 横線エッジ強調
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges = cv2.filter2D(gray, -1, kernel)
    
    # 3. マルチラインスキャン（画像中央60%の範囲を細かくスキャン）
    scan_results = []
    start_x = int(w * 0.2)
    end_x = int(w * 0.8)
    
    for x in range(start_x, end_x, 10): # 10ピクセルごとにスキャン
        line = edges[:, x]
        # ノイズ除去のための閾値処理
        thresh = np.max(line) * 0.2
        count = 0
        peak = False
        for val in line:
            if val > thresh:
                if not peak:
                    count += 1
                    peak = True
            else:
                peak = False
        if count > 1: # 極端に少ない結果は除外
            scan_results.append(count)

    # 統計的に最も確からしい数値を算出
    if scan_results:
        # 外れ値を除いた平均を採用
        final_ans = int(np.percentile(scan_results, 75)) # 高めに出る値を採用（影が消えやすいため）
    else:
        final_ans = 0

    # --- 表示 ---
    st.image(img, use_column_width=True)
    st.markdown(f"""
        <div style="text-align: center; border: 2px solid #1f77b4; padding: 20px; border-radius: 10px;">
            <h3>判定された段数</h3>
            <h1 style="font-size: 80px; color: #1f77b4;">{final_ans} <span style="font-size: 30px;">段</span></h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ もしズレが大きい場合は、真横から、できるだけ明るい場所で撮影してください。")
