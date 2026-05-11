import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="段数カウンター Ver.4", layout="centered")
st.title("🤖 高精度AI段数カウンター")

# --- AIの脳（学習データ）の初期化 ---
if 'brain_data' not in st.session_state:
    st.session_state.brain_data = [] # 画像特徴
    st.session_state.best_thresholds = [] # 最適だった感度

# --- サイドバー：設定 ---
st.sidebar.header("AI設定")
# 初期感度
current_threshold = 35

# --- 写真の入力 ---
option = st.radio("写真の入力方法", ("カメラで撮影", "ライブラリから選ぶ"))
uploaded_file = st.camera_input("撮影") if option == "カメラで撮影" else st.file_uploader("選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 画像特徴の抽出（明るさ・コントラスト）
    img_feature = [np.mean(gray), np.std(gray)]

    # --- AIによる感度の自動予測 ---
    confidence = 0
    if len(st.session_state.brain_data) >= 2:
        knn = KNeighborsRegressor(n_neighbors=min(2, len(st.session_state.brain_data)))
        knn.fit(st.session_state.brain_data, st.session_state.best_thresholds)
        current_threshold = int(knn.predict([img_feature])[0])
        
        # 自信度の計算（データが増えるほど、また過去データに近いほど上昇）
        dist, _ = knn.kneighbors([img_feature])
        avg_dist = np.mean(dist)
        confidence = max(0, min(99, 100 - int(avg_dist))) 
    else:
        confidence = 10 # 初期値

    # --- 段数カウント処理 (物理スキャン) ---
    h, w = img.shape[:2]
    target_h = 1000
    scale = target_h / h
    proc_img = cv2.resize(img, (int(w * scale), target_h))
    gray_proc = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges = cv2.filter2D(gray_proc, -1, kernel)
    
    line = edges[:, int(edges.shape[1] * 0.5)]
    count = 0
    in_peak = False
    for val in line:
        if val > current_threshold:
            if not in_peak:
                count += 1
                in_peak = True
        else:
            in_peak = False
    
    # --- 結果表示 ---
    st.image(img, use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="判定結果", value=f"{count} 段")
    with col2:
        st.metric(label="AI自信度", value=f"{confidence} %")

    # --- 学習インターフェース (常に表示) ---
    st.write("---")
    st.subheader("🎓 AIに追加学習させる")
    st.write("もし結果が間違っていたら、正しい段数を入力して「学習」を押してください。")
    
    true_count = st.number_input("正しい段数は？", min_value=1, value=count if count > 0 else 40)
    
    if st.button("この画像を正解として学習させる"):
        # 正解の段数に一番近くなる「感度」を逆算（総当たりで探す）
        best_t = 35
        min_diff = 999
        for t in range(10, 80):
            temp_count = 0
            temp_in_peak = False
            for v in line:
                if v > t:
                    if not temp_in_peak:
                        temp_count += 1
                        temp_in_peak = True
                else:
                    temp_in_peak = False
            
            diff = abs(temp_count - true_count)
            if diff <= min_diff:
                min_diff = diff
                best_t = t
        
        # 算出した最適感度をAIに覚えさせる
        st.session_state.brain_data.append(img_feature)
        st.session_state.best_thresholds.append(best_t)
        st.success(f"学習完了！(記憶数: {len(st.session_state.brain_data)})。次回の予測精度が向上します。")
        st.rerun()
