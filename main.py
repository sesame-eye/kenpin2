import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="段数AI Ver4", layout="centered")
st.title("🛡️ 段数カウントAI Pro")

# --- 学習データの初期化 ---
if 'brain_data' not in st.session_state:
    st.session_state.brain_data = [] # 画像特徴
    st.session_state.best_thresholds = [] # 正解だった感度

# --- ロジック関数 ---
def count_layers(img, threshold):
    h, w = img.shape[:2]
    proc_img = cv2.resize(img, (int(w * (1000/h)), 1000))
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges = cv2.filter2D(gray, -1, kernel)
    line = edges[:, int(edges.shape[1] * 0.5)]
    count, in_peak = 0, False
    for val in line:
        if val > threshold:
            if not in_peak:
                count += 1
                in_peak = True
        else: in_peak = False
    return count

# --- 入力 ---
uploaded_file = st.camera_input("撮影")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = [np.mean(gray_full), np.std(gray_full)]

    # --- AI予測 ＆ 自信度算出 ---
    current_threshold = 35 # 初期値
    confidence = 0
    
    if len(st.session_state.brain_data) >= 1:
        knn = KNeighborsRegressor(n_neighbors=min(3, len(st.session_state.brain_data)))
        knn.fit(st.session_state.brain_data, st.session_state.best_thresholds)
        current_threshold = int(knn.predict([features])[0])
        
        # 自信度の計算（過去のデータとの距離から簡易算出）
        dists, _ = knn.kneighbors([features])
        avg_dist = np.mean(dists)
        confidence = max(5, min(99, int(100 - (avg_dist * 0.5)))) # 距離が近いほど高%
    
    # 判定
    result_count = count_layers(img, current_threshold)

    # --- 結果表示 ---
    st.image(img, use_column_width=True)
    
    col1, col2 = st.columns(2)
    col1.metric("判定結果", f"{result_count} 段")
    col2.metric("AI自信度", f"{confidence}%" if len(st.session_state.brain_data) > 0 else "測定不能")

    # --- 学習セクション（常に表示） ---
    st.write("---")
    st.subheader("🎓 AIに正解を教えて鍛える")
    true_val = st.number_input("本当の段数は何段ですか？", min_value=1, value=result_count)
    
    if st.button("この画像を正解として学習させる"):
        # 逆算ロジック：入力された段数に最も近くなる「感度」を自動で見つける
        best_t = 35
        min_diff = 999
        for t in range(10, 100):
            c = count_layers(img, t)
            if abs(c - true_val) < min_diff:
                min_diff = abs(c - true_val)
                best_t = t
        
        st.session_state.brain_data.append(features)
        st.session_state.best_thresholds.append(best_t)
        st.success(f"学習完了！(正解:{true_val}段 / 最適感度:{best_t})")
        st.rerun()

# データリセット
if st.sidebar.button("学習データを初期化"):
    st.session_state.brain_data = []
    st.session_state.best_thresholds = []
    st.rerun()
