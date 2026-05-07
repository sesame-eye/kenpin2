import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="段数カウンター AI-Ver3", layout="centered")
st.title("🤖 学習型段数カウンター")

# --- AIの脳（学習データ）の初期化 ---
if 'brain_data' not in st.session_state:
    st.session_state.brain_data = [] # 画像の明るさなどの特徴
    st.session_state.best_thresholds = [] # その時の正解感度

# --- サイドバー：感度設定 ---
st.sidebar.header("設定")
manual_threshold = st.sidebar.slider("現在の感度（手動調整用）", 10, 100, 35)

# --- 写真の入力 ---
option = st.radio("写真の入力方法", ("カメラで撮影", "ライブラリから選ぶ"))
uploaded_file = st.camera_input("撮影") if option == "カメラで撮影" else st.file_uploader("選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 画像の読み込みと特徴の抽出
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # AIが判断するための「画像の雰囲気（明るさと色の差）」を数値化
    img_feature = [np.mean(gray), np.std(gray)]

    # --- AIによる感度の自動予測（3回以上学習したら発動） ---
    current_threshold = manual_threshold
    if len(st.session_state.brain_data) >= 3:
        knn = KNeighborsRegressor(n_neighbors=min(3, len(st.session_state.brain_data)))
        knn.fit(st.session_state.brain_data, st.session_state.best_thresholds)
        predicted_threshold = knn.predict([img_feature])[0]
        current_threshold = int(predicted_threshold)
        st.sidebar.info(f"AI予測により感度を {current_threshold} に自動調整しました")

    # --- 段数カウント処理 ---
    h, w = img.shape[:2]
    target_h = 1000
    scale = target_h / h
    proc_img = cv2.resize(img, (int(w * scale), target_h))
    gray_proc = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges = cv2.filter2D(gray_proc, -1, kernel)
    
    pw = edges.shape[1]
    # 中央の1本をスキャン
    line = edges[:, int(pw * 0.5)]
    count = 0
    in_peak = False
    for val in line:
        if val > current_threshold:
            if not in_peak:
                count += 1
                in_peak = True
        else:
            in_peak = False
    
    # 表示
    st.image(img, use_column_width=True)
    st.markdown(f"<h1 style='text-align: center;'>{count} 段</h1>", unsafe_allow_html=True)

    # --- ここが「学習ボタン」です ---
    st.write("---")
    st.subheader("🎓 AIに正解を覚えさせる")
    st.write("1. 左のバーを動かして段数を正解に合わせる")
    st.write("2. 下のボタンを押してAIに記憶させる")
    
    if st.button("この設定（感度）をAIに覚えさせる"):
        st.session_state.brain_data.append(img_feature)
        st.session_state.best_thresholds.append(manual_threshold)
        st.success(f"学習しました！(現在 {len(st.session_state.brain_data)} パターン記憶済み)")
        if len(st.session_state.brain_data) < 3:
            st.warning("あと数回学習させると、自動調整が始まります。")