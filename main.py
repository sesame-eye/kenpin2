import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Pro", layout="centered")
st.title("🔢 段数デジタルカウンター")
st.write("画像の中央をスキャンして、段の境目を物理的にカウントします。")

# --- 設定（サイドバー） ---
st.sidebar.header("判定感度の調整")
# 感度：数字が小さいほど敏感に、大きいほど鈍感になります
threshold = st.sidebar.slider("感度設定", 10, 100, 35)

# --- 写真の入力 ---
option = st.radio("写真の入力方法", ("カメラで撮影", "ライブラリから選ぶ"))
uploaded_file = st.camera_input("パシャリと撮影") if option == "カメラで撮影" else st.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    
    # 処理用にリサイズ（高さを1000pxに固定して基準を作る）
    target_h = 1000
    scale = target_h / h
    proc_img = cv2.resize(img, (int(w * scale), target_h))
    
    # グレー変換
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    
    # 縦方向の変化（エッジ）を抽出
    # 段の境目（横線）を強調します
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges = cv2.filter2D(gray, -1, kernel)
    
    # 中央付近の3箇所をスキャンして精度を高める
    pw = edges.shape[1]
    scan_lines = [int(pw * 0.45), int(pw * 0.5), int(pw * 0.55)]
    
    counts = []
    for x in scan_lines:
        line = edges[:, x]
        count = 0
        in_peak = False
        for val in line:
            if val > threshold: # 境目を発見
                if not in_peak:
                    count += 1
                    in_peak = True
            else:
                in_peak = False
        counts.append(count)
    
    # 3箇所の平均を整数で出す
    final_ans = int(np.median(counts))

    # --- 結果表示 ---
    st.image(img, caption='スキャン完了', use_column_width=True)
    
    st.markdown(f"""
    <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h2 style="margin: 0; color: #1f77b4;">判定結果</h2>
        <span style="font-size: 80px; font-weight: bold; color: #1f77b4;">{final_ans}</span>
        <span style="font-size: 30px; color: #1f77b4;"> 段</span>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 うまく数えられない場合は、左側の「感度設定」を動かしてみてください。")