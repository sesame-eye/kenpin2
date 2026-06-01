import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.15", layout="centered")
st.title("🔴 点検出（Blob）カウンター")

# --- 設定データの初期化 ---
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 30, "黄": 25, "緑": 35, "青": 40, "茶": 30}

# --- サイドバー ---
st.sidebar.header("🎨 カラー/モード")
color_mode = st.sidebar.selectbox("テープの色", ["白", "黄", "緑", "青", "茶"])
true_count = st.sidebar.number_input("正解の段数を入力", min_value=1, value=20)
learn_button = st.sidebar.button(f"{color_mode}を学習/最適化")

# --- 写真入力 ---
option = st.radio("写真入力", ("カメラ撮影", "ライブラリ参照"), horizontal=True)
uploaded_file = st.camera_input("撮影") if option == "カメラ撮影" else st.file_uploader("写真を選択", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    proc_h = 1000
    img = cv2.resize(img, (int(w * (proc_h/h)), proc_h))
    display_img = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 表面の細かなざらざらを消し、段差の「影の溝」を強調する
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 黒い「溝（シャドウ）」を抽出するためのブラックハット処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    
    target_thresh = st.session_state.brain.get(color_mode, 30)

    def analyze_by_dots(thresh_val):
        # 影の濃さで二値化（点として孤立させる）
        _, thresh = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 中央の「スキャン縦帯」を限定（幅50ピクセルだけ）
        pw = thresh.shape[1]
        x_start = int(pw * 0.45)
        x_end = int(pw * 0.55)
        
        # 輪郭（点の塊）を抽出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dot_centers = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 中央の縦帯エリアに入っている「点」だけを採用
                if x_start <= cX <= x_end:
                    # あまりに小さすぎるノイズや、縦に長すぎるゴミは除外
                    _, _, cw, ch = cv2.boundingRect(c)
                    if 2 < ch < 40:
                        dot_centers.append((cX, cY))
                        
        # 縦座標（Y）でソート（下から上、または上から下へ綺麗に並べる）
        dot_centers = sorted(dot_centers, key=lambda x: x[1])
        
        # 近すぎる点（重複検知）をマージする
        filtered_dots = []
        if dot_centers:
            filtered_dots.append(dot_centers[0])
            for p in dot_centers[1:]:
                if p[1] - filtered_dots[-1][1] > 12: # 12ピクセル以上離れていれば独立した「段」とみなす
                    filtered_dots.append(p)
                    
        # 段数は「検出された点の数 + 1」（隙間の数 + 1 ＝ 段数）
        est_count = len(filtered_dots) + 1 if filtered_dots else 1
        
        return est_count, filtered_dots, x_start, x_end

    # 学習処理
    if learn_button:
        best_t = 30
        min_error = 999
        for t in range(10, 100, 2):
            est_c, _, _, _ = analyze_by_dots(t)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_t = t
        st.session_state.brain[color_mode] = best_s = best_t
        st.sidebar.success(f"最適化完了: 点検出閾値 {best_t}")
        st.rerun()

    # 判定実行
    final_ans, detected_dots, x_start, x_end = analyze_by_dots(target_thresh)

    # --- 可視化の描画 ---
    # スキャンエリア（中央の縦帯）を薄く表示
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, 1000), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.1, display_img, 0.9, 0, display_img)

    # 検知した「段差の点」に赤い丸をプロット
    for i, pt in enumerate(detected_dots):
        cv2.circle(display_img, pt, 8, (0, 0, 255), -1)
        # デバッグ用に番号を横に小さく描画
        cv2.putText(display_img, str(i+1), (pt[0] + 15, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 結果表示
    st.image(display_img, use_column_width=True, caption="点検出（Blob）モード実行中。赤丸：検知した段差の影")
    
    st.metric("点検出による判定結果", f"{final_ans} 段")
    st.caption(f"中央エリア内で検知した有効な影の数: {len(detected_dots)} 個")
    
    st.info("💡 写真を読み込ませた後、左の『学習/最適化』ボタンを押すと、赤丸がちょうど段差の隙間に1個ずつ配置されるように自動で感度が調整されます。")
