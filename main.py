import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="段数カウンター Ver.16", layout="centered")
st.title("📊 輝度プロファイル（波形解析）カウンター")

# --- 設定データの初期化 ---
# 閾値を「近接する段差の最小距離（ピクセル数）」に変更します
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 25, "黄": 25, "緑": 25, "青": 25, "茶": 25}

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
    
    # 中央の「スキャン縦帯」の範囲を定義
    pw = gray.shape[1]
    x_start = int(pw * 0.45)
    x_end = int(pw * 0.55)
    
    # スキャンエリアを横方向に平均化して「1次元の縦の輝度配列(長さ1000)」にする
    # これにより、テープの表面のざらざらノイズが完全に消えます
    scan_line = gray[:, x_start:x_end]
    profile = np.mean(scan_line, axis=1)
    
    # 縦方向にも少し平滑化（ガウシアンフィルタの代わり）
    profile_smoothed = np.convolve(profile, np.ones(5)/5, mode='same')

    # 最適化対象：段差とみなす最小の間隔（ピクセル）
    target_distance = st.session_state.brain.get(color_mode, 25)

    def analyze_by_profile(min_dist):
        """輝度の『谷（周囲より暗い場所）』を検出する関数"""
        valleys = []
        # 1次元配列からローカルミニマ（谷）を探す
        # 左右の一定範囲(min_dist)の中で自分が最も暗い点を探す
        for y in range(min_dist, proc_h - min_dist):
            local_region = profile_smoothed[y - min_dist : y + min_dist + 1]
            if profile_smoothed[y] == np.min(local_region):
                # 同じ値が連続して重複検知されるのを防ぐ
                if not valleys or (y - valleys[-1]) > min_dist:
                    valleys.append(y)
        
        # 段数 ＝ 影（谷）の数 + 1
        est_count = len(valleys) + 1 if valleys else 1
        return est_count, valleys

    # 学習処理（正解の段数に一番近くなる最小間隔「min_dist」を自動探索）
    if learn_button:
        best_dist = 25
        min_error = 999
        for d in range(10, 60):
            est_c, _ = analyze_by_profile(d)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_dist = d
        st.session_state.brain[color_mode] = best_dist
        st.sidebar.success(f"最適化完了: 最小段差間隔 {best_dist} px")
        st.rerun()

    # 判定実行
    final_ans, detected_valleys = analyze_by_profile(target_distance)

    # --- 可視化の描画 ---
    # スキャンエリア（中央の縦帯）を薄く表示
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, proc_h), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.1, display_img, 0.9, 0, display_img)

    # 検知した「輝度の谷（段差）」に青い丸をプロット
    # 横座標はスキャンエリアの中心にする
    cX = int((x_start + x_end) / 2)
    for i, y_pos in enumerate(detected_valleys):
        cv2.circle(display_img, (cX, y_pos), 8, (255, 0, 0), -1)
        cv2.putText(display_img, str(i+1), (cX + 15, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 結果表示
    st.image(display_img, use_column_width=True, caption="波形解析モード実行中。青丸：検知した段差（輝度の谷）")
    
    st.metric("波形解析による判定結果", f"{final_ans} 段")
    st.caption(f"中央エリア内で検知した有効な段差の数: {len(detected_valleys)} 個")
    st.info("💡 うまく合わない場合は、左側の『学習/最適化』ボタンを押してください。画像の解像度に合わせて最適な段差の間隔（感度）を自動計算します。")
