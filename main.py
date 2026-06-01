import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="段数カウンター Ver.16", layout="centered")
st.title("🔴 エッジ連動型ピクセルカウンター (Ver.16)")

# --- 設定データの初期化 ---
if 'brain' not in st.session_state:
    st.session_state.brain = {"白": 35, "黄": 30, "緑": 40, "青": 45, "茶": 35}

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
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    proc_h = 1000
    img_bgr = cv2.resize(img_bgr, (int(w * (proc_h/h)), proc_h))
    
    # ★【色のバグ修正】表示用の画像をBGRからRGBに正しく変換（これで青が黄色になりません）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display_img = img_rgb.copy()
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 虚空対策：あらかじめ画像全体の「強い横エッジ（段差）」のマップを作っておく
    edge_map = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.uint8(np.absolute(edge_map))

    target_sens = st.session_state.brain.get(color_mode, 35)

    def analyze_by_verified_dots(sens):
        pw = gray.shape[1]
        x_start = int(pw * 0.45)
        x_end = int(pw * 0.55)
        
        # 中央エリアの縦輝度プロファイル（影の谷を探す）
        roi = blurred[:, x_start:x_end]
        projection = np.mean(roi, axis=1)
        
        # 輝度の「谷（黒い隙間）」を検出するため反転
        inv_projection = 255 - projection
        
        # 波形解析から点の候補（段差の影）を抽出
        peaks, _ = find_peaks(inv_projection, height=sens, distance=12)
        
        verified_dots = []
        for p in peaks:
            # 安全のため、画像の上下端すぎるものは除外
            if p < 10 or p > 990:
                continue
                
            # ★【虚空誤検知対策】
            # 検出された点の周囲（左右のエリア）に、本物の「テープのエッジ」が一定以上存在するか確認
            edge_check_area = edge_map[p-3:p+3, x_start:x_end]
            edge_intensity = np.mean(edge_check_area)
            
            # 背景の黒ボード（虚空）や滑らかな段差ではない場所は、エッジ強度が極端に低くなるため除外
            if edge_intensity > 8: 
                # 中央帯の真ん中の座標に点を打つ
                verified_dots.append((x_start + (x_end - x_start) // 2, p))
        
        # 点の数から段数を算出（隙間の数 + 1）
        est_count = len(verified_dots) + 1 if verified_dots else 1
        return est_count, verified_dots, x_start, x_end

    # 学習処理
    if learn_button:
        best_s = 35
        min_error = 999
        for s in range(10, 120, 2):
            est_c, _, _, _ = analyze_by_verified_dots(s)
            error = abs(est_c - true_count)
            if error < min_error:
                min_error = error
                best_s = s
        st.session_state.brain[color_mode] = best_s
        st.sidebar.success(f"最適化完了: 感度 {best_s}")
        st.rerun()

    # 判定実行
    final_ans, verified_dots, x_start, x_end = analyze_by_verified_dots(target_sens)

    # --- 可視化の描画 ---
    # スキャンエリアを薄い緑のシースルーで表示
    overlay = display_img.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, 1000), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.08, display_img, 0.92, 0, display_img)

    # 検知した「本物の段差の点」に赤丸をプロット
    for i, pt in enumerate(verified_dots):
        cv2.circle(display_img, pt, 8, (255, 0, 0), -1) # RGB空間なので（255, 0, 0）が「赤」
        cv2.putText(display_img, str(i+1), (pt[0] + 15, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 結果表示
    st.image(display_img, use_column_width=True, caption="エッジ連動型ピクセルカウンター（赤丸：エッジ検証を通過した本物の段差）")
    
    st.metric("確定判定結果", f"{final_ans} 段")
    st.caption(f"背景ノイズを除外した、有効な段差の影の数: {len(verified_dots)} 個")
