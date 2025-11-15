# streamlit run this_file.py
import base64, io, importlib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit.elements import image as _st_image
import numpy as np
from PIL import Image, Image as _PILImage
import cv2
import sys
import torch

# ===== ページ設定（描画より前に） =====
st.set_page_config(page_title="画像→線画→イラスト (Pix2PixHD)", layout="wide")

# ===== Pix2PixHD (教材版) =====
from options.test_options import TestOptions
from models.models import create_model

# ---------- 表示サイズ ----------
W_ORIGINAL = 420
W_RESULT   = 320
W_LABEL    = 128
W_STEP     = 140
CANVAS_W = CANVAS_H = 256

# ---------- ランタイム ----------
USE_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0); torch.manual_seed(0)
if not torch.cuda.is_available():
    sys.argv += ['--gpu_ids', '-1']

IMG_SIZE = 64

# ---------- Pix2PixHD 準備 ----------
opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.name = 'line2illust'
opt.which_epoch = 'chapt07-model.pth'
opt.label_nc = 2
opt.no_instance = True

model = create_model(opt).to(USE_DEVICE)
model.eval()

# ---------- UTILS ----------
def pil_to_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _image_to_url_compat(image_data,
                         width=None, clamp=False, channels="RGB",
                         output_format="PNG", layout_config=None, *args, **kwargs):
    # PIL.Image へ正規化
    if isinstance(image_data, np.ndarray):
        arr = image_data
        if arr.dtype != np.uint8:
            if arr.size and arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L").convert("RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil = Image.fromarray(arr, mode="RGB").convert("RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
    elif isinstance(image_data, Image.Image):
        pil = image_data.convert("RGBA")
    else:
        raise ValueError(f"Unsupported type: {type(image_data)}")

    # ★ 常に PNG(RGBA) で data URL を返す
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _patch_streamlit_image_to_url():
    """
    実際に参照される可能性のある3モジュールすべてに image_to_url をインストール。
    """
    for modname in (
        "streamlit.elements.image",
        "streamlit.elements.lib.image",
        "streamlit.elements.lib.image_utils",  # ← ここ重要
    ):
        try:
            m = importlib.import_module(modname)
            setattr(m, "image_to_url", _image_to_url_compat)
        except Exception:
            pass

# st_canvas を import した後にも念のため再パッチ
_patch_streamlit_image_to_url()


def letterbox_gray(img_gray: np.ndarray, dst_size=(CANVAS_W, CANVAS_H), bg=255) -> np.ndarray:
    """アスペクト比維持で正方形に貼り付け（余白=白）。"""
    H, W = img_gray.shape[:2]
    dw, dh = dst_size
    s = min(dw / W, dh / H)
    nw, nh = max(1, int(W*s)), max(1, int(H*s))
    resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.full((dh, dw), bg, np.uint8)
    x0, y0 = (dw - nw)//2, (dh - nh)//2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out

def image_to_line_gray(bgr: np.ndarray, th1: int, th2: int):
    """BGR→Gray→Canny→反転(白地黒線)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, th1, th2)      # 白=線
    inv   = cv2.bitwise_not(edges)         # 白=地/黒=線
    return gray, edges, inv

def build_binary_from_gray(gray256: np.ndarray, thresh: int, dilate_px: int, min_area: int) -> np.ndarray:
    """
    256x256の白地黒線→二値化(線=1)→膨張→小領域除去 を行った「線=1/地=0」の配列を返す。
    """
    # 線(黒=0)を拾う: 0..255 のうち thresh 未満を線とみなす
    bin_img = (gray256 < thresh).astype(np.uint8)

    # 線を太らせる（膨張）
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        bin_img = cv2.dilate(bin_img, k, iterations=1)

    # ノイズ除去（最小面積）
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        keep = np.zeros_like(bin_img)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        bin_img = keep

    return bin_img

def to_label64_from_binary(bin256: np.ndarray, or_pool: bool = True) -> np.ndarray:
    """
    256x256 (0/1) → 64x64 (0/1)
      or_pool=True: 4x4ブロックで OR プーリング（線が消えにくい）
      or_pool=False: 最近傍リサイズ
    """
    if or_pool:
        h, w = bin256.shape
        assert h == 256 and w == 256
        a = bin256.reshape(64, 4, 64, 4)
        pooled = a.max(axis=(1, 3))  # OR と同等（0/1なのでmaxでOK）
        return pooled.astype(np.int64)
    else:
        small = cv2.resize(bin256, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        return small.astype(np.int64)

@torch.no_grad()
def inference_from_label(label64: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(label64.reshape(1,1,IMG_SIZE,IMG_SIZE)).to(USE_DEVICE)
    y = model.inference(x, torch.tensor(0))
    out = y.detach().cpu().numpy()[0].transpose(1,2,0)
    out = (np.clip(out, 0, 1)*255).astype(np.uint8)
    return out

# ---------- UI ----------
st.title("画像から線画を生成しイラスト化")
uploaded = st.sidebar.file_uploader("画像をアップロード", type=["png","jpg","jpeg"])
photo = st.sidebar.camera_input("カメラで撮影（任意）")

st.sidebar.markdown("---")
st.sidebar.header("設定")
th1 = st.sidebar.slider("Canny Edge 1 (閾値1)", 0, 255, 100, 1)
th2 = st.sidebar.slider("Canny Edge 2 (閾値2)", 0, 255, 200, 1)
drawing_mode = st.sidebar.selectbox("描画ツール", ("freedraw","point","line","rect","circle","transform"))
stroke_width = st.sidebar.slider("ストロークの幅", 1, 25, 3)
stroke_color = st.sidebar.color_picker("ストローク色", "#000000")

bin_thresh = st.sidebar.slider("二値化しきい値(線=黒を拾う)", 0, 255, 180, 1)
dilate_px  = st.sidebar.slider("線を太らせる(dilate px)", 0, 5, 2, 1)
min_area   = st.sidebar.slider("ノイズ除去(最小面積px)", 0, 500, 80, 10)
or_pool    = st.sidebar.checkbox("64×64はORプーリングで縮小", value=True)
auto_generate = st.sidebar.checkbox("設定変更で自動生成", value=True)
draw_threshold = st.sidebar.slider("手描きのしきい値(α>0領域)", 0, 255, 128, 1)
show_steps = st.sidebar.checkbox("処理ステップのサムネを表示", value=False)

# 入力画像
src_img = None
if uploaded is not None:
    src_img = Image.open(uploaded).convert("RGB")
elif photo is not None:
    src_img = Image.open(photo).convert("RGB")

if src_img is None:
    st.info("左のサイドバーから画像をアップロードするか、カメラで撮影してください。")
    st.stop()

st.image(src_img, caption="元画像", width=W_ORIGINAL)

bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
gray, edges, inv = image_to_line_gray(bgr, th1, th2)

if show_steps:
    c1,c2,c3 = st.columns(3)
    with c1: st.image(gray,  caption="グレースケール", width=W_STEP)
    with c2: st.image(edges, caption="エッジ",         width=W_STEP)
    with c3: st.image(inv,   caption="白黒反転",       width=W_STEP)

# 256へレターボックス（白地黒線）
line_gray_256 = letterbox_gray(inv, (CANVAS_W, CANVAS_H))
bg_rgb = cv2.cvtColor(line_gray_256, cv2.COLOR_GRAY2RGB)

# ← 追加: 背景画像の内容に紐づく key を作る（変われば別インスタンスになる）
import hashlib
bg_hash = hashlib.md5(bg_rgb.tobytes()).hexdigest()[:8]
canvas_key = f"canvas-{bg_hash}"
bg_pil_rgba = Image.fromarray(bg_rgb).convert("RGBA")
fabric_bg = {
    "version": "5.3.0",  # 数字は任意でOK
    "objects": [{
        "type": "image",
        "left": 0, "top": 0,
        "width": CANVAS_W, "height": CANVAS_H,
        "opacity": 1,
        "selectable": False, "evented": False,  # 背景を掴めないように
        "src": pil_to_data_url(bg_pil_rgba),
    }]
}
canvas = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="rgba(0,0,0,0)",    # 完全透明
    # background_image は使わない
    initial_drawing=fabric_bg,           # ← これで背景を確実に表示
    height=CANVAS_H, width=CANVAS_W,
    drawing_mode=drawing_mode,
    key=canvas_key,
    update_streamlit=True,
)


# 手描き合成（透明は白）
if canvas.image_data is not None:
    rgba  = canvas.image_data.astype(np.uint8)
    alpha = rgba[...,3]
    if np.max(alpha) == 0:
        final_gray = line_gray_256.copy()
    else:
        draw_gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        draw_gray[alpha == 0] = 255
        draw_bin = np.where(draw_gray < draw_threshold, 0, 255).astype(np.uint8)
        final_gray = np.minimum(line_gray_256, draw_bin)  # 黒優先
else:
    final_gray = line_gray_256

st.image(final_gray, caption="合成線画（最終入力）", width=W_RESULT)

# ===== ラベル化（強化版） =====
bin256  = build_binary_from_gray(final_gray, bin_thresh, dilate_px, min_area)  # 0/1
label64 = to_label64_from_binary(bin256, or_pool=or_pool)                      # int64

st.image(label64*255, caption="ラベル(64x64)", width=W_LABEL, clamp=True)

# 推論
do_run = auto_generate or st.button("イラスト生成")
if do_run:
    try:
        out = inference_from_label(label64)
        out_resized = cv2.resize(out, (W_RESULT, W_RESULT), interpolation=cv2.INTER_LINEAR)
        st.image(out_resized, caption="生成結果", channels="RGB", width=W_RESULT)
    except Exception as e:
        st.error(f"生成時にエラーが発生しました: {e}")
