"""
app.py

Streamlit frontend for the Real-ESRGAN x4 image upscaler.

Run with:
    streamlit run app.py
"""

import io
import os
import time

import onnxruntime as ort
import streamlit as st
from PIL import Image, UnidentifiedImageError

from upscaler import RealESRGANUpscaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "real_esrgan_x4plus.onnx")
MAX_INPUT_DIM = 1024  # safety cap so CPU inference doesn't take forever in the UI
SLOW_TILE_WARNING_THRESHOLD = 40  # warn if a run would need more tiles than this on CPU

st.set_page_config(page_title="Real-ESRGAN Upscaler", page_icon="✨", layout="centered")


# ---------------------------------------------------------------------- #
# Model loading
# ---------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_model(providers: tuple[str, ...]) -> RealESRGANUpscaler:
    return RealESRGANUpscaler(MODEL_PATH, providers=list(providers))


def available_providers() -> list[str]:
    return ort.get_available_providers()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def image_to_bytes(image: Image.Image, fmt: str, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG" and image.mode == "RGBA":
        # JPEG has no alpha channel; flatten onto white first.
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        background.save(buf, format="JPEG", quality=quality)
    else:
        image.save(buf, format=fmt, quality=quality) if fmt == "JPEG" else image.save(buf, format=fmt)
    return buf.getvalue()


def reset_result():
    st.session_state.pop("result", None)
    st.session_state.pop("result_meta", None)


def st_image_full_width(image: Image.Image, caption: str = None):
    """st.image wrapper that works across Streamlit versions: newer versions
    use `use_container_width`, older ones (pre-1.29) only support
    `use_column_width`. Avoids a hard TypeError crash on older installs.
    """
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    st.title("✨ Image Upscaler (Real-ESRGAN x4)")
    st.write(
        "Upload an image and this will upscale it **4x** using the Real-ESRGAN "
        "x4plus ONNX model, running fully locally."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`. Make sure "
            "`real_esrgan_x4plus.onnx` (and its matching `.data` file) are in "
            "the `model/` folder next to this app."
        )
        st.stop()

    providers = available_providers()

    with st.sidebar:
        st.header("Settings")

        overlap = st.slider(
            "Tile overlap (px)",
            min_value=0,
            max_value=64,
            value=24,
            step=4,
            help="Higher overlap reduces visible seams between tiles but is slower.",
        )

        batch_size = st.slider(
            "Tile batch size",
            min_value=1,
            max_value=8,
            value=4,
            help="Tiles processed per inference call. Higher can be faster but "
            "uses more memory. Ignored if the model doesn't support batching.",
        )

        provider_choice = st.selectbox(
            "Execution provider",
            options=providers,
            index=0,
            help="CUDAExecutionProvider (if available) runs on GPU and is much "
            "faster than CPU.",
        )

        st.divider()
        st.caption(
            "The model processes the image in 128x128 tiles internally "
            "(each tile becomes 512x512). Larger images and lower overlap "
            "settings both affect speed."
        )

    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"]
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        reset_result()
        return

    # Reset any previous result if a new file is uploaded.
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        st.session_state["uploaded_name"] = uploaded_file.name
        reset_result()

    try:
        image = Image.open(uploaded_file)
        image.load()
    except UnidentifiedImageError:
        st.error("Couldn't read that file as an image. Please try a different file.")
        st.stop()

    orig_w, orig_h = image.size
    file_size = human_size(uploaded_file.size)

    if max(orig_w, orig_h) > MAX_INPUT_DIM:
        st.warning(
            f"Image is {orig_w}x{orig_h}. For speed, images are capped at "
            f"{MAX_INPUT_DIM}px on the longest side; it will be downscaled first."
        )
        scale_factor = MAX_INPUT_DIM / max(orig_w, orig_h)
        image = image.resize(
            (max(1, int(orig_w * scale_factor)), max(1, int(orig_h * scale_factor))),
            Image.LANCZOS,
        )
        orig_w, orig_h = image.size

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st_image_full_width(image, caption=f"{orig_w} x {orig_h} · {file_size}")

    model = load_model(tuple([provider_choice]) if provider_choice else tuple(providers))
    total_tiles = model.count_tiles(image, overlap=overlap)
    st.caption(f"This will process **{total_tiles} tile(s)** at the current settings.")

    if provider_choice == "CPUExecutionProvider" and total_tiles > SLOW_TILE_WARNING_THRESHOLD:
        st.warning(
            f"{total_tiles} tiles on CPU may take a while. Consider lowering "
            "overlap, using a smaller image, or switching providers if a GPU "
            "option is available."
        )

    run_col, clear_col = st.columns([3, 1])
    run_clicked = run_col.button("Upscale 4x", type="primary", use_container_width=True)
    if clear_col.button("Clear result", use_container_width=True):
        reset_result()
        st.rerun()

    if run_clicked:
        progress_bar = st.progress(0.0, text=f"Processing tile 0 / {total_tiles}...")
        start = time.time()

        def on_progress(done, total):
            elapsed_so_far = time.time() - start
            rate = elapsed_so_far / done if done else 0
            remaining = rate * (total - done)
            progress_bar.progress(
                done / total,
                text=f"Processing tile {done} / {total}... (~{remaining:.0f}s left)",
            )

        try:
            result = model.upscale(
                image, overlap=overlap, batch_size=batch_size, progress_callback=on_progress
            )
        except Exception as exc:  # noqa: BLE001 - surface any inference failure to the user
            progress_bar.empty()
            st.error(f"Upscaling failed: {exc}")
            st.stop()

        elapsed = time.time() - start
        progress_bar.empty()

        st.session_state["result"] = result
        st.session_state["result_meta"] = {
            "elapsed": elapsed,
            "overlap": overlap,
            "batch_size": batch_size,
        }

    result = st.session_state.get("result")
    if result is not None:
        meta = st.session_state["result_meta"]
        st.success(f"Done in {meta['elapsed']:.1f}s (overlap={meta['overlap']}, batch={meta['batch_size']})")

        with col2:
            st.subheader("Upscaled (4x)")
            st_image_full_width(result, caption=f"{result.width} x {result.height}")

        st.divider()
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
        with dl_col1:
            fmt = st.selectbox("Format", ["PNG", "JPEG"], index=0)
        quality = 95
        with dl_col2:
            if fmt == "JPEG":
                quality = st.slider("Quality", 50, 100, 95)
        with dl_col3:
            st.write("")
            st.write("")
            data = image_to_bytes(result, fmt, quality)
            st.download_button(
                "Download upscaled image",
                data=data,
                file_name=f"upscaled.{fmt.lower()}",
                mime=f"image/{fmt.lower()}",
                use_container_width=True,
            )
        st.caption(f"Output file size: {human_size(len(data))}")


if __name__ == "__main__":
    main()