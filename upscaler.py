"""
upscaler.py

Core inference engine for the Real-ESRGAN x4plus ONNX model.

The model has a FIXED input/output size (128x128 -> 512x512), so to upscale
an image of arbitrary size we:
  1. Pad the image so its dimensions are covered by an integer number of tiles.
  2. Slide a 128x128 window across the image with overlap (to avoid seams).
  3. Run each tile (optionally batched) through the ONNX model to get 512x512
     (4x) tiles.
  4. Blend overlapping output tiles using a feathered (raised-cosine) weight mask.
  5. Crop the stitched result back to (original_width*4, original_height*4).

Improvements over the baseline version:
  - Safe reflect-padding that works even when the image is smaller than the
    tile size (plain np.pad(mode="reflect") raises on small images).
  - Alpha channel is preserved (upscaled separately via bicubic resize) instead
    of being silently dropped by convert("RGB").
  - Raised-cosine feather window instead of a linear ramp, for smoother blends.
  - Larger default overlap (better seam suppression on detail-heavy content).
  - Optional batched tile inference for speed.
  - Tuned ONNX Runtime SessionOptions.
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from PIL import Image


class RealESRGANUpscaler:
    """Wraps the Real-ESRGAN x4plus ONNX model with tiling support."""

    TILE_SIZE = 128       # fixed model input size (square)
    SCALE = 4              # fixed model upscale factor
    DEFAULT_OVERLAP = 24    # pixels of overlap between neighboring tiles (input space)

    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
    ):
        """
        Args:
            model_path: path to the real_esrgan_x4plus.onnx file. The matching
                        real_esrgan_x4plus.data file must sit in the same folder.
            providers: ONNX Runtime execution providers, e.g. ["CUDAExecutionProvider",
                       "CPUExecutionProvider"]. Defaults to CPU only.
            intra_op_num_threads: threads for CPU inference. Defaults to os.cpu_count().
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = intra_op_num_threads or (os.cpu_count() or 4)

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Detect whether the model supports a dynamic batch dimension so we
        # know if batched inference is safe to use.
        input_shape = self.session.get_inputs()[0].shape
        self._supports_batching = isinstance(input_shape[0], str) or input_shape[0] is None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def upscale(
        self,
        image: Image.Image,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = 1,
        progress_callback=None,
    ) -> Image.Image:
        """
        Upscale a PIL image 4x using tiled inference.

        Args:
            image: input PIL Image (any mode). Alpha, if present, is preserved.
            overlap: overlap in pixels between adjacent input tiles. Larger
                     values reduce seam artifacts but increase compute time.
            batch_size: number of tiles to run per inference call. Only used
                        if the underlying ONNX model supports a dynamic batch
                        axis; otherwise falls back to 1 automatically.
            progress_callback: optional callable(done_tiles, total_tiles).

        Returns:
            A new PIL Image, 4x the width and height of the input. Mode
            matches the input (RGBA in, RGBA out; otherwise RGB).
        """
        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        rgba = image.convert("RGBA") if has_alpha else None
        rgb_image = image.convert("RGB")

        orig_w, orig_h = rgb_image.size

        overlap = max(0, min(overlap, self.TILE_SIZE - 1))
        stride = self.TILE_SIZE - overlap
        if not self._supports_batching:
            batch_size = 1
        batch_size = max(1, batch_size)

        # Pad the source image (reflect padding) so we have an integer number
        # of strides covering the whole image, with a full tile fitting at the edge.
        padded_w = self._padded_length(orig_w, stride)
        padded_h = self._padded_length(orig_h, stride)

        arr = np.asarray(rgb_image, dtype=np.float32) / 255.0  # H, W, 3 in [0,1]
        pad_bottom = padded_h - orig_h
        pad_right = padded_w - orig_w
        arr = self._safe_reflect_pad(arr, pad_bottom, pad_right)

        tile_positions = self._tile_positions(padded_w, padded_h, stride)
        total_tiles = len(tile_positions)

        out_h, out_w = padded_h * self.SCALE, padded_w * self.SCALE
        canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

        blend_mask = self._feather_mask(self.TILE_SIZE * self.SCALE, overlap * self.SCALE)

        done = 0
        for batch_start in range(0, total_tiles, batch_size):
            batch_positions = tile_positions[batch_start : batch_start + batch_size]
            tiles = [
                arr[y : y + self.TILE_SIZE, x : x + self.TILE_SIZE, :]
                for (x, y) in batch_positions
            ]
            output_tiles = self._run_tiles(tiles)

            for (x, y), output_tile in zip(batch_positions, output_tiles):
                oy, ox = y * self.SCALE, x * self.SCALE
                canvas[oy : oy + output_tile.shape[0], ox : ox + output_tile.shape[1], :] += (
                    output_tile * blend_mask
                )
                weight_sum[oy : oy + output_tile.shape[0], ox : ox + output_tile.shape[1], :] += (
                    blend_mask
                )
                done += 1

            if progress_callback is not None:
                progress_callback(done, total_tiles)

        weight_sum = np.clip(weight_sum, 1e-6, None)
        canvas = canvas / weight_sum

        # Crop off padding (scaled) to get exactly orig_size * SCALE
        canvas = canvas[: orig_h * self.SCALE, : orig_w * self.SCALE, :]
        canvas = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(canvas, mode="RGB")

        if has_alpha:
            # Alpha carries no high-frequency texture detail the SR model would
            # help with, so a plain high-quality resize is sufficient and much
            # cheaper than running it through the network.
            alpha = rgba.split()[-1]
            alpha_upscaled = alpha.resize(
                (orig_w * self.SCALE, orig_h * self.SCALE), resample=Image.BICUBIC
            )
            result = result.convert("RGBA")
            result.putalpha(alpha_upscaled)

        return result

    def count_tiles(self, image: Image.Image, overlap: int = DEFAULT_OVERLAP) -> int:
        """Return how many tiles `upscale` will need to process for this image."""
        w, h = image.size
        overlap = max(0, min(overlap, self.TILE_SIZE - 1))
        stride = self.TILE_SIZE - overlap
        padded_w = self._padded_length(w, stride)
        padded_h = self._padded_length(h, stride)
        return len(self._tile_positions(padded_w, padded_h, stride))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_tiles(self, tiles_hwc: list[np.ndarray]) -> list[np.ndarray]:
        """Run a batch of 128x128x3 [0,1] float tiles through the model.

        Returns a list of 512x512x3 [0,1] float arrays, one per input tile.
        Falls back gracefully to single-tile calls if the session does not
        support batching.
        """
        if not self._supports_batching and len(tiles_hwc) > 1:
            return [self._run_tiles([t])[0] for t in tiles_hwc]

        batch = np.stack(
            [np.transpose(t, (2, 0, 1)) for t in tiles_hwc]
        ).astype(np.float32)  # N,3,128,128
        result = self.session.run([self.output_name], {self.input_name: batch})[0]
        result = np.clip(result, 0.0, 1.0)  # N,3,512,512
        return [np.transpose(r, (1, 2, 0)) for r in result]  # each 512,512,3

    def _padded_length(self, length: int, stride: int) -> int:
        """Smallest length >= `length` such that a TILE_SIZE window fits with
        integer stride steps starting at 0."""
        if length <= self.TILE_SIZE:
            return self.TILE_SIZE
        n_strides = -(-(length - self.TILE_SIZE) // stride)  # ceil div
        return self.TILE_SIZE + n_strides * stride

    def _tile_positions(self, padded_w: int, padded_h: int, stride: int) -> list[tuple[int, int]]:
        xs = list(range(0, padded_w - self.TILE_SIZE + 1, stride))
        ys = list(range(0, padded_h - self.TILE_SIZE + 1, stride))
        if xs[-1] != padded_w - self.TILE_SIZE:
            xs.append(padded_w - self.TILE_SIZE)
        if ys[-1] != padded_h - self.TILE_SIZE:
            ys.append(padded_h - self.TILE_SIZE)
        return [(x, y) for y in ys for x in xs]

    @staticmethod
    def _safe_reflect_pad(arr: np.ndarray, pad_bottom: int, pad_right: int) -> np.ndarray:
        """np.pad(mode="reflect") requires each pad amount to be smaller than
        the corresponding dimension. For small images (e.g. upscaling a 50x50
        thumbnail against a 128px tile) a single reflect pad isn't enough, so
        pad iteratively in chunks that respect that constraint.
        """
        while pad_bottom > 0 or pad_right > 0:
            h, w = arr.shape[:2]
            pb = min(pad_bottom, max(h - 1, 0))
            pr = min(pad_right, max(w - 1, 0))
            if pb == 0 and pr == 0:
                # Degenerate case (h or w == 1): fall back to edge padding
                # for the remainder, since reflect is impossible.
                arr = np.pad(arr, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
                break
            arr = np.pad(arr, ((0, pb), (0, pr), (0, 0)), mode="reflect")
            pad_bottom -= pb
            pad_right -= pr
        return arr

    @staticmethod
    def _feather_mask(size: int, overlap_scaled: int) -> np.ndarray:
        """Raised-cosine (Hann-style) feather ramp on each edge, tiled to a 2D
        (size, size, 1) mask, so overlapping tiles blend smoothly instead of
        showing seams. A cosine ramp blends more smoothly than a linear one,
        especially on high-frequency detail like hair, foliage, or text.
        """
        ramp_len = max(1, overlap_scaled // 2)
        w = np.ones(size, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp_len, dtype=np.float32))
        w[:ramp_len] = ramp
        w[-ramp_len:] = ramp[::-1]
        mask_2d = np.outer(w, w)
        return mask_2d[:, :, None]