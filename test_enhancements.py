import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure model directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from lumaforge.pipeline import LumaForgePipeline

def run_tests():
    print("=== Testing LumaForge Pipeline Enhancements ===")
    
    # Initialize pipeline
    pipeline = LumaForgePipeline(device="cpu") # run on CPU for unit testing
    
    # Create a test input image (512x512 with a colored square in the center as the 'person/object')
    width, height = 512, 512
    img_arr = np.ones((height, width, 3), dtype=np.uint8) * 120 # gray background
    # Add a red circle in the center (foreground)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = (y - center_y)**2 + (x - center_x)**2 < 120**2
    img_arr[mask] = [220, 80, 80] # red foreground circle
    
    input_image = Image.fromarray(img_arr)
    
    # ---------------------------------------------
    # Test 1: Vectorized Background Removal (Chroma Key fallback)
    # ---------------------------------------------
    print("\n[Test 1] Testing background removal fallback...")
    t0 = time.time()
    cutout = pipeline.remove_background(input_image, mock=True)
    latency_bg = (time.time() - t0) * 1000
    
    print(f"-> Background removal completed in {latency_bg:.2f} ms")
    assert cutout.mode == "RGBA", f"Expected RGBA mode, got {cutout.mode}"
    assert cutout.size == (width, height), f"Expected size {(width, height)}, got {cutout.size}"
    
    # Verify background (corners) is transparent and foreground (center) is kept
    corner_pixel = cutout.getpixel((0, 0))
    center_pixel = cutout.getpixel((width // 2, height // 2))
    print(f"-> Corner pixel alpha: {corner_pixel[3]}")
    print(f"-> Center pixel alpha: {center_pixel[3]}")
    
    assert corner_pixel[3] == 0, "Expected background to be transparent"
    assert center_pixel[3] > 200, "Expected foreground to be mostly opaque"
    assert latency_bg < 100, f"Background removal was too slow: {latency_bg:.2f} ms (expected <100ms)"
    print("-> Test 1 PASSED!")
    
    # ---------------------------------------------
    # Test 2: Cartoon/Ghibli/Anime Shader
    # ---------------------------------------------
    print("\n[Test 2] Testing high-fidelity cartoon/Ghibli style shader...")
    t0 = time.time()
    cartoon_res = pipeline._generate_mock_img2img(input_image, "make it studio ghibli style anime", strength=0.6, seed=42)
    latency_cartoon = (time.time() - t0) * 1000
    
    print(f"-> Cartoon shader completed in {latency_cartoon:.2f} ms")
    assert cartoon_res.mode == "RGB", f"Expected RGB mode, got {cartoon_res.mode}"
    assert cartoon_res.size == (width, height), f"Expected size {(width, height)}, got {cartoon_res.size}"
    
    # Let's check that the output has cell shading (reduced distinct luminance values in Y channel)
    ycbcr = cartoon_res.convert("YCbCr")
    y_channel = np.array(ycbcr.split()[0])
    unique_y = np.unique(y_channel)
    print(f"-> Number of unique luminance values: {len(unique_y)}")
    
    # Let's check if colors are preserved
    # Center pixel should still be red-ish, corner pixel should still be gray-ish
    center_rgb = cartoon_res.getpixel((width // 2, height // 2))
    corner_rgb = cartoon_res.getpixel((0, 0))
    print(f"-> Center color (expect red accents): {center_rgb}")
    print(f"-> Corner color (expect gray accents): {corner_rgb}")
    assert center_rgb[0] > center_rgb[1] and center_rgb[0] > center_rgb[2], "Center should remain red dominant"
    print("-> Test 2 PASSED!")
    
    # ---------------------------------------------
    # Test 3: Vectorized Pencil Sketch
    # ---------------------------------------------
    print("\n[Test 3] Testing vectorized pencil sketch dodge-blend...")
    t0 = time.time()
    sketch_res = pipeline._generate_mock_img2img(input_image, "sketch pencil drawing of this picture", strength=0.7, seed=123)
    latency_sketch = (time.time() - t0) * 1000
    
    print(f"-> Sketch shader completed in {latency_sketch:.2f} ms")
    assert sketch_res.mode == "RGB", f"Expected RGB mode, got {sketch_res.mode}"
    assert sketch_res.size == (width, height), f"Expected size {(width, height)}, got {sketch_res.size}"
    
    # Sketch should be mostly white/black grayscale
    sketch_gray = sketch_res.convert("L")
    avg_intensity = np.mean(sketch_gray)
    print(f"-> Sketch average intensity: {avg_intensity:.2f} (typically bright background > 150)")
    assert avg_intensity > 150, "Pencil sketch background should be bright white"
    assert latency_sketch < 50, f"Sketch shader was too slow: {latency_sketch:.2f} ms (expected <50ms)"
    print("-> Test 3 PASSED!")

    # ---------------------------------------------
    # Test 4: Composited Background Replacement
    # ---------------------------------------------
    print("\n[Test 4] Testing composited background replacement...")
    t0 = time.time()
    bg_replace_res = pipeline._generate_mock_img2img(input_image, "replace background with neon city", strength=0.8, seed=999)
    latency_bg_rep = (time.time() - t0) * 1000
    
    print(f"-> Background replacement completed in {latency_bg_rep:.2f} ms")
    assert bg_replace_res.size == (width, height)
    
    # The center red foreground should be preserved because of background replacement cutout compositing!
    center_rgb = bg_replace_res.getpixel((width // 2, height // 2))
    print(f"-> Center color (expect red foreground kept): {center_rgb}")
    assert center_rgb[0] > center_rgb[1] and center_rgb[0] > center_rgb[2], "Center red object should be kept intact"
    print("-> Test 4 PASSED!")

    print("\nAll pipeline enhancement tests PASSED successfully!")

if __name__ == "__main__":
    run_tests()
