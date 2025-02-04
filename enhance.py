import cv2
import numpy as np
import argparse
import os
from cv2 import dnn_superres

def load_model(model_choice, use_gpu, use_cpu_egpu):
    sr = dnn_superres.DnnSuperResImpl_create()
    model_file = 'EDSR_x2.pb' if model_choice == 'x2' else 'EDSR_x3.pb'
    sr.readModel(model_file)
    sr.setModel('edsr', 2 if model_choice == 'x2' else 3)
    
    if use_gpu:
        # Use NVIDIA GPU (requires CUDA)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif use_cpu_egpu:
        # Use Intel eGPU (via OpenCL) for acceleration
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    else:
        # Default to CPU processing
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return sr

def adjust_saturation(image, saturation_scale=1.02):
    # Convert image to HSV, adjust saturation, then convert back to BGR
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
    h, s, v = cv2.split(hsv)
    s *= saturation_scale
    s = np.clip(s, 0, 255)
    hsv_adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_adjusted.astype('uint8'), cv2.COLOR_HSV2BGR)


def adjust_contrast_brightness(image, alpha=1.02, beta=1):
    # alpha > 1 increases contrast, beta increases brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_gamma_correction(image, gamma=0.95):
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


# Enhance shadows using CLAHE on the L channel in LAB color space.
def enhance_shadows(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Enhance image details using OpenCV's detailEnhance.
def enhance_details(image):
    detail = cv2.detailEnhance(image, sigma_s=3, sigma_r=0.1)
    return cv2.addWeighted(image, 0.9, detail, 0.1, 0)


# Denoise image using fastNlMeansDenoisingColored.
def denoise_image(image):
    # Advanced denoising
    return cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)


# Sharpen image using a blending approach to avoid overly drawn edges.
def sharpen_image(image, blend=0.3):
    # Enhanced sharpening for better detail
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 1 - blend, sharpened, blend, 0)


# Enhance lighting using a multi-scale Retinex approach on the V channel.
def enhance_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    
    # Multi-scale lighting enhancement with softer transitions
    sigma_list = [3, 15, 30]
    result = np.zeros_like(v)
    
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(v, (0, 0), sigma)
        detail = v - blur
        result += detail
    
    # Softer blending
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = cv2.addWeighted(v, 0.9, result, 0.1, 0)
    
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)


# Enhance atmosphere using a bloom effect
def enhance_atmosphere(image):
    # Very subtle bloom to preserve details
    blur = cv2.GaussianBlur(image, (0, 0), 15)
    return cv2.addWeighted(image, 0.97, blur, 0.03, 0)


# Apply simple gray-world color correction.
def color_correction(image):
    result = image.copy().astype('float32')
    
    # Split channels
    b, g, r = cv2.split(result)
    
    # Very subtle color adjustments
    r *= 1.01  # Slightly enhance reds
    b *= 0.99  # Slightly reduce blues
    
    # Merge and clip
    result = cv2.merge([b, g, r])
    result = np.clip(result, 0, 255)
    
    # Subtle color temperature adjustment
    result = cv2.cvtColor(result.astype('uint8'), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    # Minimal warming
    a = cv2.add(a, 1)
    b = cv2.add(b, 1)
    result = cv2.merge([l, a, b])
    
    # Smooth the result
    result = cv2.GaussianBlur(result, (3,3), 0)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def soft_sharpen(image):
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    sharpened = cv2.addWeighted(image, 1.05, blurred, -0.05, 0)
    return cv2.addWeighted(image, 0.95, sharpened, 0.05, 0)


def enhance_image(image, sr):
    original = image.copy()
    print("Upscaling the image...")
    image = sr.upsample(image)
    original = cv2.resize(original, (image.shape[1], image.shape[0]))
    print("Applying denoising...")
    image = denoise_image(image)
    print("Enhancing lighting...")
    image = enhance_lighting(image)
    print("Adjusting contrast and brightness...")
    image = adjust_contrast_brightness(image, alpha=1.01, beta=0)
    print("Enhancing shadows...")
    image = enhance_shadows(image)
    print("Enhancing details...")
    image = enhance_details(image)
    print("Adjusting saturation...")
    image = adjust_saturation(image, saturation_scale=1.01)
    print("Applying color correction...")
    image = color_correction(image)
    print("Applying final refinements...")
    image = soft_sharpen(image)
    print("Blending with original for natural look...")
    result = cv2.addWeighted(image, 0.9, original, 0.1, 0)
    print("Applying final softening...")
    return cv2.GaussianBlur(result, (5,5), 0)


def process_file(input_file, output_file, sr):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not read image {input_file}")
        return False
    enhanced = enhance_image(image, sr)
    cv2.imwrite(output_file, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Lossless compression
    print(f"Enhanced image saved to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Enhance anime images to be more beautiful and cinematic")
    parser.add_argument('input_path', type=str, nargs='?', default="input", help="Path to the input image file or directory (default: input)")
    parser.add_argument('output_path', type=str, nargs='?', default="output", help="Path to save the enhanced image file or directory (default: output)")
    parser.add_argument('--model', type=str, choices=['x2', 'x3'], default='x2', help="Choose EDSR model for upscaling: x2 or x3 (default: x2)")
    parser.add_argument('--use_gpu', action='store_true', help="Use NVIDIA GPU via CUDA for processing if available")
    parser.add_argument('--use_cpu_egpu', action='store_true', help="Use Intel eGPU via OpenCL for acceleration if available")
    args = parser.parse_args()

    sr = load_model(args.model, args.use_gpu, args.use_cpu_egpu)

    if os.path.isfile(args.input_path):
        if os.path.isdir(args.output_path):
            output_file = os.path.join(args.output_path, os.path.basename(args.input_path))
        else:
            output_file = args.output_path
        process_file(args.input_path, output_file, sr)
    elif os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
                input_file = os.path.join(args.input_path, filename)
                output_file = os.path.join(args.output_path, filename)
                process_file(input_file, output_file, sr)
    else:
        print("Error: input path is neither a file nor a directory.")
        exit(1)


if __name__ == '__main__':
    main() 