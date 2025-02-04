import cv2
import os
import argparse
from enhance import enhance_image, load_model

def process_video(input_video_path, output_video_path, sr):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return False

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames from {input_video_path}...")

    # Process each frame
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance the frame
        enhanced_frame = enhance_image(frame, sr)

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

        frame_number += 1
        print(f"Processed frame {frame_number}/{frame_count}")

    # Release resources
    cap.release()
    out.release()
    print(f"Enhanced video saved to {output_video_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Enhance anime videos to be more beautiful and cinematic")
    parser.add_argument('input_video_dir', type=str, help="Path to the input video directory")
    parser.add_argument('output_video_dir', type=str, help="Path to save the enhanced video files")
    parser.add_argument('--model', type=str, choices=['x2', 'x3'], default='x2', help="Choose EDSR model for upscaling: x2 or x3 (default: x2)")
    parser.add_argument('--use_gpu', action='store_true', help="Use NVIDIA GPU via CUDA for processing if available")
    parser.add_argument('--use_cpu_egpu', action='store_true', help="Use Intel eGPU via OpenCL for acceleration if available")
    args = parser.parse_args()

    # Load the model
    sr = load_model(args.model, args.use_gpu, args.use_cpu_egpu)

    # Ensure the output directory exists
    if not os.path.exists(args.output_video_dir):
        os.makedirs(args.output_video_dir, exist_ok=True)

    # Process each video in the input directory
    for filename in os.listdir(args.input_video_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_video_path = os.path.join(args.input_video_dir, filename)
            output_video_path = os.path.join(args.output_video_dir, filename)
            process_video(input_video_path, output_video_path, sr)

if __name__ == '__main__':
    main() 