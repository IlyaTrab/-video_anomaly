import argparse
import os
import cv2
from video_processor import process_video, save_anomalies_to_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def parse_arguments():
    parser = argparse.ArgumentParser(description="Обнаружение аномалий в видеоданных")
    parser.add_argument("video_paths", nargs='+', help="Пути к видеофайлам для анализа")
    parser.add_argument("--output", choices=['console', 'file'], default='console', help="Режим вывода результатов (console/file)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Пороговое значение для обнаружения аномалий")
    return parser.parse_args()

def main():
    args = parse_arguments()
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    for video_path in args.video_paths:
        anomalies = process_video(video_path, threshold=args.threshold, background_subtractor=background_subtractor)
        if args.output == 'file':
            output_file = f"{video_path}_anomalies.txt"
            save_anomalies_to_file(anomalies, output_file)
        elif args.output == 'console':
            print(f"Аномалии обнаружены в следующих кадрах: {anomalies}")

if __name__ == "__main__":
    main()

