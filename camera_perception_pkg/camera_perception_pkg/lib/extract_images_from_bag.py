#!/usr/bin/env python3

import os
import argparse

import cv2
import rclpy
from rclpy.serialization import deserialize_message

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def extract_images(
    bag_dir: str,
    output_dir: str,
    topic_name: str = "/image_raw",
    interval_sec: float = 0.5,
    image_ext: str = "jpg",
):
    os.makedirs(output_dir, exist_ok=True)

    bridge = CvBridge()

    storage_options = StorageOptions(
        uri=bag_dir,
        storage_id="sqlite3"
    )

    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    saved_count = 0
    read_count = 0

    first_time_sec = None
    last_saved_time_sec = None

    print("========== ROS2 Bag Image Extractor ==========")
    print(f"Bag directory : {bag_dir}")
    print(f"Topic         : {topic_name}")
    print(f"Output dir    : {output_dir}")
    print(f"Interval      : {interval_sec} sec")
    print(f"Image format  : .{image_ext}")
    print("=============================================")

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic != topic_name:
            continue

        read_count += 1

        current_time_sec = timestamp_ns * 1e-9

        if first_time_sec is None:
            first_time_sec = current_time_sec

        relative_time_sec = current_time_sec - first_time_sec

        # 첫 프레임은 저장, 이후에는 interval_sec 이상 지난 프레임만 저장
        if last_saved_time_sec is not None:
            if current_time_sec - last_saved_time_sec < interval_sec:
                continue

        msg = deserialize_message(data, Image)

        try:
            # 핵심 수정:
            # 8UC3 같은 encoding은 bgr8로 바로 변환하지 말고 passthrough로 읽어야 함
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        except Exception as e:
            print(f"[WARN] Failed to convert image at {relative_time_sec:.3f}s")
            print(f"       encoding: {msg.encoding}")
            print(f"       error   : {e}")
            continue

        # numpy array 형태 점검
        if cv_image is None:
            print(f"[WARN] Empty image at {relative_time_sec:.3f}s")
            continue

        # 8UC3이면 보통 H x W x 3 형태
        # OpenCV imwrite는 3채널 이미지를 BGR로 간주해서 저장함.
        # 실제 색상이 뒤집혀 보이면 아래 COLOR_RGB2BGR 변환을 켜면 됨.
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            save_image = cv_image

            # 색상이 이상하게 저장되면 아래 줄 주석 해제
            # save_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        elif len(cv_image.shape) == 2:
            # grayscale 이미지
            save_image = cv_image

        else:
            print(
                f"[WARN] Unsupported image shape at {relative_time_sec:.3f}s: "
                f"{cv_image.shape}, encoding={msg.encoding}"
            )
            continue

        filename = f"image_{saved_count:06d}_t{relative_time_sec:08.3f}s.{image_ext}"
        save_path = os.path.join(output_dir, filename)

        success = cv2.imwrite(save_path, save_image)

        if not success:
            print(f"[WARN] Failed to save: {save_path}")
            continue

        print(
            f"[SAVE] {filename} | "
            f"encoding={msg.encoding}, shape={cv_image.shape}"
        )

        saved_count += 1
        last_saved_time_sec = current_time_sec

    print()
    print("=============== Result ===============")
    print(f"Read image messages : {read_count}")
    print(f"Saved images        : {saved_count}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract /image_raw images from ROS2 bag every 0.5 seconds."
    )

    parser.add_argument(
        "bag_dir",
        type=str,
        help="ROS2 bag directory path containing metadata.yaml and .db3 file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./extracted_image_raw",
        help="Directory to save extracted images"
    )

    parser.add_argument(
        "--topic",
        type=str,
        default="/image_raw",
        help="Image topic name"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Extraction interval in seconds"
    )

    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format"
    )

    args = parser.parse_args()

    rclpy.init()

    extract_images(
        bag_dir=args.bag_dir,
        output_dir=args.output,
        topic_name=args.topic,
        interval_sec=args.interval,
        image_ext=args.ext,
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()