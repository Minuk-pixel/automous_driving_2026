#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract images from a ROS2 bag at a fixed time interval."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default="rosbag2_2026_06_11-15_13_39",
        help="Path to the rosbag2 directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="objectdetection",
        help="Directory where extracted images will be saved.",
    )
    parser.add_argument(
        "-t",
        "--topic",
        default="/image_raw",
        help="Image topic to extract.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.1,
        help="Capture interval in seconds.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=("jpg", "png"),
        help="Output image format.",
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="OpenCV/cv_bridge desired encoding. Use 'auto' or 'passthrough' for raw encodings like 8UC3.",
    )
    return parser.parse_args()


def make_reader(bag_path):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def topic_type_map(reader):
    return {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }


def image_to_cv2(bridge, msg, encoding):
    if encoding == "auto":
        try:
            return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


def extract_images(bag_path, output_dir, topic, interval_sec, image_format, encoding):
    import cv2
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_path = Path(bag_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if interval_sec <= 0.0:
        raise ValueError("--interval must be greater than 0.")

    reader = make_reader(bag_path)
    topics = topic_type_map(reader)
    if topic not in topics:
        available = ", ".join(sorted(topics))
        raise RuntimeError(f"Topic '{topic}' was not found. Available topics: {available}")

    msg_type = get_message(topics[topic])
    bridge = CvBridge()
    interval_ns = int(interval_sec * 1_000_000_000)
    next_capture_ns = None
    saved_count = 0
    first_timestamp_ns = None

    print(f"Bag: {bag_path}")
    print(f"Topic: {topic} ({topics[topic]})")
    print(f"Interval: {interval_sec:.3f} sec")
    print(f"Output: {output_dir}")

    while reader.has_next():
        read_topic, serialized_msg, timestamp_ns = reader.read_next()
        if read_topic != topic:
            continue

        if first_timestamp_ns is None:
            first_timestamp_ns = timestamp_ns
            next_capture_ns = timestamp_ns

        if timestamp_ns < next_capture_ns:
            continue

        msg = deserialize_message(serialized_msg, msg_type)
        cv_image = image_to_cv2(bridge, msg, encoding)

        elapsed_sec = (timestamp_ns - first_timestamp_ns) / 1_000_000_000.0
        filename = output_dir / f"image_{saved_count:06d}_{elapsed_sec:010.3f}s.{image_format}"
        ok = cv2.imwrite(str(filename), cv_image)
        if not ok:
            raise RuntimeError(f"Failed to write image: {filename}")

        saved_count += 1
        next_capture_ns += interval_ns
        while timestamp_ns >= next_capture_ns:
            next_capture_ns += interval_ns

    print(f"Saved {saved_count} images.")


def main():
    args = parse_args()
    extract_images(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        topic=args.topic,
        interval_sec=args.interval,
        image_format=args.image_format,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
