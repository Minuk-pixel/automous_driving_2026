#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage


class LaneDetect(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # 커스텀 laneinfo 메시지 패키지가 없으므로 publish는 하지 않습니다.
        # 대신 self.lane_info 딕셔너리에 최신 차선 정보를 내부 변수로 저장합니다.
        self.lane_info = {
            'left_x': 0.0,
            'left_y': 0.0,
            'left_slope': 0.0,
            'right_x': 0.0,
            'right_y': 0.0,
            'right_slope': 0.0,
            'center_x': 0.0,
            'center_error': 0.0,
            'valid_left': False,
            'valid_right': False,
        }

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/usb_cam/image_raw/compressed',
            self.camera_callback,
            10
        )

        self.get_logger().info('lane_detection_node started.')
        self.get_logger().info('Subscribing: /usb_cam/image_raw/compressed')

    def camera_callback(self, msg: CompressedImage):
        try:
            # ROS2 CompressedImage -> OpenCV BGR image
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                self.get_logger().warn('Failed to decode compressed image.')
                return

            self.lane_info = self.process_image(img)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    def warpping(self, image):
        # 주의:
        # 아래 source 좌표는 기존 CARLA/샘플 이미지 기준입니다.
        # 실제 USB 카메라 해상도/시야각에 맞게 반드시 다시 튜닝해야 합니다.
        source = np.float32([
            [235, 305],
            [590, 305],
            [60, 400],
            [800, 400]
        ])

        destination = np.float32([
            [0, 0],
            [250, 0],
            [0, 460],
            [250, 460]
        ])

        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        bird_image = cv2.warpPerspective(image, transform_matrix, (250, 460))

        return bird_image

    def color_filter(self, image):
        lower = np.array([120, 120, 120])
        upper = np.array([255, 255, 255])

        white_mask = cv2.inRange(image, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)

        return masked

    def plothistogram(self, image):
        # 이미지 아래쪽 절반에서 x축 방향으로 흰색 픽셀 누적합 계산
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

        midpoint = int(histogram.shape[0] / 2)
        leftbase = int(np.argmax(histogram[:midpoint]))
        rightbase = int(np.argmax(histogram[midpoint:]) + midpoint)

        return leftbase, rightbase, histogram

    def slide_window_search(self, binary_warped, left_current, right_current):
        nwindows = 15
        window_height = int(binary_warped.shape[0] / nwindows)

        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        margin = 30
        minpix = 10

        left_lane = []
        right_lane = []

        # binary_warped가 이미 0/255 이미지라서 그대로 3채널화합니다.
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height
            win_y_high = binary_warped.shape[0] - w * window_height

            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin

            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            cv2.rectangle(
                out_img,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                out_img,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0),
                2
            )

            good_left = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_xleft_low) &
                (nonzero_x < win_xleft_high)
            ).nonzero()[0]

            good_right = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_xright_low) &
                (nonzero_x < win_xright_high)
            ).nonzero()[0]

            if len(good_left) > minpix:
                left_lane.append(good_left)
                left_current = int(np.mean(nonzero_x[good_left]))

            if len(good_right) > minpix:
                right_lane.append(good_right)
                right_current = int(np.mean(nonzero_x[good_right]))

        left_lane = np.concatenate(left_lane) if len(left_lane) > 0 else np.array([], dtype=np.int64)
        right_lane = np.concatenate(right_lane) if len(right_lane) > 0 else np.array([], dtype=np.int64)

        leftx = nonzero_x[left_lane] if len(left_lane) > 0 else np.array([])
        lefty = nonzero_y[left_lane] if len(left_lane) > 0 else np.array([])

        rightx = nonzero_x[right_lane] if len(right_lane) > 0 else np.array([])
        righty = nonzero_y[right_lane] if len(right_lane) > 0 else np.array([])

        valid_left = len(leftx) > 0 and len(lefty) > 0
        valid_right = len(rightx) > 0 and len(righty) > 0

        if valid_left:
            left_fit = np.polyfit(lefty, leftx, 1)    # x = a*y + b
        else:
            left_fit = np.array([0.0, 0.0])

        if valid_right:
            right_fit = np.polyfit(righty, rightx, 1)  # x = a*y + b
        else:
            right_fit = np.array([0.0, 0.0])

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]

        height, width = binary_warped.shape[:2]

        if valid_left:
            for i in range(len(ploty)):
                lx = int(left_fitx[i])
                y = int(ploty[i])
                if 0 <= lx < width:
                    cv2.circle(out_img, (lx, y), 1, (255, 255, 0), -1)

        if valid_right:
            for i in range(len(ploty)):
                rx = int(right_fitx[i])
                y = int(ploty[i])
                if 0 <= rx < width:
                    cv2.circle(out_img, (rx, y), 1, (255, 255, 0), -1)

        draw_info = {
            'left_fitx': left_fitx,
            'right_fitx': right_fitx,
            'ploty': ploty,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'valid_left': valid_left,
            'valid_right': valid_right,
        }

        return draw_info, out_img

    def process_image(self, img):
        # Step 1: BEV 변환
        warpped_img = self.warpping(img)

        # Step 2: Blurring을 통해 노이즈 제거
        blurred_img = cv2.GaussianBlur(warpped_img, (0, 0), 1)

        # Step 3: 색상 필터링 및 이진화
        filtered_img = self.color_filter(blurred_img)
        gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

        # Step 4: 히스토그램으로 차선 시작점 탐색
        left_base, right_base, hist = self.plothistogram(binary_img)

        # Step 5: 슬라이딩 윈도우 + 피팅
        draw_info, out_img = self.slide_window_search(binary_img, left_base, right_base)

        vehicle_center_x = 130.0
        y_ref = float(draw_info['ploty'][-1])

        lane_info = {
            'left_x': 0.0,
            'left_y': y_ref,
            'left_slope': 0.0,
            'right_x': 0.0,
            'right_y': y_ref,
            'right_slope': 0.0,
            'center_x': vehicle_center_x,
            'center_error': 0.0,
            'valid_left': bool(draw_info['valid_left']),
            'valid_right': bool(draw_info['valid_right']),
        }

        if draw_info['valid_left']:
            lane_info['left_x'] = float(vehicle_center_x - draw_info['left_fitx'][-1])
            lane_info['left_slope'] = float(np.arctan(draw_info['left_fit'][0]))

        if draw_info['valid_right']:
            lane_info['right_x'] = float(draw_info['right_fitx'][-1] - vehicle_center_x)
            lane_info['right_slope'] = float(np.arctan(draw_info['right_fit'][0]))

        # 중심선 계산
        if draw_info['valid_left'] and draw_info['valid_right']:
            left_bottom_x = float(draw_info['left_fitx'][-1])
            right_bottom_x = float(draw_info['right_fitx'][-1])
            center_x = (left_bottom_x + right_bottom_x) / 2.0

            lane_info['center_x'] = center_x
            lane_info['center_error'] = center_x - vehicle_center_x

        elif draw_info['valid_left']:
            # 왼쪽 차선만 있을 때는 차선 폭 가정이 필요합니다.
            # 여기서는 임시로 중심선 계산을 하지 않고 0으로 둡니다.
            pass

        elif draw_info['valid_right']:
            # 오른쪽 차선만 있을 때도 차선 폭 가정이 필요합니다.
            # 여기서는 임시로 중심선 계산을 하지 않고 0으로 둡니다.
            pass

        # 디버깅 출력
        self.get_logger().info(
            f"lane_info: "
            f"L(valid={lane_info['valid_left']}, x={lane_info['left_x']:.2f}, slope={lane_info['left_slope']:.3f}), "
            f"R(valid={lane_info['valid_right']}, x={lane_info['right_x']:.2f}, slope={lane_info['right_slope']:.3f}), "
            f"center_error={lane_info['center_error']:.2f}",
            throttle_duration_sec=1.0
        )

        # OpenCV 디버깅 창
        cv2.imshow('raw_img', img)
        cv2.imshow('warpped_img', warpped_img)
        cv2.imshow('filter_img', filtered_img)
        cv2.imshow('binary_img', binary_img)
        cv2.imshow('result_img', out_img)
        cv2.waitKey(1)

        return lane_info


def main(args=None):
    rclpy.init(args=args)

    node = LaneDetect()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
