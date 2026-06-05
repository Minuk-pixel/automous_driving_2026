import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'decision_making_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hhk-laptop',
    maintainer_email='whaihong@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_planner_node = decision_making_pkg.motion_planner_node:main',
            'path_planner_node = decision_making_pkg.path_planner_node:main',
            'stanley_controller_node = decision_making_pkg.stanley_controller_node:main',
            'obstacle_mission_controller_node = decision_making_pkg.obstacle_mission_controller_node:main',
            'parking_control_node = decision_making_pkg.parking_control_node:main',
            'test_node = decision_making_pkg.sliding_window_lane_detect_ros2:main',
            'test_node_2 = decision_making_pkg.sliding_window_lane_detect_image_raw_ros2:main'

        ],
    },
)
