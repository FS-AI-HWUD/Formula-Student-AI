from setuptools import setup
import os
from glob import glob
package_name = 'hydrakon_simulation'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),  # Added for RViz config
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joseph Abdo',
    maintainer_email='gravityfallsuae@gmail.com',
    description='Carla autonomous driving package with cone detection',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_camera_fusion = hydrakon_simulation.lidar_camera_fusion:main',  # Added the LiDAR fusion node
        ],
    },
)
