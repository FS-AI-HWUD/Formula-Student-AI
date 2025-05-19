from setuptools import setup

package_name = 'lidar_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/lidar_launch.py']),
        ('share/' + package_name + '/config', ['config/lidar_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Handles LiDAR initialization in Carla',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_node = lidar_handler.lidar_node:main',
        ],
    },
)
