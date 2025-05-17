from setuptools import setup

package_name = 'carla_vehicle_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vehicle_launch.py']),
        ('share/' + package_name + '/config', ['config/vehicle_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='ROS2 Carla integration package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vehicle_node = carla_vehicle_manager.vehicle_node:main',
            'imu_node = carla_vehicle_manager.imu_node:main',
        ],
    },
)
