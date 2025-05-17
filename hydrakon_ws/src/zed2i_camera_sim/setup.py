from setuptools import setup

package_name = 'zed2i_camera_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/zed_launch.py']),
        ('share/' + package_name + '/config', ['config/zed_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Simulated ZED2i camera in Carla',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_node = zed2i_camera_sim.zed_node:main',
        ],
    },
)
