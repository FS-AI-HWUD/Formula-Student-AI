from setuptools import setup

package_name = 'hydrakon_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hydrakon_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Unified launch for Carla vehicle + sensors + RViz2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={},
)