from setuptools import setup

package_name = 'planning_module'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/planning_launch.py']),
        ('share/' + package_name + '/config', ['config/planning_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Planning and control logic for Carla driving',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planning_node = planning_module.planning_node:main',
        ],
    },
)
