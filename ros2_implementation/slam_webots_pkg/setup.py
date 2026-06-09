from setuptools import setup
import os
from glob import glob

package_name = 'slam_webots_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mmesoma',
    maintainer_email='amaraizummesoma@gmail.com',
    description='SLAM + Webots OPE package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['behaviour_policy = slam_webots_pkg.behaviour_policy:main',],
    },
)
