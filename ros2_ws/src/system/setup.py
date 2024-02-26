from setuptools import setup

package_name = 'system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anja',
    maintainer_email='anjakovacevvic@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_raw = system.webcam_pub:main ',
            'all_landmarks = system.landm_node2:main',
            'get_eyes = system.get_eyes:main',
            # 'eyes_viewer = system.eye_viewer:main',
            'face_det = system.roi_face:main',
            'gaze_capture = system.gaze_capture:main',
            'gui_node = system.gui:main',
        ],
    },
)
