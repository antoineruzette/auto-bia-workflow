from setuptools import setup, find_packages

setup(
    name='auto-bia',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',

)
