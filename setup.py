from setuptools import setup, find_packages

setup(
    name='auto-bia',
    author='Antoine A. Ruzette',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'scikit-image',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',

)
