from setuptools import setup, find_packages

setup(
    name='visual-icl-3d',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for visual in-context learning with 3D perception.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'opencv-python',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'pyyaml',
        'tqdm',
        'Pillow',
        'scipy',
        'imageio',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)