from setuptools import setup, find_packages

setup(
    name='memoria',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,  # Include template and static files
    install_requires=[

        'Flask',
        'numpy',
        'torch',
        'transformers',
        'torchvision',
        'Pillow',
        'deepface',
        'opencv_python',
        'scikit_learn'
    ],
    entry_points={
        'console_scripts': [
            'start-your-app=app:main',  # Change 'main' to the appropriate function if needed
        ],
    },
)
