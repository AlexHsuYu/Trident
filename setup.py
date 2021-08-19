from setuptools import setup, find_packages

setup(
    # package information
    name='trident',
    version='0.1.0',
    description='Smart Manufacturing Big Data Contest [group: pancake]',
    py_modules=['trident'],
    packages=find_packages(),

    # dependencies
    python_requires='>=3.5.*',
    install_requires=[
        'numpy>=1.14.5',
        'scipy',
        'pandas',
        'scikit-learn',
        'click'
    ],
    entry_points={
        'console_scripts': [
            'trident = trident.cli:main'
        ]
    }
)
