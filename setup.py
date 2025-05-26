from setuptools import setup, find_packages

setup(
    name='dnn_lib',
    version='0.1.0',
    description='A personal deep neural network library.',
    author='Jesse St. Amand', # Replace with your name
    author_email='JesseStAmand@gmail.com', # Replace with your email
    packages=find_packages(include=['dnn_lib', 'dnn_lib.*']),
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or another license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
) 