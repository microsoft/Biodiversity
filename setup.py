from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
with open(HERE / 'README.md', encoding="utf8") as file:
        long_description = file.read()
VERSION = (HERE / 'version.txt').read_text().strip()

setup(
    name='PytorchWildlife',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.yml"]},
    url='https://github.com/microsoft/Biodiversity/',
    license='MIT',
    author='Andres Hernandez, Zhongqi Miao, Daniela Ruiz Lopez, Isai Daniel Chacon Silva',
    author_email='v-hernandres@microsoft.com, zhongqimiao@microsoft.com, v-druizlopez@microsoft.com, v-ichaconsil@microsoft.com',  
    description='a PyTorch Collaborative Deep Learning Framework for Conservation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'tqdm',
        'Pillow', 
        'supervision==0.23.0',
        'gradio',
        'ultralytics',
        'chardet',
        'wget',
        'yolov5',
        'setuptools',
        'scikit-learn',
        'timm',
        'omegaconf',
        'lightning',
        'setuptools==68.2.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='pytorch_wildlife, pytorch, wildlife, megadetector, conservation, animal, detection, classification',
    python_requires='>=3.8',
)
