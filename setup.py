from setuptools import setup, find_packages

setup(
  name = 'marge-pytorch',
  packages = find_packages(),
  version = '0.2.8',
  license='MIT',
  description = 'Marge - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/marge-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'pre-training'
  ],
  install_requires=[
    'einops>=0.3',
    'faiss-gpu',
    'numpy',
    'torch>=1.6',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)