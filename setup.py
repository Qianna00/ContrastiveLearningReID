from setuptools import setup, find_packages


setup(name='ContrastiveLearningReID',
      version='1.0.0',
      description='Contrastive Learning for Unsupervised Object Re-Identification',
      author='Qian Zhang',
      author_email='zhq9669@gmail.com',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn==1.0', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
