import io
import os
from setuptools import setup, find_packages

__version__ = '0.0.6'


# for file in os.listdir("dist"):
#     if __version__ not in file:
#         os.remove(os.path.join('dist', file))
        

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')


setup(
    name='automl-engine',
    version=__version__,
    url='https://github.com/lugq1990/automl-engine',
    license='MIT License',
    author='guangqiang.lu',
    install_requires=[
        'pandas',
        'scikit-learn',
        'pyyaml',
        'tensorflow >= 2.1.0',
        'keras-tuner',
        'google-cloud-storage',
        'lightgbm',
        'xgboost',
        'flask',
        'flask_restful',
        'tqdm',
        'numpy'
    ],
    author_email='gqianglu@outlook.com',
    description='3 lines of code for automate machine learning for classification and regression.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    # package_data={'automl': ['*.yml']},
    platforms='any',
    # packages=['automl'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        ],
)