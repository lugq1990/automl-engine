import io
import shutil
from setuptools import setup, find_packages

# folder_to_be_deleted = ['build', 'automl_engine.egg-info']

# for folder in folder_to_be_deleted:
#     shutil.rmtree(folder)


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
    version='0.0.1',
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
    packages=['automl'],
    include_package_data=True,
    platforms='any',
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