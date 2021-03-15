from setuptools import setup, find_packages, Command
from shutil import rmtree
import os
import sys


# Package meta-data.
NAME = 'auto-ml-cl'
DESCRIPTION = 'Auto machine learning with scikit-learn and TensorFlow framework.'
URL = 'https://github.com/lugq1990/auto_ml'
EMAIL = 'gqianglu@outlook.com'
AUTHOR = 'guangqiang.lu'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.2'

REQUIRED = [
        'scikit-learn',
        'pyyaml',
        'tensorflow >= 2.1.0',
        'keras-tuner',
        'google-cloud-storage',
        'lightgbm',
        'xgboost',
        'flask',
        'flask_restful'
    ]

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, 'README.md'), encoding='utf-8') as f:
        README = '\n' + f.read()
except FileNotFoundError:
    README = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(path, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(path, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)



