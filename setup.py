from distutils.core import setup

_dependencies = [
    'cycler==0.10.0',
    'kiwisolver==1.0.1',
    'matplotlib==3.0.3',
    'numpy==1.16.2',
    'pyparsing==2.4.0',
    'python-dateutil == 2.8.0',
    'scikit-learn==0.20.3',
    'scipy==1.2.1',
    'six==1.12.0',
    'termcolor==1.1.0',
    'tqdm==4.31.1',
    'ujson==1.35']

setup(
    name='Transcend',
    version='0.9',
    description='Transcend: A library for detecting '
                'concept drift using conformal evaluation.',
    maintainer='Feargus Pendlebury',
    maintainer_email='Feargus.Pendlebury[at]kcl.ac.uk',
    url='',
    packages=['transcend'],
    setup_requires=_dependencies,
    install_requires=_dependencies
)
