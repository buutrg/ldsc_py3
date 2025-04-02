from setuptools import setup

setup(name='ldsc',
      version='1.0.1',
      description='LD Score Regression (LDSC)',
      url='http://github.com/bulik/ldsc',
      author='Brendan Bulik-Sullivan and Hilary Finucane',
      author_email='',
      license='GPLv3',
      packages=['ldscore'],
      scripts=['ldsc.py', 'munge_sumstats.py'],
      install_requires = [
            'bitarray>=2.6.0,<2.7.0',
            'nose>=1.3.7,<1.4.0',
            'pybedtools>=0.9.0,<0.10.0',
            'scipy>=1.10.0,<1.11.0',
            'numpy>=1.24.0,<1.25.0',
            'pandas>=1.5.0,<1.6.0'
      ],
      python_requires='>=3.6'
)
