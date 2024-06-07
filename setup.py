#from setuptools import setup
from py2exe.build_exe import py2exe
from distutils.core import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='pyVisualStim',
      version='0.1',
      description='Code for visual stimulation',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1'
      ],
      url='https://github.com/Sebasto7/pyVisualStim',
      author='Sebastian Molina-Obando',
      author_email='sebastian.molina.obando@gmail.com',
      license='',
      packages=['twopstim'],
      install_requires=[
          'PsychoPy3',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      console = ["run.py"],
      options = {
                    "py2exe":{
                        "skip_archive": True,
                        "optimize": 2
                    }
                }
      )