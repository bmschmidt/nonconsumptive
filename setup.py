from setuptools import setup

setup(name='nonconsumptive',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/bmschmidt/nonconsumptive',
      author='Ben Schmidt',
      author_email='bmschmidt@gmail.com',
      license='MIT',
      packages=['nonconsumptive'],
      install_requires=[
          'pyarrow',
          'polars'
      ],
      zip_safe=False)
