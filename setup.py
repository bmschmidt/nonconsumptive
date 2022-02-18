from setuptools import setup

setup(name='nonconsumptive',
      version='0.1',
      description='Research, share, and reproduce digital libraries without sharing full text.',
      url='http://github.com/bmschmidt/nonconsumptive',
      author='Ben Schmidt',
      author_email='bmschmidt@gmail.com',
      license='MIT',
      packages=['nonconsumptive'],
      install_requires=[
          'pyarrow>=7.0.0',
          'polars>=0.13.1',
          'pysrp',
          "pyyaml",
          "bounter"
      ],
        entry_points={
        'console_scripts': [
            'nonconsumptive = nonconsumptive.commander:main'
        ],
    },
      zip_safe=False
)
