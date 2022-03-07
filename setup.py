from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  
  name="wasar",
  version="0.0.1",
  author="Miguel González Jiménez & Carolina Guardiola Albert",
  author_email="miguigonn@gmail.com, email_carolina",
  description="Python package for analyzing DInSAR time series.",
  long_description=long_description,
  license="GNU GENERAL PUBLIC LICENSE Version 3",
  url="https://github.com/MiguelonGonzalez/wasar",
  install_requires=[
                  "geopandas>=0.10.2",
                  'pandas>=1.3.2',
                  'matplotlib>=3.4.2',
                  'folium'
                   ],
  packages=find_packages(),                 
  classifiers=[
              'Programming Language :: Python',
              'Programming Language :: Python :: 3.8',
              'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
              'Operating System :: Microsoft :: Windows',
              'Natural Language :: Spanish',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: GIS',
              'Topic :: Scientific/Engineering :: Hydrology'
              ]
      )
