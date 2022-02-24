from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  
  name="dinsar",
  version="0.1",
  author="Miguel González Jiménez",
  author_email="miguigonn@gmail.com",
  description="Python package for analyzing D-InSAR time series.",
  long_description=long_description,
  #license="",
  #package_dir={"": "src"},
  url="https://github.com/MiguelonGonzalez/dinsar/tutorials",
  install_requires=[
                  "geopandas>=0.10.2",
                  'pandas>=1.3.2',
                  'matplotlib>=3.4.2',
                  'folium'
                   ],
  classifiers=[
              'Programming Language :: Python',
              'Programming Language :: Python :: 3.8',
              'License :: OSI Approved :: MIT License',
              'Operating System :: Microsoft :: Windows',
              'Natural Language :: Spanish',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: GIS',
              'Topic :: Scientific/Engineering :: Hydrology'
              ],
    package_dir={"": "dinsar"},
    python_requires=">=3.8",
    packages=find_packages(where="dinsar")

      )
