# Dinsar: Analysis of DInSAR time series.
![image](https://user-images.githubusercontent.com/75794654/156266031-0137cf89-f55f-4ce8-9695-688a7909c7af.png)

This package allows to analyze the ground deformation of a region and to compare it with other climatic variables, such as groundwater levels or precipitation. In addition, the inclusion of **wavelet** tools allows to analyze the main periodicities of the model variables and estimate cause-effect processes.

## Introduction

Many regions worlwide are affected by **ground subsidence phenomena**. Abusive water withdrawal from aquiferes is one the factors than can lead to this kind of processes. Although ground motion monitoring could be performed with in-situ instruments, one of the most widely used technique in the last decades are the **Differential Interferometry Synthetic Aperture Radar** (DInSAR). The DInSAR technique consists on the superposition of numerous SAR images of the same region of the Earth, thus obtaining an image with the ground motion occurring between the SAR images acquisitions. Analyzing multiple of those images of a given region and comparing it with the groundwater levels (or other variables) is the manin purpose of this program.

## Install

Released source packages are available on [PyPi](https://pypi.org/). You can **simply install** it as:

`pip install dinsar`

Since geopandas dependencies could cause conflicts with other spatial packages, it's highly recommended to create first a [new environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), as well as taking a look at the `geopandas` [installation guidelines](https://geopandas.org/en/stable/getting_started/install.html).

An optional package must be installed for the use of **wavelet** analysis. It's `rpy2` package, and it can be simply installed via ´pip´:

`pip install rpy2`

### Dependencies

- `geopandas`
- `pandas`
- `matplotlib`
- `folium`

## Contact

You can contact me directly via GitHub or via e-mail: miguigonn@gmail.com
    
## Get started

I have included several jupyter notebooks within 'example' folder in order to get you started with the code.

## Licence
This project is licensed under the terms of the GNU General Public License v3.0

## How to cite dinsar

## Example








