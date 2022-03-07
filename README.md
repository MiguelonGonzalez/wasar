# Wasar: Analysis of DInSAR time series.
![image](https://user-images.githubusercontent.com/75794654/157041747-26d4aad0-95d3-442e-a3f8-6dc1072b1185.png)

This package allows to analyze the ground deformation of a region and to compare it with other climatic variables, such as groundwater levels or precipitation. In addition, the inclusion of **wavelet** tools allows to analyze the main periodicities of the model variables and estimate cause-effect processes.

## Introduction

Many regions worldwide are affected by **ground subsidence phenomena**. Abusive water withdrawal from aquifers is one the main factors than can lead to this kind of processes. Although ground motion monitoring can be performed with in-situ instruments, one of the most widely used techniques in the last decades is the **Differential Interferometry Synthetic Aperture Radar** (DInSAR). The DInSAR technique consists on the superposition of numerous SAR images of the same region of the Earth, thus obtaining an image of the ground motion occurring between the SAR images acquisitions. Analyzing multiple time series of ground movement in a given region, and comparing them with groundwater level variatons (or other variables), are the manin purposes of the present program.

## Install

Released source packages are available on [PyPi](https://pypi.org/). You can **simply install** it as:

`pip install wasar`

Since geopandas dependencies could cause conflicts with other spatial packages, it's highly recommended to create first a [new environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), as well as taking a look at the `geopandas` [installation guidelines](https://geopandas.org/en/stable/getting_started/install.html).

An optional package must be installed for the use of **wavelet** analysis. It's `rpy2` package, and it can be simply installed via `pip`:

`pip install rpy2`

### Dependencies

- `geopandas`
- `pandas`
- `matplotlib`
- `folium`

## Contact

We are Miguel González Jiménez and Carolina Guardiola Albert. You can contact us just via GitHub or through our e-mails: miguigonn@gmail.com and c.guardiola@igme.es.
    
## Get started

In the *example* folder you can find several **tutorials** that will help you to get started with the program. Also, the functions, classes and modules are fully explained in Spanish, so if you have doubts about their behavior, just use the built-in `help`, the `?` mark or the tab button in Jupyter Notebook.

Example:    `help(wasar.Dataset.find_element)` or `wasar.Dataset.find_element?` or `wasar.Dataset.find_element` + `.` + *`press tab`*

## Licence
This project is licensed under the terms of the GNU General Public License v3.0

## How to cite wasar

## Example

    >>> import wasar
    >>> Model = wasar.example.get_model()

    >>> mymap = Model.mapa(LayerControl=False)
    >>> Model.get('Asc').mapa(m=mymap)
![map](https://user-images.githubusercontent.com/75794654/156733794-922a0bfe-e42b-4f4e-93fa-bf0cdcf71511.png)

A very useful tool of `wasar` are **wavelet tools**, which allow to perform frequency analysis of the time series.

The following example shows the **common periodicities** between a rainfall station and a piezometer, being the **annual** frequency the main common one.

    >>> from wasar import Wavelet
    >>> Doñana = wasar.example.get_model()

    >>> piezometer = Doñana.get('Piezo_bbdd').take('104080065')
    >>> piezometer = piezometer.pivot(index='Fechas',columns='Nombre', values='Valores')

    >>> rainfall = Doñana.get('P').take('Almonte')

    >>> Wavelet('M', piezometer, rainfall, dt=2, dj=1/20, lowerPeriod=2, upperPeriod=30)

![wavelet](https://user-images.githubusercontent.com/75794654/156804199-e8ec12db-75b8-4fce-8a47-e06a74044843.png)
