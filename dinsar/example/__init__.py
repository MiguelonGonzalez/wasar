"""

Módulo para el manejo del ejemplo del programa, relativo éste a una zona cercana
al Parque Natural de Doñana (España).

A continuación se describen los datos utilizados, siendo todos ellos de carácter
público:

. Base de datos de piezometría. Compuesta por la información asociada a los
	piezómetros '104080065', '104130038', '104170015', '104240124', '114150046' 
	y '104140056', obtenidos de la Confederación Hidrográfica del Guadalquivir (*)
	y la Base de Datos de Aguas del Instituto Geológico y Minero de
	España (IGME , CN-CSIC; **).

. Base de datos de pluviometría. Compuesta por la información asociada a las 
	estaciones de 'Almonte', 'La_Palma_del_Condado', 'Moguer' y 'Niebla', pertenecientes
	a la Junta de Andalucía (***).

. Los conjuntos de datos (data-sets) de deformación del terreno utilizados (
	('sentinel-asc' y 'sentinel-desc') han sido obtenidos y procesados en la
	plataforma GEP de la ESA (European Space Agency)


Los datos Copernicus Sentinel-1 se obtuvieron y procesaron en la plataforma GEP 
de la ESA en el marco del GEP Early Adopters Programme.
- Agradecemos al IGME y a la CHG por proporcionar los datos de piezometría, así 
como a la Junta de Andalucía por suministrar los datos meteorológicos.


"""

import os
import dinsar


_module_path = os.path.dirname(__file__)

__all__ = ["get_path", "get_model"]

def get_path(name):
	""" Devuelve la ruta de acceso a la parte indicada (name) del modelo de
	ejemplo.

	name: str
		Opciones: ['sentinel-asc', 'sentinel-desc', 'piezometria_bd',
				   'piezometria_shp', 'precipitacion_bd', 'precipitacion_shp',
				   'agregados']
	"""

	assert isinstance(name, str), 'Introduce un str.'
	name = name.lower()

	files = {'sentinel-asc':'shapefiles/Sentinel_Asc.shp',
	         'sentinel-desc':'shapefiles/Sentinel_Desc.shp',
	         'piezometria_bd':'bds/bd_piezo.txt',
	         'piezometria_shp':'shapefiles/Piezometros.shp',
	         'precipitacion_bd':'bds/bd_precipi.txt',
	         'precipitacion_shp':'shapefiles/Estaciones.shp',
	         'agregados':'shapefiles/Agregados.shp' }

	if name in files.keys():
		path = files[name]
		return os.path.abspath(os.path.join(_module_path, 'Data', path))

	else:
		msg = f"{name} no es válido. Opciones: {list(files.keys())}"
		raise ValueError(msg)

def get_model():
	"""Devuelve un modelo de ejemplo (dinsar.Model) con datos relativos a una 
	zona cercana al Parque Natural de Doñana, localizado en el sur de España.
	Para más información sobre el origen de los datos utilizados en este ejemplo,
	consulte la descripción de este módulo:
		>>> help(dinsar.example).
	"""

	Asc = dinsar.Dataset(get_path('Sentinel-Asc'),
						 'Asc', color='#99F550')
	Desc = dinsar.Dataset(get_path('Sentinel-Desc'),
						  'Desc', color='#FBA608')

	bbdd = dinsar.Piezometria(get_path('Piezometria_bd'), 'Piezo_bbdd', sep='\t')
	bbdd.append_geometry(get_path('Piezometria_shp'))

	precipi = dinsar.Precipitacion(get_path('Precipitacion_bd'), 'P', sep='\t')
	precipi.append_geometry(get_path('Precipitacion_shp'))

	Doñana = dinsar.Model(get_path('Agregados'))
	[Doñana.append(i) for i in [Asc, Desc, bbdd, precipi]]

	return Doñana
