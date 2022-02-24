"""

Módulo para el manejo del ejemplo del programa, relativo éste a una porción
del Parque Natural de Doñana (España)

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
	"""Devuelve un modelo de ejemplo con datos relativos a una pequeña porción
	del Parque Natural de Doñana, localizado en el sur de España."""

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
