"""

Módulo que contiene la mayor parte de las herramientas sobre la que se apoya 
la reprsentación gráfica, tanto temporal como espacial, de wasar.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .utils._utils import _update_kwargs, _checking_folium, _importing_folium

if _importing_folium(raise_warning=False):
    import folium


# Bold letter
_b = lambda x: f"\033[1m{x}\033[0m"

def _colorbar(ax):
    """Creación y customización de un eje lateral (cax) que se utilizará como
    barra de color."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return cax

def random_color():
    """Devuelve un color aleatorio según la nomenclatura RGB"""
    return tuple(np.random.random() for i in range(3))

def _get_datasets_cmap(gdf):
    """Devuelve una paleta de colores con los colores de la columna 'Color'."""
    import matplotlib.colors as colors
    return colors.ListedColormap(gdf['Color'].unique().tolist())

def _wasar_plot_params(deactivate=False):
    """Activación de los parámetros por defecto de wasar en la representación 
    gráfica. Son el resultado de la customización de los parámetros por defecto 
    de Matplotlib (rcParams).

    Si pueden desactivar ejecutando la función con el argumento 'deactivate' 
    fijado a True.

    Devuelve
    --------
    None

    """

    if deactivate:
        mpl.rcdefaults()

    mpl.rcParams['figure.figsize'] = 15,8
    mpl.rcParams['lines.color'] = '#C24A0F'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.which'] = 'both' 

    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['xtick.labelsize'] = 15

    mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#C24A0F', '#ff7f0e',
                                      '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    mpl.rcParams['date.autoformatter.year'] = '%Y'
    mpl.rcParams['date.autoformatter.month'] = '%m-%Y'
    mpl.rcParams['date.autoformatter.day'] = '%d-%m-%Y'

    mpl.rcParams['image.cmap'] = 'Spectral'

def _add_basemap(gdf):
    """Añade un mapa base a través de Folium para la posterior adición de elementos
    espaciales de un geopandas.GeoDataFrame (gdf) a través del método 'explore'.
    
    Devuelve
    --------
    folium.Map
    """

    x1,y1,x2,y2 = gdf.geometry.to_crs(epsg=4326).total_bounds
    mymap = folium.Map(tiles='openstreetmap', control_scale=True)

    mymap.fit_bounds([[y1, x1], [y2, x2]])

    # Carga de algunos mapas base (obtenidos de '_some_tiles')
    for name, stuff in _some_tiles.items():
        tile, attr = stuff.values()
        try:
            folium.TileLayer(tile,
                            attr=attr,
                            name=name).add_to(mymap)
        except:
            pass

    return mymap

def _mapa(gdf, own_kwargs, user_kwargs, basemap, savefig=False):
    """
    Función privada para la representación espacial del gdf (GeoDataFrame) de
    entrada, usando folium (basemap=True) o no (False), guardando la figura
    (savefig=True) o no, y añadiendo los **kwargs que el usuario especifique,
    los cuales siempre primarán sobre los propios de wasar (own_kwargs).
    """

    _wasar_plot_params(deactivate=True)
    basemap = _checking_folium(basemap)
    kwargs = _update_kwargs(own_kwargs, user_kwargs)
    LayerControl = kwargs['LayerControl']
    kwargs.pop('LayerControl')

    # **************************************************************************
    # Representar con un mapa base a través de la librería folium:
    # **************************************************************************
    if basemap:
        if 'm' in kwargs.keys():
            mymap = kwargs['m']
            msg = f"'{mymap}' no es un objeto de tipo folium.Map"
            assert isinstance(mymap, folium.Map), msg

        else:
            mymap = _add_basemap(gdf)
            kwargs['m'] = mymap

        gdf.explore(**kwargs)

        if LayerControl:
            folium.LayerControl().add_to(mymap)

        if savefig:
            mymap.save("Mapa.html")

        _wasar_plot_params()

        return mymap


    # **************************************************************************
    # Representar a través de Matplotlib:
    # **************************************************************************
    else:

        # Adición de una barra de color cuando la columna a representar sea continua.
        # Para ello se requiere de un eje (ax), necesario para añadir cax.
        is_cat = ['object', 'category', 'bool'] # Pandas columns categorical dyptes
        if set(['legend', 'column']).issubset(kwargs.keys())\
                    and not gdf[kwargs['column']].dtype.name in is_cat:

            own_kwargs = dict(figsize=(10,10), cmap='Spectral')
            kwargs = _update_kwargs(own_kwargs, user_kwargs)

            if not 'ax' in kwargs:
                _, ax = plt.subplots(figsize=kwargs['figsize'])
                kwargs['ax'] = ax
                del _
            else:
                ax = kwargs['ax']

            kwargs['cax'] = _colorbar(ax)
            kwargs['cax'].set_title(kwargs['column'])

        ax = gdf.plot(**kwargs)
        ax.axis('off')

        if savefig:
            ax.figure.savefig(f"{self.name}.png")

        _wasar_plot_params()

        return ax

_some_tiles = {   
    'ESRI-Imagen':dict(
        tile='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imager\
y/MapServer/tile/{z}/{y}/{x}',
        attr="""Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AE\
X, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"""
                      ),
    'ESRI-Callejero':dict(
        tile='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street\
_Map/MapServer/tile/{z}/{y}/{x}',
        attr="""Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, \
Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand)\
, TomTom, 2012"""
                         ),
    'Stamen Terrain':dict(
        tile='Stamen Terrain',
        attr="""Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a \
href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map dat\
a &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> co\
ntributors""",
                         ),
    'Blank':dict(
        tile='',
        attr="""blank""",
                         )
              }
