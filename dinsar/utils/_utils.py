"""

Módulo privado que contiene funciones útiles para el resto del paquete.

"""
from functools import wraps
from warnings import warn


def _update_kwargs(own_kwargs, kwargs):
    """ Actualizar los **kwargs propios de wasar (own_kwargs) con los **kwargs
    introducidos por el usuario, haciendo primar éstos últimos.
    Así se logra evitar el error >>> TypeError: 'multiple values for argument'.
    """
    [own_kwargs.pop(i) for i in own_kwargs.copy() if i in kwargs.keys()]

    return dict(**own_kwargs, **kwargs)

def _check_geom(func):
    """ Decorador. Genera un error que indica que esa base de datos no tiene
    información espacial asociada."""

    @wraps(func)
    def inner_func(self):

        if self.has_geometry():
            return func(self)

        else:
            msg = (f"La base de datos '{self.name}' no tiene asociada información "
                    "espacial. Puedes añadirla a través del método 'append_geometry'.")
            raise SystemError(msg)

    return inner_func

def _filtro_agregados(args, agregados):
    """Comprueba que los agregados introducidos en la función Model.agregado
    o en la clase Lotes sean válidos."""

    if not all([isinstance(i, str) for i in args]):
        raise TypeError("Indica el nombre del Agregado como un str.")

    for i in args:
        if not i in agregados:
            raise ValueError(f"El agregado '{i}' no existe.")
            
def _check_color(color):
    """Comprueba que el color indicado es legible por Matplotlib y lo devuelve
    si es así."""    
    from matplotlib.colors import is_color_like
    
    if not is_color_like(color):
        raise ValueError(
            f"No es posible elegir el color '{color}', escoge uno válido.\n"
            "Más información: \n"
            "https://matplotlib.org/stable/tutorials/colors/colors.html" )

    else:
        return color

def _importing_folium(raise_warning=True):
    """Función para comprobar la existencia de la librería sin que se genere
    un error."""

    try:
        import folium
        return True

    except (ModuleNotFoundError, ImportError):
        if raise_warning:
            warn(
                "La librería folium no está instalada, por lo que no será posible "
                "visualizar los elementos espacial con un mapa de fondo cuando se "
                "utilice la función 'mapa'.\n Puedes instalarla de la siguiente "
                "forma: \n pip install folium ó \n conda install folium")
        return False

def _checking_folium(basemap):
    """Comprueba que exista la librería folium y, de no ser así, fija 'basemap'
    a False para que se utilice Matplotlib."""

    from wasar import _is_folium
    if not _is_folium and basemap:
        warn("'basemap' fijado a False. La librería folium no está instalada.\n"
             "La representación se realizará mediante matplotlib.")
        return False
    else:
        return basemap

def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
