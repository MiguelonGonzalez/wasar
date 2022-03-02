"""
Este  módulo contiene las partes que componen el modelo:
    _ Dataset
    _ Piezometria
    _ Precipitacion
    _ DataBase
    
Todas ellas derivan de la clase privada _Part, que contiene los atributos y
métodos comunes a todas ellas. A su vez y por la misma razón, las clases
Piezometría y Precipitación derivan de la clase _Bd.

"""
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import copy

from .plot import _mapa, random_color
from .utils._utils import _check_geom, _update_kwargs, _check_color


__all__ = ['Dataset', 'Piezometria', 'Precipitacion', 'DataBase']

class _Part:
    """ Clase base de las partes del modelo. """  

    def __init__(self, units=None):
        self._units = units
    
    @property
    def name(self):
        return self._name

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, name):
        assert isinstance(name, str), f"'{name}' no es un str."
        self._units = name
    
    @property
    def crs(self):
        """ Devuelve el SRC (Sistema de Referencia de Coordenadas) de la geometría
        si ésta existe y devuelve error si no es el caso."""
        return self.gdf.crs

    @property
    @_check_geom
    def gdf(self):
        """Devuelve el GeoDataFrame de la capa si ésta tiene asociada una
        geometría."""
        return self._gdf

    def has_geometry(self):
        """ Comprueba si se ha añadido un shapefile a la base de datos.
        Return bool
        """
        return hasattr(self, '_gdf')

    def take_point(self, point='all'):
        """ Este método selecciona los puntos indicados a través de un str o
        una lista de str, devolviendo. Devuelve un GeoDataFrame.
        
        Parámetros
        ----------
        point: str ó list. Default 'all' --> Coge todos los puntos existentes.
        
        Devuelve
        -------
        gdf: objeto de geopandas.GeoDataFrame 
        
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Doñana.get('Asc').take_point(['352866', '352918'])
        """
        
        if isinstance(point, str) and isinstance(self, _Bd):

            if point == 'all':
                gdf = self.gdf

            else:
                assert point in self.gdf['Nombre'].unique(), f"El punto {point} no existe."
                gdf = self.gdf.loc[self.gdf['Nombre']==point].copy()

        elif isinstance(point, list) and isinstance(self, _Bd):
            gdf = self.gdf.copy().set_index('Nombre').loc[point].reset_index()

        elif isinstance(point, str) and isinstance(self, Dataset):

            if point == 'all':
                gdf = self.gdf

            else:
                assert point in self.gdf['ID'].unique(), f"El punto {point} no existe."
                gdf = self.gdf.loc[self.gdf['ID']==point].copy()

        elif isinstance(point, list) and isinstance(self, Dataset):
            gdf = self.gdf.copy().set_index('ID').loc[point].reset_index()

        else:
            raise TypeError(f"El tipo de datos {type(point)} no es válido (solo str o list) ")

        return gdf
    
    def xy(self, punto):
        """ Devuelve las coordenadas XY del punto indicado, muy útil para
        combinar con el método 'find_element'.
        
        Parámetros
        ----------
        punto str:
            Nombre o ID del punto indicado. Esta variable se pasa al método 'take_point'
            para la obtención del punto.
            
        Devuelve
        --------
            objeto shapely.Point o shapely.MultiPoint
            
        Ejemplo
        -------
        # Obtención de los puntos del Dataset 'Ascending' presentes en un radio 
            de 15 Km a la redonda desde la estación pluviométrica de 'Almonte':
                
        >>> Doñana = dinsar.example.get_model()
        >>> xy = Doñana.get('P').xy('Almonte')
        >>> Doñana.get('Asc').find_element(xy, way='radius', radius=15)
            
        Observaciones
        --------
        Aunque en el cuarderno de Jupyter esta función parezca que devuelve
        un objeto de tipo NoneType, esto no es así. Puedes consultar los métodos
        disponibles del objeto saliente en la documentación de shapely:
            
        https://shapely.readthedocs.io/en/stable/manual.html
        
        """
        
        assert isinstance(punto, str), f"'{punto}' no es un str."
        
        return self.take_point(punto).geometry.iat[0] 
    
    def find_element(self, xy, way, orden=1, radius=10):
        """ Selección espacial de uno o varios elementos según la distancia 
        existente desde una ubicación exacta (xy).
        
        Parámetros
        ----------
        xy shapely.Point
            Contiene las coordenadas de la ubicación desde la que se realiza la
            selección.
        
        way str {'nearest', 'radius'}
            Forma de seleccionar el sensor.
            'nearest': Se ordenan los sensores según la cercanía al punto
                       indicado (xy) y se selecciona el deseado según
                       el grado de cercanía (orden).
            'radius': Se seleccionan todos los piezómetros dentro de el radio 
                      de acción determinado por 'radius'.
        
        orden int Default 1, opcional
            Orden de selección del piezómetro cuando way='nearest'
            Ejemplo: orden=1 (se selecciona el piezómetro más cercano)
                     orden=2 (se selecciona el segundo piezómetro más cercano)
                     orden=-1 (se selecciona el piezómetro más lejano)
                     
        radius int, Default 1, opcional
            Radio de acción (en Km) cuando way='radius'.
        
        Devuelve
        --------
        pandas.DataFrame
        
        Ejemplo
        -------
        
        >>> Doñana = dinsar.example.get_model()
        
        1. Selección del piezómetro más cercano a la estación 'Almonte':
        # Acceso a las coordenadas del punto (objeto de tipo shapely.Point)
        
        >>>> xy = Doñana.get('P').take_point('Almonte').geometry.iat[0]
        
        >>> Doñana.get('Piezo_bbdd').find_element(xy, way='nearest')
        
        2. Selección de todos los piezómetros en un área de 15 Km a la redonda:
        >>> Doñana.get('Piezo_bbdd').find_element(xy, 
                                                  way='radius', radius=15)
        """

        assert isinstance(way, str), f"La variable 'way' no es un str."

        gdf = self.gdf.copy()
        
        # Selección de los elementos del GeoDataFrame

        gdf['Dist'] = gdf.apply(lambda x: xy.distance(x.geometry), axis=1)
        name = 'ID' if isinstance(self, Dataset) else 'Nombre'
        if way == 'nearest':
            elements = gdf.sort_values('Dist', ascending=True).iloc[orden,:][name]
            
        elif way == 'radius':
            radius = radius*1000  # To meters.
            elements = gdf.loc[gdf['Dist']<radius].sort_values('Dist')[name]
            
            if elements.empty:
                raise ValueError(f"No se encontró ningún elemento en un radio "
                                 f"de {radius/1000} km. Prueba a especificar "
                                 "uno mayor")
                
        # Selección de las series temporales de los puntos seleccionados:
        if isinstance(self, Dataset):
            return self.df[elements].copy()
            
        else:
            df = self.df.set_index('Nombre').loc[elements].reset_index().copy()
            return df.pivot(index='Fechas', columns='Nombre', values='Valores')
                
    def mapa(self, point='all', basemap=True, savefig=False, LayerControl=True,
                   **kwargs):
        """ Función para representar espacialmente los puntos indicados mediante
        'point'. La representación se realiza a través de la librería Folium 
        (si basemap=True) o Matplotlib (basemap=False), guardándose la imagen 
        resultante (si savefig=True) como un archivo 'html' en el primer caso y 
        'png' en el segundo.
        Los puntos a representar (argumento 'point') se seleccionan a través del
        método 'take_point'.
        
        Parámetros
        ----------
        point str o list. Default 'all' 
            Puntos a representar. Por defecto se representan todos los puntos.
            Esta selección se realiza a través del método 'take_point'.
            
        basemap bool Default True, Opcional
            Adición de mapas base mediante Folium.
            
        savefig bool Default False, Opcional
            Guardado de la imagen en el directorio de trabajo.

        LayerControl: bool, Default True. Optional
            Añade una leyenda al mapa cuando éste es generado mediante folium
            (ie, basemap=True). Cuando se quieren añadir más entidades a un mapa
            ya creado (objeto folium.Map), este argumento debe ser False en el
            primero de ellos.

            Ejemplo:
            >>> m = Doñana.get('Asc').mapa(LayerControl=False)
            >>> Doñana.get('Piezo_bbdd').mapa(m=m)

        **kwargs: Opcional
            Argumentos adicionales. Se pasan a la función folium.Map o 
            geopandas.GeoDataFrame.plot() según el parámetro 'basemap'.
        
        Devuelve
        -------
        objeto folium.Map (si basemap=True) o matplotlib.axes.Axes (False)
        
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Doñana.get('P').mapa(['Moguer', 'Niebla'])
        
        """
        
        gdf = self.take_point(point=point)

        tooltip = 'ID' if isinstance(self, Dataset) else 'Nombre'

        if basemap:
            own_kwargs = dict(tooltip=tooltip, color=self.color, name=self.name,
                              style_kwds=dict(opacity=0.6, weight=1,
                                              fillOpacity=0.6),
                              marker_kwds=dict(radius=3))
        else:
            own_kwargs = dict(color=self.color, alpha=0.8, ec=None)

        own_kwargs['LayerControl'] = LayerControl

        return _mapa(gdf, own_kwargs, kwargs, basemap, savefig=savefig)

class Dataset(_Part):
    """
    Objeto Dataset que contiene una base de datos espacial de tipo puntal con 
    series temporales de deformación, típicamente obtenidas mediante técnicas
    D-InSAR (Differential SAR Interferometry).
    
    Se inicializa a partir de la ruta al archivo que contiene la información. 
    Aunque lo más común es que este archivo sea de tipo 'shapefile', cualquier
    formato es válido siempre y cuando sea leído por la función 
    'geopandas.read_file()'. Para más información:
    https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
    
    Si los datos no están albergados en un archivo espacial, como pueden ser
    aquellos con extensión 'csv' o 'txt', se puede realizar la conversión a 
    un fichero espacial a través de cualquier SIG (ej:QGIS, ArcGIS) o mediante
    la librería geopandas.
    
    La estructura que debe presentar este archivo es la siguiente:
        
    _FILAS: puntos de medición de la deformación.
    _COLUMNAS: contienen las series temporales y otras variables de interés.
        
        a) Obligatorio una columna 'ID' con la identificación del punto en 
           formato de texto (str). Esta columna se utilizará como clave primaria.

        b) Columnas que empiezan por D: momento temporal. Cada columna con este
           prefijo conforma un momento de la serie temporal de los puntos, siendo
           el formato de las fechas el siguiente: 'D%Y%m%d'
           
           Ej: D20141027 = día 27 del mes 01 (enero) del año 2014.
           ------------- NO se concibe otro formato de fechas -----------------
                    
        c) Se pueden incluir tantas otras columnas como se desee. Se podrán 
           utilizar para graduar los puntos en la visualización.
           
           Importante: la descomposición vertical de los puntos únicamente
           funciona cuando existe la columna 'cosU'.
           Ver más: dinsar.zoom.Subset.vm
    
    Este formato tiende a seguir el formato de los ficheros de salida del 
    procesado PSBAS (Parallel Small BAseline Subset) en la plataforma
    GeoHazards (*)
                  
    Parámetros
    ----------
    fname str
        Ruta absoluta o relativa al archivo de datos.
        
    name str
        Nombre con el que se bautiza al objeto.
        
    color Opcional
        Color con el que se representará este Dataset. Ha de ser válido por 
        la librería Matplotlib (**). Te puede resultar de gran ayuda esta (***)
        página web para seleccionar el color que desees.
            
    Devuelve
    -------
    Objeto dinsar.parts.Dataset
    
    Ejemplos
    --------
    >>> fname = dinsar.example.get_path('sentinel-asc')
    >>> Asc = dinsar.Dataset(fname, name='Asc', color='#99F550')   
    
    Referencias
    ----------

    (*)   https://terradue.github.io/doc-tep-geohazards-v2/tutorials/gep-sbas-s1.html
          #results-download-and-visualization
          
    (**)  https://matplotlib.org/stable/tutorials/colors/colors.html
    
    (***) https://htmlcolorcodes.com/es/
    
    """
    
    def __init__(self, fname, name, color=None, units='cm', **kwargs):

        super().__init__(units=units)

        self._name = name
        shape = gpd.read_file(fname, **kwargs)

        # # Validación del campo 'ID' y conversión a str:
        if shape.index.name == 'ID':
            shape.reset_index(inplace=True)
            shape['ID'] = shape['ID'].astype(str)

        elif 'ID' in shape.columns:
            shape['ID'] = shape['ID'].astype(str)

        else:
            raise TypeError("La capa no tiene una columna 'ID' con la "
                "identificación del punto o dicho campo no se llama 'ID'.")

        shape.set_index('ID', inplace=True)

        _gdf = shape[[i for i in shape.columns if i[0] != 'D']] # Índice en el campo 'ID'
        _gdf = _gdf.round(3)
        _gdf[['Dataset', 'Color']] = name, color
        _gdf.reset_index(inplace=True)
        self._gdf = _gdf

        _df = shape[[i for i in shape.columns if i[0] == 'D']].transpose().reset_index()
        _df.rename(columns={'index':'Fechas'}, inplace=True)
        _df['Fechas'] = pd.to_datetime(_df['Fechas'], format='D%Y%m%d') # Incluyo la 'D' como parte del fmt
        _df.set_index('Fechas', inplace=True)
        self._df = _df

        if not color:
            self.color = random_color()
        
        else:
            self.color = _check_color(color)
            
        self._is_subset = False
    
    def __call__(self):
        return self.gdf
    
    @property
    def name(self):
        return super().name   
    
    @name.setter
    def name(self, name):

        assert isinstance(name, str), f"'{name}' no es un str."
        
        if self.is_subset():
            self._name = name
            
        else:
            raise AttributeError("No puedes cambiar el nombre")

    @property
    def puntos(self):
        """Lista con los puntos del dataset."""
        return self.gdf['ID'].unique().tolist()

    @property
    def df(self):
        """Devuelve un pandas.DataFrame con las series temporales de todos los 
        puntos del dataset. """
        return self._df

    @property
    def gdf(self):
        """Devuelve un geopandas.GeoDataFrame con los puntos del dataset."""
        return self._gdf
    
    def is_subset(self):
        """ Devuelve True si este objeto no representa el Dataset original, 
        habiéndose recortado a los puntos indicados en el método 'subset'."""
        return self._is_subset
    
    def subset(self, puntos, vm=False):
        """ Selección de uno o varios puntos del dataset. Genera una copia profunda
        del objeto (deep copy) con el dataset recortado a la selección. Todos 
        los métodos del objeto continuán funcionando correctamente. Para ver
        si se encuentra ante el dataset original o una parte 'subseteada' del
        mismo puedes utilizar el método 'is_subset'.

        Parámetros
        ----------
        puntos str or list of str
            Puntos a seleccionar del Dataset. Se pueden consultar los puntos
            disponibles mediante 'dataset.puntos'.
            
        vm bool default False
            Descomponer el movimiento observado a su componente vertical. Requiere
            de la columna 'cosU' en el GeoDataFrame: columna que contiene los valores
            del vector unitario LOS (Line Of Sigh) a lo largo de la dirección
            vertical.
             
        Devuelve
        -------
        dinsar.parts.Dataset
        
        Ejemplos
        --------
        >>> Asc = Doñana.get('Asc')     # Selección del Dataset 'Ascending'
        >>> puntos = ['352866', '352918', '353090', '353378']
        >>> subset = Asc.subset(puntos) # La variable subset es un objeto de
                                          tipo 'Dataset'
        >>> subset.plot()               # Representación de las series temporales
                                          de los puntos seleccionados.
        >>> subset.puntos               # Los puntos del Dataset se limitan a la selección.
        ['352866', '352918', '353090', '353378']
        
        >>> subset.is_subset()   # ¿es 'subset' un recorte del Dataset original?
        >>> True
        """

        puntos = self.puntos if isinstance(puntos, str) and puntos.lower()=='all' else puntos
        
        # Copia del objeto
        piece = copy.deepcopy(self)
        piece._is_subset = True
        
        # Selección espacial de los puntos
        piece._gdf = self.take_point(point=puntos)

        if isinstance(puntos, str) and puntos.lower() != 'all':
            puntos = [puntos]
        
        # Selección de las series temporales y descomposición vertical del
        # movimiento (si procede)
        assert isinstance(vm, bool), "El argumento 'vm' no es de tipo bool."
        piece._is_vm = vm
        piece._df = piece._vm(self.df[puntos].copy()) if vm else self.df[puntos].copy()

        return piece

    def mean(self):
        """Devuelve una serie temporal con la deformación promedio del dataset
        (o del subset si el método 'is_subset' devuelve True)."""
        ts = self.df.mean(axis=1)
        ts.name = self.name
        return ts
    
    def std(self):
        """Devuelve una serie temporal con la desviación estandar de los puntos
        del dataset (o del subset, si el método 'is_subset' devuelve True)."""
        ts = self.df.std(axis=1)
        ts.name = self.name
        return ts

    def _vm(self, df):
        """Método privado para el cálculo del movimiento vertical de todos los puntos"""
        return df/self.gdf.set_index('ID')['cosU']

    @property
    def vm(self):
        """ Devuelve True si los valores de deformación de las serires temporales 
        únicamente representan la componente vertical, y False si representan 
        los valores de deformación según la dirección del LOS (Line Of Sigh)."""
        return self._is_vm
 
    def plot(self, mean=False, std=True, savefig=False, **kwargs):
        """ Método para representar las series temporales de los puntos del 
        dataset. Se pueden representar todas (mean=False) o su promedio (mean=
        True). Si se realiza según esta última opción, se puede sombrear en la 
        gráfica (std=True) el área que abarca +- una desviación estándar de
        los puntos.
        
        Parámetros
        ----------
        mean bool Default False, Opcional
            Visualizar el promedio de la deformación. Si es False y el número
            de puntos a representar supera el valor de 50, salta un aviso.
            
        std bool Default True, Opcional
            Visualizar el área de +- 1 desviación estándar. Solo funciona si
            mean=True.
        
        savefig bool Default False, Opcional
            Guarda la gráfica resultante en el directorio de trabajo con el
            siguiente nombre:'Deformación - nombre.png''
            También se puede guardar accediendo a la función
            'matplotlib.pyplot.savefig', lo que permite especificar parámetros:
            >>> Asc.plot().figure.savefig(*args, **kwargs)
            
        **kwargs Opcional
            Se pasan a la función de representación de matplotlib a través de
            pandas: DataFrame.plot(**kwargs)
            S
            
        Devuelve
        --------
        matplotlib.axes.Axes
            
        Ejemplo
        -------
        >>> Asc = Doñana.get('Asc')
        >>> Asc.plot(mean=True, title='Ejemplo', color='red')
        """

        own_kwargs = dict(legend=True, title=f"{self.name} - puntos",
                          ylabel=f"Deformación ({self.units})")
        kwargs = _update_kwargs(own_kwargs, kwargs)

        if len(self.puntos) > 50 and not mean:
            msg = (f"\n{len(self.df.columns)} series a plotear... "
                  "Esto puede tardar varios minutos.")
            warnings.warn(msg)

        if mean:
            ax = self.mean().plot(label='Mean', **kwargs)
            if std:
                ax.fill_between(self.df.index, self.mean() + self.std(),
                    self.mean() - self.std(), color='#EFEFEF')

        else:
            ax = self.df.plot(**kwargs)

        ax.set_xlim(self.df.index[0], self.df.index[-1])
        ax.set_ylabel(kwargs['ylabel'], fontsize=15, labelpad = 28, rotation=270)
        ax.set_title(kwargs['title'], fontsize=24, pad=15)
        ax.set_xlabel(None)

        if savefig:
            ax.figure.savefig(f"Deformación - {self.name}.png")

        return ax

class _Bd(_Part):
    """
    Objeto que contiene una base de datos.
    La base de datos ha de estar contenida en un archivo de texto (csv o txt)
    o Excel y debe presentar la siguiente estructura:
        
    a) Los registros como filas y las variables como columnas.
    
    b) Han de figurar (exactamente con el mismo nombre) la siguientes columnas:
        
       . Nombre: str Columna con el nombre del sensor al que corresponde los registros. 
       . Fechas: str o date. Fechas del registro. El formato de esta columna es
         configurable.
       . Valores: int o floar. Valores de los registros.
       
      La presencia de otras columnas es posible, aunque éstas no se tendrán en
      cuenta.

    Se eliminan los registros duplicados (salvo el primero) si existen registros
    duplicados con la misma fecha para un mismo sensor.
    
    Es posible añadir información espacial a la base de datos a través del 
    método 'append_geometry'.

    """

    def __init__(self, fname, name, date_format=None, color=None,
                       units=None, **kwargs):

        super().__init__(units=units)

        _df = self._open_file(fname, **kwargs)
        columnas = ['Nombre','Fechas','Valores']
        
        # Comprobación de las columnas:
        if not set(columnas).issubset(_df.columns):
            raise KeyError("Alguna de las siguientes columnas no existe en la "
                  f"base de datos o su nombre está mál escrito: \n f{columnas}")
            
        _df = _df[['Nombre','Fechas','Valores']]
                           
        _df['Nombre'] = _df['Nombre'].astype(str)
        _df['Fechas'] = pd.to_datetime(_df['Fechas'], format=date_format) if\
                        _df['Fechas'].dtype.name == 'object' else _df['Fechas']

        _df['Valores'] = _df['Valores'].astype(float)

        _df.drop_duplicates(subset=['Nombre','Fechas'], keep='first', inplace=True)
        
        self._df = _df
        self._name = name
        self.color = color
        
    def __call__(self, elemento):
        return self.take(elemento)
    
    def _open_file(self, fname, **kwargs):
        """Método privado para leer la base de datos.
        Devuelve un pandas.DataFrame"""
        
        if isinstance(fname, str):
            func = pd.read_excel if fname.split('.')[-1] in ['xls','xlsx'] \
                                 else pd.read_csv
               
            try:
                return func(fname, **kwargs)
            except:
                print(f"Hay problemas con la lectura del fichero: '{fname}'"
                      "o con los **kwargs asociados.")
            raise

        elif isinstance(fname, pd.DataFrame):
            return fname

        else:
            raise TypeError(f"'{fname}' no es válido o bien se ha introducido "
                             "un objeto que no es un pandas.DataFrame.")

    def append_geometry(self, fname, **kwargs):
        """ Añade una geometría (shapefile) a la base de datos, conviertiéndola
        así en una 'geodatabase'. Es posible acceder a esta geometría a través
        del método 'gdf'.
        
        Parámetros
        ----------
        fname: str o geopandas.GeoDataFrame
            Ruta al shapefile (incluyendo la extensión) u objeto de tipo
            GeoDataFrame.
        
        **kwargs: Opcional
            Se pasarán a la función 'geopandas.read_file()', encargada de leer
            la capa indicada. Útiles solo cuando fname=str.
            
        Devuelve
        -------
        None
        
        Ejemplo
        -------
        >>> bd_fname = dinsar.example.get_path('piezometria_bd')
        >>> Piezo = dinsar.Piezometria(bd_fname, name='Piezo', sep='\t')
        >>> shp_fname = dinsar.example.get_path('piezometria_shp')
        >>> Piezo.append_geometry(shp_fname)

        """
        
        if isinstance(fname, str):
            self._gdf = gpd.read_file(fname, **kwargs)

        elif isinstance(fname, gpd.GeoDataFrame):
            self._gdf = fname

        else:
            raise TypeError(f"'{fname}' no es válido o bien se ha introducido "
                             "un objeto que no es un geopandas.GeoDataFrame.")


        print('Geometría añadida correctamente.')

    @property
    def df(self):
        """Devuelve un pandas.DataFrame con la base de datos."""
        return self._df

    @property
    def elementos(self):
        """Devuelve una lista con los elementos de la base de datos."""
        return self.df['Nombre'].unique().tolist()

    def take(self, elemento):
        """Selección de una o varias series temporales de los sensores
        de la base de datos.
        
        Parámetros
        ----------
        elemento str o list
            Nombre de los sensores a seleccionar.
        
        Devuelve
        -------
        pandas.DataFrame
        
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Doñana.get('Piezo_bbdd').take(['104080065', '104130038'])
        """

        if isinstance(elemento, str):

            if elemento.lower() == 'all':
                return self.df

            elif elemento in self.elementos:
                return self.df.set_index('Nombre').loc[elemento].reset_index().copy()

            else:
                raise ValueError(f"El {self._bd} {elemento} no existe.")

        elif isinstance(elemento, list):
            coinciden = all([i in self.df['Nombre'].unique() for i in elemento])
            assert coinciden, f"Alguno de los {self._sensor_name}s indicados no existen."
            return self.df.set_index('Nombre').loc[elemento].reset_index().copy()

        else:
            msg = f"El tipo de datos {type(elemento)} no es válido: (solo str o list)."
            raise TypeError(msg)

    def plot(self, points='all', savefig=False, **kwargs):
        """
        Representación de las series temporales de los sensores indicados.
        
        Parámetros
        ----------
        points str o list, Default 'all'. Opcional
            'all' --> Plotear todos los sensores.
            Este argumento se introduce al método 'take' para obtener las
            series temporales.
            
        savefig bool Default False. Opcional
            Guarda la gráfica resultante en el directorio de trabajo con el
            siguiente nombre:'Sensores.png'. 
            También se puede guardar accediendo a la función
            'matplotlib.pyplot.savefig', lo que permite especificar parámetros:
            >>> Piezo.plot('all').figure.savefig(*args, **kwargs)
            
        **kwargs Opcional
            Se pasan a la función de representación de matplotlib a través
            de pandas: DataFrame.plot(**kwargs).
            
        Devuelve
        --------
        matplotlib.axes.Axes
            
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Piezo = Doñana.get('Piezo_bbdd')
        >>> Piezo.plot()
        """

        name = f"{self._sensor_name}".capitalize()

        points = self.elementos if isinstance(points, str) \
                                and points.lower() == 'all' else points

        df = self.take(points).pivot(index='Fechas', columns='Nombre',
                                     values='Valores')

        own_kwargs = dict(legend=True, title=name)
        kwargs = _update_kwargs(own_kwargs, kwargs)
        
        ax = df.plot(**kwargs)

        ax.set_ylabel(f"{name} ({self.units})",
                        fontsize=15, labelpad = 28, rotation=270)
        ax.set_title(kwargs['title'], fontsize=24, pad=15)

        if savefig:
            ax.figure.savefig(f'{name}.png')

        return ax

class Precipitacion(_Bd):
    """
                        BASE DE DATOS DE PRECIPITACIÓN
                           
    Parámetros
    ----------
    fname str o pandas.DataFrame
        Ruta relativa o absoluta al archivo (si str) u objeto pandas.DataFrame.
        Los **kwargs y el resto de argumentos son utilizados solo si fname=str.
    
    name str 
        Nombre con el que se bautiza a la base de datos.
    
    date_format str (opcional) Default '%Y-%m-%d'
        Formato de las fechas.
        Nomenclatura: https://www.dataindependent.com/pandas/pandas-to-datetime/

    **kwargs: Argumentos opcionales que se pasan a la función pandas.read_excel
        o pandas.read_csv según la naturaleza del archivo.
        
    Devuelve
    --------
    dinsar.parts.Precipitacion
    
    Ejemplo
    -------
    >>> fname = dinsar.example.get_path('precipitacion_bd')
    >>> Preci = dinsar.Precipitacion(fname, 'P', sep='\t')
    
    """

    def __init__(self, *args,  **kwargs):

        kwargs = _update_kwargs(dict(color='#D1843C', units='mm'), kwargs)

        super().__init__(*args, **kwargs)

        self._bd_sensors = 'pluviómetros'


    __doc__ += _Bd.__doc__

    estaciones = _Bd.elementos

    def take(self, estacion, values='std'):
        """Selección de la serie temporal de una estación de la base de 
        datos.
        
        Parámetros
        ----------
        estacion str
            Nombre de la estación a elegir.
            
        values str, Default 'std'
            La función devuelve los valores originales de precipitación ('original')
            o aquellos resultantes de la desviación estándar acumulada ('std')
            
        Devuelve
        -------
        pandas.Series
        
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Doñana.get('P').take('Almonte')
        
        """

        if not isinstance(estacion, str):
            raise TypeError(f"{type(estacion)} no es válido. Solo str.")

        assert estacion in self.estaciones, f"La estación '{estacion}' no existe."

        df = self.df.set_index('Nombre').loc[estacion].reset_index().copy()
        ts = df.pivot(index='Fechas',columns='Nombre', values='Valores').squeeze()

        if values == 'std':
            return np.cumsum(ts - ts.mean())

        elif values == 'original':
            return ts

        else:
            raise ValueError(f"{values} no coincide con 'original' o 'std'.")

    def plot(self, estaciones='all', savefig=False, **kwargs):
        """
        Representación de las series temporales de las estaciones indicadas.
        
        Parámetros
        ----------
        estaciones str o list
            'all' --> Plotear todos las estaciones.
            Este argumento se introduce al método 'take' para obtener las
            series temporales.
            
        savefig bool Default False, opcional
            Guarda la gráfica resultante en el directorio de trabajo con el
            siguiente nombre:'Estaciones.png'. 
            También se puede guardar accediendo a la función
            'matplotlib.pyplot.savefig', lo que permite especificar parámetros:
            >>> Precipi.plot('all').figure.savefig(*args, **kwargs)
            
        **kwargs Opcional
            Se pasan a la función de representación de matplotlib a través
            de pandas: DataFrame.plot(**kwargs)
            
        Devuelve
        --------
        matplotlib.axes.Axes
            
        Ejemplo
        -------
        >>> Doñana = dinsar.example.get_model()
        >>> Precipi = Doñana.get('P')
        >>> Precipi.plot(['Moguer', 'Niebla'])
        """

        if 'values' in kwargs.keys():
            values = kwargs['values']
            kwargs.pop('values')
        else:
            values = 'std'

        own_kwargs = dict(legend=True, title='Estaciones pluviométricas')
        kwargs = _update_kwargs(own_kwargs, kwargs)

        estaciones = self.estaciones if isinstance(estaciones, str) \
                     and estaciones.lower() == 'all' else estaciones
        
        if isinstance(estaciones, str) and estaciones.lower() == 'all':
            series = pd.DataFrame([self.take(i, values=values) for i
                                   in self.estaciones]).T
            
        elif isinstance(estaciones, str) and not estaciones.lower() == 'all':
            series = self.take(estaciones, values=values).to_frame()
                      
        elif isinstance(estaciones, list):
            series = pd.DataFrame([self.take(i, values=values) for i
                                   in estaciones]).T
            
        ax = series.plot(**kwargs)
        
        ax.set_ylabel(f"Precipitación ({self.units})",
                        fontsize=15, labelpad = 28, rotation=270)
        ax.set_title(kwargs['title'], fontsize=24, pad=15)

        if savefig:
            ax.figure.savefig('Estaciones.png')

        return ax

class Piezometria(_Bd):
    """
                           BASE DE DATOS DE PIEZOMETRÍA
                           
    Parámetros
    ----------
    fname str o pandas.DataFrame
        Ruta relativa o absoluta al archivo (si str) u objeto pandas.DataFrame.
        Los **kwargs y el resto de argumentos son utilizados solo si fname=str.
    
    name str 
        Nombre con el que se bautiza a la base de datos.
    
    date_format str (opcional) Default '%Y-%m-%d'
        Formato de las fechas.
        Nomenclatura: https://www.dataindependent.com/pandas/pandas-to-datetime/
        
    **kwargs: Argumentos opcionales que se pasan a la función pandas.read_excel
        o pandas.read_csv según la naturaleza del archivo.
        
    Devuelve
    --------
    dinsar.parts.Piezometria
    
    Ejemplo
    -------
    >>> fname = dinsar.example.get_path('piezometria_bd')
    >>> Piezo = dinsar.Piezometria(fname, 'Piezo_bd', sep='\t')
    
    """
    
    def __init__(self, *args, **kwargs):

        kwargs = _update_kwargs(dict(color='#3C8BD1', units='msnm'), kwargs)

        super().__init__(*args, **kwargs)

        self._sensor_name = 'piezómetro'

    __doc__ += _Bd.__doc__

    piezos = _Bd.elementos


class DataBase(_Bd):
    """
                           BASE DE DATOS DE OTRO NATURALEZA
    Funciona exactamente igual que las bases de datos de Piezometría y Precipitación.
                           
    Parámetros
    ----------
    fname str o pandas.DataFrame
        Ruta relativa o absoluta al archivo (si str) u objeto pandas.DataFrame.
        Los **kwargs y el resto de argumentos son utilizados solo si fname=str.
    
    name str 
        Nombre con el que se bautiza a la base de datos.
    
    date_format str (opcional) Default '%Y-%m-%d'
        Formato de las fechas.
        Nomenclatura: https://www.dataindependent.com/pandas/pandas-to-datetime/

    **kwargs: Argumentos opcionales que se pasan a la función pandas.read_excel
        o pandas.read_csv según la naturaleza del archivo.
        
    Devuelve
    --------
    dinsar.parts.DataBase
    
    """
    
    def __init__(self, *args, **kwargs):

        kwargs = _update_kwargs(dict(color=random_color(), units=''), kwargs)

        super().__init__(*args, **kwargs)

        self._sensor_name = 'sensor'

    __doc__ += _Bd.__doc__
