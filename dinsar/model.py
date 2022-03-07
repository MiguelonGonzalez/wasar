"""

Módulo que alberga la clase Model, objeto principal del paquete.

"""
import geopandas as gpd

from .agregado import Agregado
from .parts import *
from .plot import _mapa
from .utils._utils import _filtro_agregados


class Model:
    """
    Objeto principal del programa.
    Su inicialización requiere de la entrada de un shapefile de polígonos. Esta
    capa se lee y maneja a través de la librería geopandas y actúa como elemento
    central de esta clase. Los polígonos de esta capa (en adelante llamados 
    'agregados') conformarán las áreas de la zona de estudio donde se estudiará
    la deformación del terreno.

    El correcto funcionamiento de este objeto requiere de la inclusión (a través
    del método 'append') de las diferentes partes que lo componene. Entre éstas,
    la principal y más importante es la de tipo 'Dataset', la cual contiene un 
    conjunto de datos de deformación del terreno (datos D-InSAR). Existen otros 
    tres tipo de partes distintas, diseñadas para mejorar la visualización e
    interpretación de la deformación observada en cada agregado. Éstas se corresponden
    con los objetos de tipo Piezometria, Precipitacion y Bd, que albergan
    (respectivamente) bases de datos (espaciales o no) con información piezométrica,
    pluviométrica o de cualquier otra naturaleza.

    Se pueden definir y añadir al modelo tantas partes como se desee, encontrándose
    las especificaciones de su definición en el módulo 'wasar.parts', siendo 
    también posible acceder a esta información desde la ayuda de cada una de ellas:

    >>> help(wasar.Piezometria)

    El sistema de referencia de coordenadas (SRC, o CRS por sus siglas en 
    inglés) de la capa de entrada dictará el SRC del proyecto. Las partes añadidas
    al modelo se reproyectarán al SRC de éste en caso de que no coincidir con él.
    Es posible cambiar el SRC del proyecto a través del método 'reproyect'.    

    Parámetros
    ----------
    fname str 
        Ruta al shapefile de entrada.

    epsg int
        SRC del modelo. Si la capa de entrada no se encuentra en ese sistema,
        será reproyectada. Por defecto es el SRC de la capa de entrada.

    **kwargs
        Los argumentos adicionales se pasan a la función geopandas.read_file.

    Devuelve
    -------
    wasar.Model

    Ejemplo
    --------

    >>> fname = wasar.example.get_path('Agregados')
    >>> Doñana = wasar.Model(fname)

    """

    def __init__(self, fname, epsg=None, **kwargs):

        self._fname = fname
        _gdf = gpd.read_file(fname, **kwargs)
        self._gdf = _gdf

        # No existe SRC y no se especifica ninguno:
        if not _gdf.crs and not epsg:
            raise ("Esta capa no tiene un sistema de referencia (SRC) asociado. "
                  "Consulta cuál es e indícalo mediante el argumento 'epsg'.\n"
                  "Ejemplo del epsg del SRC ETRS89: 25830")

        # No existe SRC y se indica uno:
        if not _gdf.crs and epsg:
            _gdf.set_crs(self.crs, inplace=True)

        # Existe SRC y no cuadra con el indicado:
        if epsg and (_gdf.crs != epsg):
            self.reproyect(epsg=epsg)

        # Adecuación de la capa de Agregados:
        if _gdf.index.name == 'Agregado':
            _gdf.reset_index(inplace=True)
            _gdf['Agregado'] = _gdf['Agregado'].astype(str)

        elif 'Agregado' in _gdf.columns:
            _gdf['Agregado'] = _gdf['Agregado'].astype(str)

        else:
            raise TypeError("No existe la columna 'Agregado'.")

        # Diccionarios para albergar las partes del modelo
        self._estanteria_PS = dict()
        self._estanteria_piezo = dict()
        self._estanteria_precipi = dict()
        self._estanteria_other_bd = dict()

        # Diccionario general con las partes del modelo (backpack)
        self._update()

    def __call__(self):
        """Devuelve la capa de polígonos"""
        return self.gdf

    def __str__(self):
        """Imprime la información general del modelo"""
        return self._info(_print=False)

    def _check_crs(self, part):
        """Método privado. Comprueba el SRC de la capa de entrada al modelo y la
        reproyecta si éste no coincide con el del modelo"""

        def change_crs(part):
            antiguo = part.crs
            part.gdf.to_crs(epsg=self.crs, inplace=True)
            print(f"Geometría de '{part.name}' reproyectada desde "
                  f"SRC {antiguo} al {part.crs}")

        if isinstance(part, Dataset):
            if not part.crs == self.crs:
                change_crs(part)

        else:
            if part.has_geometry() and not part.crs == self.crs:
                change_crs(part)

    def is_all_crs_equal(self):
        """ Devuelve True si todas las capas del modelo presentan el mismo
        sistema de coordenadas."""

        agregados = self.crs
        datasets = [i.crs for i in self._estanteria_PS.values()
                    if self._estanteria_PS]

        if self._estanteria_piezo:
            piezo = [i.gdf.crs for i in self._estanteria_piezo.values() 
                    if  i.has_geometry()]

        if self._estanteria_precipi:
            precipi = [i.gdf.crs for i in self._estanteria_precipi.values() 
                      if i.has_geometry()]

        prjs = [agregados, *datasets, *piezo, *precipi]

        return all(i == prjs[0] for i in prjs)

    @property
    def crs(self):
        """ Devuelve el Sistema de Coordenadas (SRC) del modelo."""
        return self._gdf.crs

    @property
    def gdf(self):
        """ Devuelve el geopandas.GeoDataFrame del modelo, con los diferentes
        datasets como columnas."""
        return self._gdf

    @property
    def agregados(self):
        """ Devuelve una lista con los agregados del modelo. """
        return self.gdf['Agregado'].tolist()

    def reproyect(self, crs_diana):
        """ Reproyecta todas las capas del proyecto al SRC indicado (crs_diana).

        Parámetros
        ----------

        crs_diana: int
            Número del EPSG del SRC destino.
 
        Devuelve
        -------
        None

        Ejemplo
        --------
        >>> Doñana = wasar.example.get_model()
        >>> Doñana.reproyect(4326)
        >>> ( Doñana.is_all_crs_equal() and Doñana.crs == 4326 )
        >>> True

        """

        change_crs = lambda shp: shp.to_crs(epsg=crs_diana, inplace=True)

        change_crs(self.gdf) # ReproyecciÃ³n de los agregados.

        if self._estanteria_PS:
            [change_crs(i.gdf) for i in self._estanteria_PS.values()]

        if self._estanteria_piezo:
            [change_crs(i.gdf) for i in self._estanteria_piezo.values() 
            if i.has_geometry()]

        if self._estanteria_precipi:
            [change_crs(i.gdf) for i in self._estanteria_precipi.values() 
            if i.has_geometry()]

        self._update()

    def _update(self):
        """ Método privado. Actualiza el diccionario global 'backpack', que 
        alberga todas las partes del modelo.
        """

        self._backpack = {'PS':self._estanteria_PS,
                         'Piezo':self._estanteria_piezo,
                         'Precipi':self._estanteria_precipi,
                         'Other_bd':self._estanteria_other_bd}

    def append(self, part):
        """ 
        Añade una nueva parte al modelo con el mismo nombre con el que se definió.
        Éstas pueden ser:

            A. Dataset. Dataset DInSAR. Supone la creación de una nueva columna 
               en la capa general del modelo (gdf) con el mismo nombre del Dataset,
               que alberga en forma de lista únicamente los puntos (según su ID) 
               que caen dentro de cada agregado. Resultado de una operación de
               intersección. Los datasets añadidos se pueden consultar a través del
               atributo 'datasets'.

            B. Piezometria. Base de datos de piezometría. Se pueden consultar
               todas las añadidas a través del atributo 'piezo_bds'.

            C. Precipitacion. Base de datos de precipitacion. Consultar a través
               'precipi_bds'.

            D. DataBase: Cualquier otra base de datos que se quiera incluir en
               el modelo. Consultar a través de 'other_bds'


        Parámetros
        ----------
        part Dataset, Piezometria, Precipitacion, DataBase (en wasar.parts)

        Devuelve
        -------
        None
  
        Ejemplo
        -------
        Adición de un Dataset al modelo. Seguir el mismo proceso para añadir
        el resto de partes.

        >>> fname = wasar.example.get_path('agregados')
        >>> Doñana = wasar.Model(fname)

        >>> asc_fname = wasar.example.get_path('sentinel-asc')
        >>> Asc = wasar.Dataset(asc_fname, name='Asc')

        >>> Doñana.append(Asc)        

        """
        self._check_crs(part)
        msg = (f"Un parte con el mismo nombre que '{part.name}' ya se encuentra"
               " incluida en el modelo")

        if isinstance(part, Dataset):
            assert part.name not in self._estanteria_PS.keys(), msg
            name = part.name
            points = part.gdf
            self.gdf[part.name] = self.gdf.geometry.apply(lambda x:
                   points.loc[points.intersects(x)]['ID'].tolist())

            self._estanteria_PS.update({name:part})

        elif isinstance(part, Piezometria):
            assert part.name not in self._estanteria_piezo.keys(), msg
            self._estanteria_piezo.update({part.name:part})

        elif isinstance(part, Precipitacion):
            assert part.name not in self._estanteria_precipi.keys(), msg
            self._estanteria_precipi.update({part.name:part})

        elif isinstance(part, DataBase):
            assert part.name not in self._estanteria_other_bd.keys(), msg
            self._estanteria_other_bd.update({part.name:part})

        else:
            raise TypeError(f"{type(part)} no es un tipo válido de entrada de "
                            "datos al modelo. válidos son: {type(Dataset())}, "
                            "{type(Piezometria())}, {type(Precipitacion())}.")
             
        self._update()
   
    @property  
    def datasets(self):
        """ Devuelve una lsita con los Datasets del Modelo. """
        return list(self._estanteria_PS.keys())

    @property  
    def piezo_bds(self):
        """ Devuelve una lista con las bases de datos de piezometría del modelo."""
        return list(self._estanteria_piezo.keys())

    @property  
    def precipi_bds(self):
        """ Devuelve una lista con las bases de datos de precipitación del modelo."""
        return list(self._estanteria_precipi.keys())

    @property
    def other_bds(self):
        return list(self._estanteria_other_bd.keys())

    def get(self, name):
        """ Método para acceder a las distintas partes del modelo.

        Parámetros
        ----------
        name: str
            Nombre con el que se definión la parte.
            >>> part.name

        Devuelve
        --------
        objeto con la parte indicada. Posibilidades: Dataset, Piezometria,
        Precipitacion.

        """

        parts = [part for dic in self._backpack.values() for part in dic]
        msg = f"'{name} 'no es un str o no es una parte del modelo. Opciones: {parts}"
        assert isinstance(name, str) and name in parts, msg

        for dic in self._backpack.values():
            for key, value in dic.items():
                if key is name:
                    return value
  
    def agregado(self, *args, vm=False, _lotes=False):
        """ Selección de uno o varios polígonos (agregados) del modelo.

        Parámetros
        -----------
        *args str --> Al menos introducir uno.
            Nombre (ID) del agregado a estudiar tal cual figure en el campo 
            'Agregados' de la capa del modelo (Modelo.gdf). Se pueden incluir
            varios para un estudio conjunto de la deformación y de las funciones
            que se desprenden del objeto saliente.

        vm bool Default, False. Opcional
            Descomponer el movimiento observado a su componente vertical. Requiere
            de la columna 'cosU' en el GeoDataFrame: columna que contiene los valores
            del vector unitario LOS (Line Of Sigh) a lo largo de la dirección
            vertical.            

        _lotes: variable privada.

        Devuelve
        -------
        wasar.agregado.Agregado

        Ejemplo
        --------

        >>> Doñana = wasar.example.get_model()

        >>> Doñana.agregado('1')       # Estudio del agregado 1
        >>> Doñana.agregado('2','4')   # Estudio conjunto de los agregados 2 y 4

        """

        if len(args) == 0:
            raise TypeError("No se ha indicado ningún agregado. Indica al menos uno.")

        _filtro_agregados(args, self.agregados)

        df = self.gdf.copy()

        # Estudio de un único agregado.
        if len(args) == 1:
            agregado = args[0]
            gdf = df.loc[df['Agregado']==agregado]

        # Estudio conjunto de varios agregados.
        else:

            from shapely.geometry import MultiPolygon

            df = df.set_index('Agregado').loc[list(args)].reset_index()
            gdf = gpd.GeoDataFrame(df[self.datasets].agg('sum')).T
            gdf['geometry'] = MultiPolygon(df['geometry'].agg(list))
            gdf['Agregado'] = '-'.join( df['Agregado'].astype(str).tolist() )
            gdf.set_crs(self.crs, inplace=True)
            gdf = gdf.reindex(['Agregado','geometry', *self.datasets], axis=1)

        return Agregado(gdf, self._backpack, vm=vm, _lotes=_lotes)

    def mapa(self, *args, basemap=True, savefig=False, LayerControl=True,
                   **kwargs):
        """ Representación espacial de los agregados del modelo.
        La representación se realiza a través de la librería Folium (si basemap=True)
        o Matplotlib (basemap=False), guardándose la imagen resultante
        (si savefig=True) como un archivo 'html' en el primer caso y 'png' en el
        segundo.

        Parámetros
        ----------

        *args: Agregados a plotear. int o list

        basemap bool Default True, opcional
            Adición de mapas base mediante Folium.

        savefig bool Default False, opcional
            Guardado de la imagen en el directorio de trabajo.

        **kwargs: opcional
            Argumentos adicionales. Se pasan a la función folium.Map o 
            geopandas.GeoDataFrame.plot() según el parámetro 'basemap'.

        Devuelve
        -------
        objeto folium.Map (si basemap=True) o matplotlib.axes.Axes (False)

        Ejemplo
        -------
        >>> Doñana = wasar.example.get_model()
        >>> Doñana.mapa()         # Visualización de todos los agregados.
        >>> Doñana.mapa('3')      # Visualización del agregado '3'

        """

        _filtro_agregados(args, self.agregados)

        if len(args) == 0:
            gdf = self.gdf

        elif len(args) == 1:
            gdf = self.gdf.loc[self.gdf['Agregado']==args[0]]

        else:
            gdf = self.gdf.set_index('Agregado').loc[list(args)].reset_index()

        if basemap:
            own_kwargs = dict(tooltip='Agregado', name='Agregados',
                              style_kwds=dict(color='black', fillColor='#F39AB8',
                              fill=True, weight=1))
        else:
            own_kwargs = dict()

        own_kwargs['LayerControl'] = LayerControl

        return _mapa(gdf, own_kwargs, kwargs, basemap, savefig=savefig)

    def porLotes(self, agregados='all', 
                set_piezo=True, set_estacion=True, set_other_bd=False,
                piezo_kw=None, estacion_kw=None, other_bd_kw=False,
                mapa=True, plot=True, mapa_kw=None, plot_kw =None,
                save_arrays=True, 
                cwt=False, xwt=False, cwt_kw=None, xwt_kw=None,
                verbose=False, raise_errors=False):

        """ Método para analizar de manera automática un conjunto de agregados.
        Para cada uno de ellos realiza los cálculos indicados por los argumentos
        de la función y guarda los resultados en una carpeta con su mismo nombre.
        A su vez, todas las carpetas generadas quedan albergadas en otra denominada
        'Por Lotes', ubicada en el directorio de trabajo que se esté manejando.

        Parámetros
        ----------
        agregados str or list. Default 'all'
            Nombre de los agregados a analizar de forma automática. Por defecto se
            analizarn todos ('all').

        set_piezo: bool. Default True. Opcional.
            Vincular un piezómetro al agregado mediante el método 'set_piezo'.

        set_estacion: bool. Default True. Opcional.
            Vincular una estación al agregado mediante el método 'set_estacion'.

        set_other_bd: bool. Default True. Opcional.
            Vincular otro sensor al agregado mediante el método 'set_other_bd'.

        save_arrays bool. Default True. Opcional.
            Guardar las series temporales existentes de las variables asociadas a 
            cada agregado en un fichero txt.
        
        mapa: bool. Default True. Opcional.
            Generar un mapa del agregado (método 'mapa') y lo guarda como
            un fichero 'png' (si basemap=True) o 'html' (basemap=False).

        plot: bool. Default True. Opcional.
            Genera una gráfica del agregado (método 'plot') y la guarda como un fichero
            'png'.

        cwt: bool. Default False. Opcional.
            Genera la CWT (Continuous Wavelet Transform) de la variable del agreagdo
            (o serie temporal) indicada en 'cwt_kw'.

        xwt: bool. Default False. Opcional.
            Genera la XWT (Cross Wavelet Transform) y WTC (Wavelet Transform Coherence)
            entre las variables del agregado (o cualquiera otra) indicadas en 'xwt_kw'.

        piezo_kw: None or dict. Default None. Opcional
            **kwargs para pasar a la función 'set_piezo' cuando set_piezo es True.

        estacion_kw: None or dict. Default None. Opcional
            **kwargs para pasar al método Agregado.set_estacion cuando set_estacion
            es True.

        other_bd_kw: None or dict. Default None. Opcional.
            **kwargs para pasar al método Agregado.set_other_bd cuando set_other_bd
            es True.

        mapa_kw: None or dict. Default None. Opcional
            **kwargs para pasar al método Agregado.mapa cuando mapa es True.

        plot_kw: None or dict. Default None. Opcional
            **kwargs para pasar al método Agregado.plot cuando plot es True.

        cwt_kw: None or dict. Default None. Opcional
            **kwargs para pasar al método Agregado.wavelet cuando cwt es True.
            En este caso la frecuencia de muestreo (freq) y las series temporales
            (una-->str, varias-->list) han de ir incluidas en el dicionario (**kwargs).
            Se pueden inlcuir tantas series como transformadas CWT se quieran hacer
            de cada agregado.

            Ejemplo: >>> cwt_kw = dict(freq='M', series='piezo', dt=1)

        xwt_kw: None or dict. Default None. Opcional
            **kwargs para pasar al método Agregado.wavelet cuando xwt es True.
            Al igual que en 'cwt_kw', la frecuencia de muestreo (freq) y las series 
            temporales han de ir incluidas en el dicionario (**kwargs). 'series' ha 
            de ser una lista de dos elementos.

        verbose: bool. Default True. Opcional.
            Guardar la información de cada agregado y del modelo en un fichero de texto.

        rise_errors: bool. Default False. Opciona.
            Muestra los errores (si se produjera alguno), deteniendo así la ejecución.
        
        Devuelve
        --------
        _Lotes (clase privada)

        Ejemplo
        -------
        Analizar los agregados '2' y '5' obteniendo, además del mapa, gráfica y
        ficheros de textos con los arrays de la piezometría y la precipitación,
        un análisis wavelet conjunto entre la deformación promedio de cada agregado
        del dataset 'Asc' y el piezómetro más cercano (por defecto). Además, se ha
        mantenido la misma estación pluviométrica de referencia en todos ellos
        ('Almonte').

        >>> Doñana = wasar.example.get_model()
        >>> xwt_kw = dict(freq='M',series=['Asc', 'piezo'])    
        >>> Doñana.porLotes(agregados=['2','4','5'],
                            estacion_kw=dict(estacion='Almonte'),
                            xwt=True, xwt_kw=xwt_kw)

        """

        from .utils._lotes import _Lotes

        kwargs = locals().copy()
        del kwargs['_Lotes'], kwargs['self']

        _Lotes(self, **kwargs)

    def _info(self, _print=True):
        """ Método privado que devuelve la información del Modelo. Públicamente
        utilizado como propiedad, imprimiendo en pantalla la información.

        _print bool, Default True
            Cuando es True imprime en pantalla el texto (devuelve NoneType), de
            lo contrario devuelve la información como un str.

        """

        from .plot import _b

        texts = [
        f". {_b('A-DInSAR datasets')}: {', '.join(self.datasets)}",
        f". {_b('Bd Piezometría')}: {', '.join(self.piezo_bds)}",
        f". {_b('Bd Precipitación')}: {', '.join(self.precipi_bds)}",
        f". {_b('Otras Bd')}: {', '.join(self.other_bds)}",
        f". {_b('Nº Agregados')}: {self.gdf['Agregado'].unique().size}",
        f". Sistema de Referencia de Coordenadas del Proyecto' ({_b('SRC')}):\
            {self.crs.name}",
        f". Todas las capas presentan el {_b('mismo SRC')}: {self.is_all_crs_equal()}"
                ]

        if _print:
            print('\n'.join(texts))
        else:
            return '\n'.join(texts)

    info = property(_info)
