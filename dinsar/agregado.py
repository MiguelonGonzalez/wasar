"""

Módulo que contiene la clase Agregado.

"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .plot import  _add_basemap, _colorbar, _wasar_plot_params, _get_datasets_cmap
from .utils._utils import _update_kwargs, _importing_folium, _checking_folium

if _importing_folium:
    import folium

class Agregado:
    """ Clase con las funciones necesarias para analizar la deformación del terreno
    en uno o varios de los agregados del modelo, únicamente funcional a través
    del método 'agregado' del objeto Modelo. """

    def __init__(self, gdf, _backpack, vm=False, _lotes=False):

        self._gdf = gdf
        self._backpack = _backpack
        self.name = f"Agregado {gdf['Agregado'].iloc[0]}"
        self._lotes = _lotes
        self._vm = vm

        if len(_backpack['PS']) == 0:
            raise SystemError("El modelo no tiene datasets en este agregado.")

        self._tr_dict = dict()   # Diccionario con los datasets cribados
        self._subsets()   # Extracción de los datasets

        # ----------------------------------------------------------------------
        # Definición automática de piezómetro y pluviómetro:
        # Valores por defecto (más cercano y primera bd). Solo
        # se realiza cuando la creación de este agregado no responde
        # a una ejecución por lotes.
        # ----------------------------------------------------------------------

        if not _lotes:
            try:
                self.set_piezo(_silent=True)
            except:
                pass

            try:
                self.set_estacion(_silent=True)
            except:
                pass

    def __call__(self):
        return self._gdf

    def __str__(self):
        return self._info(_print=False)

    @property
    def gdf(self):
        """Devuelve el GeoDataFrame del agregado"""
        return self._gdf

    @property
    def area(self):
        """Devuelve el área del agregado en hectáreas, asumiendo que la unidad
        del SRC del proyecto son los metros.
        Puedes comprobar el SRC de las capas del modelo de la siguiente manera:
        >>> Doñana = wasar.example.get_model()
        >>> Doñana.crs
        """
        return self.gdf.area/10000

    @property
    def centroid(self):
        """ Centroide del polígono."""
        return self.gdf.centroid

    @property
    def datasets(self):
        """Devuelve una lista con el nombre de los datasets que contienen puntos
        en este agregado."""
        return list(self._tr_dict.keys())

    @property
    def piezos(self):
        """Devuelve un pandas.DataFrame con los piezómetros vinculados al agregado
        o None cuando no se ha vinculado ninguno.
        Ver 'set_piezo'.
        """
        if hasattr(self, '_info_piezos'):
            return self._info_piezos['piezos']
        else:
            return None            

    @property
    def estacion(self):
        """Devuelve una pandas.Serie con la estación vinculada al agregado o
        None cuando no se ha vinculado ninguna.
        Ver 'set_estacion'.
        """
        if hasattr(self, '_info_estaciones'):
            return self._info_estaciones['estacion']
        else:
            return None

    @property
    def other_sensors(self):
        """Devuelve un pandas.DataFrame con otros sensores vinculados al agregado o
        None cuando no se ha vinculado ninguna otra base de datos.
        Ver 'set_other_bd'.
        """
        if hasattr(self, '_info_other_sensors'):
            return self._info_other_sensors['sensors']
        else:
            return None

    @property
    def vm(self):
        """ Devuelve True si los valores de deformación de las serires temporales 
        únicamente representan la componente vertical, y False si representan 
        los valores de deformación según la dirección del LOS (Line Of Sigh)."""
        return self._vm

    def _subsets(self):
        """Método privado para el análisis de los datasets que contienen puntos
        en el agregado en cuestión. Se realiza únicamente durante la construcción
        del objeto, y a partir del método 'Dataset.subset'."""

        estanteria_PS = self._backpack['PS']
        for name, dataset in estanteria_PS.items():
            puntos = self.gdf[name].tolist()[0]

            # Cojo los dataset:
            if len(puntos) != 0:
                self._tr_dict.update({name:dataset.subset(puntos, vm=self._vm )})

            else:
                if not self._lotes:
                    print(f"El Dataset {name} no presenta "
                          f"puntos en este agregado ({self.name}).")

        if not self._tr_dict:
            raise RuntimeError(f"{self.name} no contiene puntos de ningún dataset.")

    def subset(self, dataset):
        """ Devuelve un objeto de tipo Dataset con la información que el agregado
        contiene del dataset indicado. Se basa en el método 'Dataset.subset',
        siendo el argumento 'points' una lista con todos los puntos disponibles
        en el agregado. En esta ocasión el argumento 'vm' se debería de indicar
        en la definición del agregado.

        Más información: >>> help(wasar.Dataset.subset)

        Parámetros
        ----------
        dataset str
            Nombre del dataset. Se puede consultar el nombre de los datasets que
            presetan información en este agregado a través del método 'datasets':
            >>> Doñana.agregado('2').datasets

        Devuelve
        --------
        wasar.Dataset

        Ejemplo
        -------
        Obtención del dataset 'sentinel-asc' (alias: 'Asc') con únicamente los
        puntos presentes en el agregado '2' del modelo.

        >>> Doñana = wasar.example.get_model()
        >>> A2 = Doñana.agregado('2').subset('Asc')

        """
        return self._tr_dict[dataset]

    def set_piezo(self, piezometro=None, bd=None,  way='nearest', orden=1,
                        radius=10, _silent=False):
        """Vincula uno o varios piezómetros al agregado si el modelo contiene
        una base de datos de piezometría (objeto part.Piezometria.

        Se basa en el método 'find_element' del objeto Piezometría, siendo 
        'xy' el centroide del agregado.

        Para más información: >>> help(wasar.Piezometria.find_element)

        Parámetros
        ----------
        piezometro: list o str. Default None, opcional
            Lista con los piezómetros a seleccionar. Cambia el valor de 'way'
            a 'manual' cuando se cambia su valor por defecto.

        bd str, Default None, Opcional
            Si el modelo presenta más de una base de datos de piezometría, nombre
            de aquella sobre la que se realizará la selección. Por defecto toma
            la primera (o única) base de datos de piezometría del modelo.

        way str, Default 'nearest'. Opcional
            Forma de seleccionar el sensor.
            'nearest': Se ordenan los sensores según la cercanía al punto
                       indicado (xy) y se selecciona el deseado según
                       el grado de cercanía (orden).
            'radius': Se seleccionan todos los piezómetros dentro de el radio 
                      de acción determinado por 'radius'.
            'manual': selección manual de los piezómetros por medio de una lista.

        orden int Default 1. Opcional
            Orden de selección del piezómetro cuando way='nearest'
            Ejemplo: orden=1 (se selecciona el piezómetro más cercano)
                     orden=2 (se selecciona el segundo piezómetro más cercano)
                     orden=-1 (se selecciona el piezómetro más lejano)
                     
        radius int, Default 1. Opcional
            Radio de acción (en Km) cuando way='radius'.

        _silent: bool, Privada.

        Devuelve
        --------
        self (el propio objeto). Hecho así para poder empalmar métodos. (ver ejemplo)

        Ejemplo
        -------
        Seleccionar el agregado '2' y vincularle el segundo piezómetro más cercano
        y la estación pluviométrica más cerana
        >>> Doñana = wasar.example.get_model()
        >>> A2 = Doñana.agregado('2')
        >>> A2.set_piezo(way='nearest', orden=2)
        >>> A2.set_estacion()

        Forma rápida de hacerlo:
        >>> A2 = Doñana.agregado('2')
        >>> A2.set_piezo(way='nearest', orden=2).set_estacion()

        """
        assert isinstance(way, str), f"'{way}' no es un str."
        way = 'manual' if piezometro != None else way
        bd_object = self._get_bd(bd, 'Piezo')
        centroid = self._gdf.geometry.centroid

        if way == 'nearest' or way == 'radius':
            _piezos =  bd_object.find_element(centroid, way=way,
                                                  orden=orden, radius=radius)

        elif way == 'manual':
            _piezos =  bd_object.take(piezometro).pivot(index='Fechas',
                                                            columns='Nombre',
                                                            values='Valores')
        else:
            raise ValueError("'way' no coincide con 'nearest', 'radius' o 'manual'")

        self._info_piezos = {'piezos':_piezos, 'bd':bd_object}

        if not _silent:
                print(f"{len(self.piezos.columns)} piezo(s) fijado(s) "
                      f"según el método '{way}'.")

        return self

    def set_estacion(self, estacion=None, bd=None, way='nearest', orden=1,  
                           values='std', _silent=False):
        """Vincula una estación de pluviometría al agregado si el modelo contiene
        una base de datos de precipitación (objeto part.Precipitacion).

        Para comprobar la estación fijada utiliza el método 'estacion'.Ej:
        >>> A2.estacion

        Se basa en el método 'find_element' del objeto Precipitacion, siendo 
        'xy' el centroide del agregado.
        Funciona de forma muy parecida al método 'set_piezo', aunque en este caso
        únicamente se puedo elegir una estación y se puede de indicar la natruraleza
        de los datos ('values').

        Para más información: >>> help(wasar.Precipitacion.find_element)

        Parámetros
        ----------
        estacion: str, Default None, Opcional
            Nombre de la estación de precipitación a vincular al agregado. Cambia
            el valor de way a 'manual' cuando se cambia su valor por defecto.

        bd str, Default None, Opcional
            Si el modelo presenta más de una base de datos de precipitación, nombre
            de aquella sobre la que se realizará la selección. Por defecto toma
            la primera (o única) base de datos de precipitación del modelo.

        way str, Default 'nearest', Opcional
            Forma de seleccionar el sensor.
            'nearest': Se ordenan las estaciones según la cercanía al agregado y 
                       se selecciona la deseada según el grado de cercanía (orden).
            'manual': selección manual de la estación por medio de una lista.

        orden int Default 1, Opcional
            Orden de selección de la estación cuando way='nearest'

        values str, Default 'std', Opcional
            La función devuelve los valores originales de precipitación ('original')
            o aquellos resultantes de la desviación estándar acumulada ('std')            

        _silent: bool, Privada.

        Devuelve
        --------
        self (el propio objeto). Hecho así para poder empalmar métodos. (ver ejemplo)

        Ejemplo
        -------
        (Ver ejemplo del método 'set_piezo')

        """
        # Comprobaciones -------------------------------------------------------
        msg = f"'{values}' no es válido. Opciones: 'std' o 'original'"
        assert values in ['std', 'original'], msg

        assert isinstance(way, str), f"'{way}' no es un str."

        msg = f"'{way}' no es válido. Opciones para 'way': 'nearest' o 'manual'"
        assert way in ['nearest', 'manual'], msg

        way = 'manual' if estacion else way

        bd_object = self._get_bd(bd, 'Precipi')
        # ----------------------------------------------------------------------

        if way == 'nearest':
            _estacion = bd_object.find_element(self.centroid, way='nearest',
                                          orden=orden).squeeze().name
        else:
            _estacion = estacion

        _estacion = bd_object.take(_estacion, values=values)

        self._info_estaciones = {'estacion':_estacion, 'bd':bd_object}

        if not _silent:
            print(f"Fijada la estación: {self.estacion.name} "
                  f"según el método '{way}'.")
        
        return self

    def set_other_bd(self, sensor=None, bd=None,  way='nearest', orden=1,
                        radius=10, _silent=False):
        """ Método análogo a 'set_piezo' pero utilizando cuando se quiere vincular
        el sensor de una base de datos de distinta naturaleza (tipo DataBase).

        Ver también
        -----------
        >>> help(Agregado.set_piezo)
        >>> help(wasar.DataBase)

        """
        assert isinstance(way, str), f"'{way}' no es un str."
        way = 'manual' if sensor != None else way
        bd_object = self._get_bd(bd, 'Other_bd')
        centroid = self._gdf.geometry.centroid

        if way == 'nearest' or way == 'radius':
            _sensores =  bd_object.find_element(centroid, way=way,
                                                  orden=orden, radius=radius)

        elif way == 'manual':
            _sensores =  bd_object.take(sensor).pivot(index='Fechas',
                                                      columns='Nombre',
                                                      values='Valores')
        else:
            raise ValueError("'way' no coincide con 'nearest', 'radius' o 'manual'")

        self._info_other_sensors = {'sensors':_sensores, 'bd':bd_object}

        if not _silent:
                print(f"{len(self.other_sensors.columns)} sensor(es) fijado(s) "
                      f"según el método '{way}'.")

        return self

    def plot(self, subsets='all', piezo=True, pluvi=True, other_sensors=False,
                   savefig=False):
        """ Representación de las series temporales promedio de los datasets
        presentes en el agregado, y adición en una gráfica inferior de las series
        temporales de los piezómetros y pluviómetros más cercanos. Aunque estos 
        últimos se vinculan durante la construcción de este objeto, se pueden 
        cambiar respectivamente a través de los métodos 'set_piezo' y 'set_estación',
        así como cualquier otro sensor introducido mediante otra base de datos.

        Parámetros
        ----------

        subsets: str o list. Default 'all'. Opcional
            Representación de todos los datasets ('all') o únicamente alguno (str)
            o algunos (list).

        piezo: bool. Default True. Opcional.
            Incluir en una gráfica inferior las series temporales de piezometría
            definidas para este agregado. Se pueden consultar o modificar a través
            de 'Agregado.piezos' o 'Agregado.set_piezo', respectivamente.

        pluvi: bool. Default True. Opcional.
            Incluir en una gráfica inferior las series temporales de precipitación
            definidas para este agregado. Se pueden consultar o modificar a través
            de 'Agregado.estacion' o 'Agregado.set_estacion', respectivamente.

        other_sensors: bool. Default False. Opcinal.
            Incluir en una gráfica inferior las series temporales de otros sensores
            definidas para este agregado.  Se pueden consultar o modificar a través
            de 'Agregado.other_sensors' o 'Agregado.set_other_bd', respectivamente.
            Únicamente se rotula su eje cuando pluvi=False.

        savefig bool Default False. Opcional.
            Guarda la gráfica resultante en el directorio de trabajo con el
            siguiente nombre: 'Agregado - nombre.png'
            También se puede guardar accediendo a la función
            'matplotlib.pyplot.savefig', lo que permite especificar parámetros:
            >>> A2 = Doñana.agregado('2')
            >>> A2.plot().figure.savefig(*args, **kwargs)

        Devuelve
        --------
        matplotlib.axes.Axes

        Ejemplo
        ------
        >>> Doñana = wasar.example.get_model()
        >>> A2 = Doñana.agregado('2')
        >>> A2.plot()

        """

        piezo = True if piezo is True and self.piezos is not None else False
        pluvi = True if pluvi is True and self.estacion is not None else False
        other_sensors = True if other_sensors is True and self.other_sensors is \
                                                          not None else False


        to_mdates = lambda x: mdates.date2num(x)

        if piezo or pluvi or other_sensors:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            fig.subplots_adjust(hspace=0)

        else:
            fig, ax1 = plt.subplots(figsize=(15,10))

        # Ploteo de los Datasets -----------------------------------------------
        subsets = self.datasets if isinstance(subsets, str) and \
                                subsets.lower() == 'all' else subsets

        if isinstance(subsets, list):
            msg = "El nombre de alguna trayectoria es incorrecto."
            assert all([i in self.datasets for i in subsets]), msg

            datasets_list = [self._tr_dict.get(i) for i in subsets]
            for dataset in datasets_list:
                ts = dataset.mean()
                ax1.plot(ts.index.map(to_mdates),ts.values,
                         color=dataset.color, label=ts.name)

        # Un único subset, introducido como string.
        elif isinstance(subsets, str):
            assert subsets in self.datasets, 'No existe ese Dataset.'
            dataset = self._tr_dict[subsets]
            ts = dataset.mean()
            ax1.plot(ts.index.map(to_mdates), ts.values,
                     color=dataset.color, label=dataset.name)

            subsets = [subsets]  # Lo convierto en lista para facilitar su gestión.

        else:
            raise TypeError('Válidos solo str o list.')

        # Ploteo de la Piezometría ---------------------------------------------
        if piezo:

            # Un único piezómetro
            if self.piezos.columns.size == 1:
                ts = self.piezos.squeeze().copy()
                ax2.plot(ts.index.map(to_mdates), ts.values,
                    color='#41A2C4', label=ts.name)

            # Varios piezómetros
            else:
                for i in self.piezos.columns:
                    ts = self.piezos[i].squeeze().copy()
                    ax2.plot(ts.index.map(to_mdates), ts.values, label=ts.name)


            piezo_label = self._info_piezos['bd'].units
            ax2.set_ylabel(f"Piezometría ({piezo_label})", fontsize=15, labelpad=11)

        elif not piezo and (pluvi or other_sensors):
            ax2.set_yticks([])

        # Ploteo de la Precipitación -------------------------------------------
        if pluvi:
            ax2t = ax2.twinx()
            ts = self.estacion.copy()
            ax2t.plot(ts.index.map(to_mdates), ts.values,
                      '--', color='k', lw=0.6, label=ts.name)

            pluvi_label = self._info_estaciones['bd'].units
            ax2t.set_ylabel(f"Precipitación ({pluvi_label})",
                            fontsize=15, labelpad = 28, rotation=270)

        # Ploteo de otro sensor si procede -------------------------------------
        if other_sensors:
            ax3t = ax2.twinx()

            # Un único sensor
            if self.other_sensors.columns.size == 1:
                ts = self.other_sensors.squeeze().copy()
                ax3t.plot(ts.index.map(to_mdates), ts.values, ls='dashdot', label=ts.name,
                    color=self._info_other_sensors['bd'].color, lw=0.8)

            # Varios sensores
            else:
                for i in self.other_sensors.columns:
                    ts = self.other_sensors[i].squeeze().copy()
                    ax3t.plot(ts.index.map(to_mdates), ts.values, ls='dashdot',
                              label=ts.name, lw=0.8)

            if not pluvi:
                sensor_name = self._info_other_sensors['bd'].name.capitalize()
                sensor_label = self._info_other_sensors['bd'].units
                ax3t.set_ylabel(f"{sensor_name} ({sensor_label})",
                            fontsize=15, labelpad = 28, rotation=270)
            else:
                ax3t.axis('off')
            
        # Leyenda:
        handles = [i.get_legend_handles_labels()[0] for i in fig.axes]
        labels = [i.get_legend_handles_labels()[1] for i in fig.axes]
        handles = [j for i in handles for j in i] 
        labels = [j for i in labels for j in i]

        if len(fig.axes) == 1:
            ax1.legend(handles, labels)

        else:
            fig.axes[-1].legend(handles, labels,
                framealpha=1, borderpad=0.7, frameon=True,
                labelspacing=0.7, fontsize=13, loc='lower left',
                bbox_to_anchor=(0.25,-0.5,0.5,0.5,), ncol=4)

        # Formato final:
        ax1.set_title(f"Agregado {self._gdf['Agregado'].values[0]}",
            fontsize=24, pad=15)

        # ------------------- RÓTULO 'Y' DE LOS DATASETS -----------------------
        # Añadir unidades si los datasets indicados presentan las mismas
        d_same_units = len(set([self._tr_dict.get(i).units for i in subsets])) == 1
        if isinstance(subsets, str) or (isinstance(subsets, list) and d_same_units):
            dataset_label = f"Deformación ({list(self._tr_dict.values())[0].units})"
        else:
            dataset_label = 'Deformación'

        ax1.set_ylabel(ylabel=dataset_label, fontsize=15, labelpad=11)
        # ----------------------------------------------------------------------

        # Ajusta los límites de los subplots:
        x_min = min([i._x[0] for i in ax1.lines])
        x_max = max([i._x[-1] for i in ax1.lines])
        ax1.set_xlim(x_min, x_max)  

        ax = ax2 if piezo or pluvi else ax1
        ax.set_xlabel(None)

        # Customización del formato de las fechas del eje X.
        months = mdates.MonthLocator(interval=3)
        months_fmt = mdates.DateFormatter('%m')
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(months_fmt)

        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter('\n%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)

        ax.tick_params(axis="x", which="major", rotation=0)
        [i.label1.set_horizontalalignment('center') for i in ax.xaxis.get_major_ticks()]

        if savefig:
            fig.savefig(f"{self.name}.png")

        return ax1

    def mapa(self, dataset=None, column=None, basemap=True,
                   savefig=False, LayerControl=True, **kwargs):
        """ Función para la representación espacial de la información de el/los
        agregado/s. Ésta se realiza a través de la librería Folium  (si basemap=True)
        o Matplotlib (basemap=False), guardándose la imagen resultante (si savefig=True)
        como un archivo 'html' en el primer caso y 'png' en el segundo.
        
        Parámetros
        ----------
        dataset str o list. Default 'all', Opcional
            Datasets a representar. De cada uno se representa el promedio de la
            deformación (Dataset.mean()). Por defecto se representan todos los
            datasets ('all').

        column str default None, Opcional.
            Columna del geopandas.GeoDataFrame del dataset según la que se quieren
            graduar los puntos. Funciona de manera análoga

        basemap bool Default True, optional
            Adición de mapas base mediante Folium.
            
        savefig bool Default False. Opcional
            Guardado de la imagen en el directorio de trabajo.

        LayerControl: bool, Default True. Opcional
            Añade una leyenda al mapa cuando éste es generado mediante folium
            (ie, basemap=True). Cuando se quieren añadir más entidades a un mapa
            ya creado (objeto folium.Map), este argumento debe ser False en el
            primero de ellos.

            Ejemplo:
            >>> m = Doñana.get('Asc').mapa(LayerControl=False)
            >>> Doñana.get('Piezo_bbdd').mapa(m=m)
            
        **kwargs: optional
            Argumentos adicionales. Se pasan a la función 
            geopandas.GeoDataFrame.explore, que a su vez los envía a la función
            folium.Map (si basemap=True) o a geopandas.GeoDataFrame.plot si
            basemap=False y únicamente se representa un dataset.

        Devuelve
        -------
        objeto folium.Map (si basemap=True) o matplotlib.axes.Axes (False)
        
        Ejemplo
        -------
        >>> Doñana = wasar.example.get_model()
        >>> Doñana.get('P').mapa(['Moguer', 'Niebla'])
        
        """

        df = self._get_all_points()

        # ------------------------ COMPROBACIONES ------------------------------
        # Cambiar entrada:
        dataset = dataset[0] if isinstance(dataset, list) and  len(dataset) == 1 \
                             else dataset
        # Comprobar el tipo de datos de 'dataset'
        msg = "'{dataset}' no es de tipo str o list."
        assert isinstance(dataset, (list, str, type(None))), msg

        # Comprobar que existe el 'dataset' si se introdujo uno o varios
        if dataset:
            _ = [dataset] if isinstance(dataset, str) else dataset
            msg = "El nombre de algún dataset no es válido."
            assert set(_).issubset(self.datasets), msg

        # ¿Existe la columna 'column' en el dataset?
        if isinstance(dataset, str) and column and not column in df.columns:
            raise ValueError(f"La columna '{column}' no está presente en el "
                              "dataset '{dataset}'.")
        # ----------------------------------------------------------------------

        basemap = _checking_folium(basemap)
        legend = kwargs['legend'] if 'legend' in kwargs else True

        # **********************************************************************
        # Representación mediante folium
        # **********************************************************************
        if basemap:

            if 'm' in kwargs.keys():
                mymap = kwargs['m']
                msg = f"'{mymap}' no es un objeto de tipo folium.Map"
                assert isinstance(mymap, folium.Map), msg
            
            else:
                mymap =  _add_basemap(self.gdf)

            if not dataset:
                own_kwargs = dict(column='Dataset', tooltip='ID', popup=False,
                    legend=True, style_kwds=dict(stroke=False, opacity=0.75),           
                    name='Datasets', m=mymap, cmap=_get_datasets_cmap(df))

            else:

                df = df.set_index('Dataset').loc[dataset].reset_index()
                posible_name = ', '.join(dataset) 
                name = posible_name if isinstance(dataset, list) \
                       and len(posible_name) < 40 else 'Datasets'
                name = dataset if isinstance(dataset, str) else name

                if column:
                    own_kwargs = dict(legend=True, tooltip='ID', column=column,
                    cmap='Reds', style_kwds=dict(stroke=False, opacity=0.75),
                    name=name, m=mymap)
                else:
                    own_kwargs = dict(legend=True, tooltip='ID',
                        column='Dataset', style_kwds=dict(stroke=False,
                                                           opacity=0.75),
                        name=name, m=mymap, cmap=_get_datasets_cmap(df))
                           
            kwargs = _update_kwargs(kwargs, own_kwargs)
            df.explore(**kwargs)

            # Representación del agregado:
            self.gdf.explore(color=None, tooltip=False, style_kwds=dict(color='black',
                                                    fillColor=None, fill=False),
                             name=self.name, m=mymap)

            
            if LayerControl:
                folium.LayerControl().add_to(mymap)

            if savefig:
                mymap.save('Mapa.html')

            return mymap

        # **********************************************************************
        # Representación sin folium (a través de geopandas, i.e.: Matplotlib)
        # **********************************************************************
        else:

            _wasar_plot_params(deactivate=True)

            pol_style = dict(facecolor='None', ec='k', lw=0.8)

            # Plotting the polygon:
            ax = self.gdf.plot(**pol_style)

            # Todos los datasets o aquellos introducidos como lista
            if not dataset or isinstance(dataset, list):
                dataset = self.datasets if not dataset else dataset

                df = df.set_index('Dataset').loc[dataset].reset_index()
                df.plot(column='Dataset', cmap=_get_datasets_cmap(df),
                        legend=legend, ax=ax, alpha=0.5, ec=None)

                # handles, labels = ax.get_legend_handles_labels()
                # ax.legend(handles, labels)

            # Solo para type(dataset) == str
            else:
                if not column:
                    own_kwargs = dict(ax=ax, legend=legend, ec=None,
                                      color=self._tr_dict[dataset].color)
                else:
                    own_kwargs = dict(ax=ax, legend=legend, column=column,
                                      cmap='Reds', cax=_colorbar(ax), ec=None)

                kwargs = _update_kwargs(own_kwargs, kwargs)
                df.loc[df['Dataset']==dataset].plot(**kwargs)

            ax.axis('off')

            if isinstance(dataset, str) and column and not 'title' in kwargs:
                title = f"{self.name} - Columna: '{column}'"

            elif isinstance(dataset, str) and column and 'title' in kwargs:
                title = kwargs['title']

            else:
                title = f"{self.name}"

            ax.set_title(title, fontsize=18)

            if savefig:
                ax.figure.savefig('Mapa.png')

            _wasar_plot_params()

            return ax
    
    def _get_bd(self, name, which):
        """Método privado. Devuelve la base de datos de la materia (which) 
        solicitada si ésta existe. 'which' puede ser 'Piezo', 'Precipi', 'PS' u
        'Other'. Si no se especifica, se coge la primera que fue añadida. """

        bd_dic = self._backpack[which]

        bd_map = {'PS':'Dataset', 'Piezo':'Piezometria', 'Precipi':'Precipitacion',
                  'Other_bd':'Otras bds'}

        msg = (f"El modelo no presenta ninguna base de datos de {bd_map[which]}."
                "\nPuedes añadirla a través del método 'append'.\n"
                "Para más información: >>> help(wasar.parts)")
        assert len(bd_dic.values()) > 0, msg

        if name:
            assert name in bd_dic.keys(), f"Esa base de datos no existe."

        # Selección de la base de datos indicada. Si name es None, cojo la primera.
        B = bd_dic[name] if name else list(bd_dic.values())[0]

        return B

    def _get_all_points(self):
        """Método privado para la creación de un geopandas.GeoDataFrame que contenga
        los puntos de todos los datasets con información en este agreagdo."""
        dfs = [pd.DataFrame(i.gdf) for i in self._tr_dict.values()]
        return gpd.GeoDataFrame(pd.concat(dfs, axis=0))

    def wavelet(self, freq, *series, **kwargs):
        """ Método para el análisis wavelet univariante o bivariante de las series
        temporales asociadas a este agregado o de cualquier otra introducida como
        un objeto de pandas. Se basa en la clase Wavelet del módulo wasar.wavelet.

        Un correcto análisis wavelet requiere de una minuciosa configuración de
        los parámetros de partida, introducidos como **kwargs. Puede consultar
        las opciones disponibles y sus funcionalidades, respectivamente, en
        >>> help(wasar.wavelet)
        y
        https://cran.r-project.org/web/packages/WaveletComp/WaveletComp.pdf

        Parámetros
        ----------

        freq: pandas.DateOffset, pandas.Timedelta or str 
            Frecuencia de muestreo de la serie temporal a la que se remuestreará
            por si se diera el caso de que ésta fuese irregular. No altera los
            valores de la serie si ésta ya presenta un muestreo regular.
            Más info:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        *series: (*args) str, pandas.Series o pandas.DataFrame
            Cuando es str: Nombre de alguna de las bases de datos vinculadas al
                agregado o del dataset a analizar. Éstos pueden ser: 'piezo', 
                'precipi' o 'other_sensors' si se quiere analizar (respectivamente)
                el piezómetro, la estación o un sensor de otra base de datos
                vinculados al agregado. Si hay vinculados varios, se cogerá el primero.
                En este último caso, el orden de las series temporales introducidas
                importa cuando se quieren estudiar relaciones de causa y efecto.

            Cuando no es un str: serie(s) temporal(es) a analizar introducidas
                como un DataFrame o una Serie de pandas.

            Si únicamente se introduce una serie, se calcula la CWT (Continuous
            Wavelet Transform). Si se introducen dos, se calcula la XWT (Cross
            Wavelet Transform) y la WTC (Wavelet Transform Coherence).

        **kwargs: Otros argumentos disponibles. Consultar en wasar.wavelet
        
        Devuelve
        --------
        wasar.Wavelet

        Ejemplo
        -------
        1. Análisis de las frecuencias comunes entre el dataset Ascending y el 
        piezómetro más cercano del Agregado '2', considerando una frecuencia de
        muestreo mensual.

        >>> Doñana = wasar.example.get_model()
        >>> A2 = Doñana.agregado('2').set_piezo(way='nearest')
        >>> W = A2.wavelet('M','Asc','piezo')
        >>> W.image         # Para obtener las figuras resultantes.
        >>> W.used_params   # Para consultar los parámetros del análisis.
        >>> W.legend        # Consulta de la nomenclatura de la imagen obtenida.
        
        2. Análisis de las periodicidades principales de la serie temporal (con
        frecuencia diaria) de la estación pluviométrica más cercana al agregado 2.

        >>> A2 = Doñana.agregado('2').set_estacion(way='nearest')
        >>> W = A2.wavelet('D','precipi')

        """

        from .wavelet import Wavelet

        def _extract(name):
            """Función Interna. Extrae la serie temporale de la parte del modelo
            indicada. Devuelve una pandas.Series
            """

            def check_and_return(what, name):
                if not isinstance(what, (pd.Series, pd.DataFrame)):
                    raise SystemError(f"No hay '{name}' vinculado en este agregado")
                else:
                    return what

            if isinstance(name, (pd.Series, pd.DataFrame)):
                return name

            elif isinstance(name, str):

                if name.lower() == 'piezo':
                    return check_and_return(self.piezos, name)

                elif name.lower() == 'precipi':
                    return check_and_return(self.estacion, name)

                elif name.lower() == 'other_sensors':
                    return check_and_return(self.other_sensors, name)

                else: # Para los Datasets
                    try:
                        return self._tr_dict[name].mean()
                    except:
                        raise NameError(f"{name} no es válido. \n Opciones: piezo,"
                        "precipi, other_sensors o alguno de los nombres de los datasets. "
                         "Consulta los métodos del Modelo para acceder a ellos.")

            else:
                raise TypeError(f"{type(name)} no es un tipo de datos válido.")
        
        kwargs['_precipi'] = self.estacion

        if len(series) == 1:
            ts = _extract(series[0])
            return Wavelet(freq, ts, **kwargs)
        
        elif len(series) == 2:
            ts1 = _extract(series[0])
            ts2 = _extract(series[1])
            return Wavelet(freq, ts1, ts2, **kwargs)

        else:
            raise TypeError("Debes introducir un mínimo de una serie y un máximo "
                            "de dos.")

    def _info(self, _print=True):
        """ Método privado que devuelve la información del agregado. Públicamente
        utilizado como propiedad, imprimiendo en pantalla la información.

        _print bool, Default True
            Cuando es True imprime en pantalla el texto (devuelve NoneType), de
            lo contrario devuelve la información como un str.

        """

        from .plot import _b
        define_str = lambda x: ', '.join(x.columns.tolist())
        piezos = estacion = sensors = ''
        bd_piezo = bd_estacion = bd_sensors = ''

        # Definir las variables si no son nulas o son DataFrames vacíos.
        if isinstance(self.piezos , pd.DataFrame):
            piezo = define_str(self.piezos)
            bd_piezo = elf._info_piezos['bd'].name

        if isinstance(self.other_sensors , pd.DataFrame):
            sensors = define_str(self.other_sensors)
            bd_sensors = self._info_other_sensors['bd'].name

        if isinstance(self.estacion , pd.DataFrame):
            estacion = self.estacion.name
            bd_estacion = self._info_estaciones['bd'].name

        area = self._gdf.area.iloc[0]/10000
        datasets = [f"{i} ({len(self.subset(i).puntos)} puntos)"for i in self.datasets]

        texts = [
                f". {_b('Agregado')}: {self.name}",
                f". {_b('Area')}: {round(area, 3)} (ha)",
                f". {_b('Dataset(s)')} en este agregado: {', '.join(datasets)}",
                f". {_b('Piezo(s) asociado(s)')}: {piezos}",
                f"  Bd --> {bd_piezo}",
                f". {_b('Estación asociada')}: {estacion}",
                f"  Bd --> {bd_estacion}",
                f". {_b('Otro(s) sensor(es) asociado(s)')}: {sensors}",
                f"  Bd --> {bd_sensors}"
                ]

        if _print:
            print( '\n'.join(texts) )
        else:
            return '\n'.join(texts)

    info = property(_info)
