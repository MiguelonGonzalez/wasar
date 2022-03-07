"""

Módulo privado para la ejecución de Model.porLotes.

"""
import os
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib as mpl

from ..plot import _b, _wasar_plot_params
from ._utils import _filtro_agregados, _update_kwargs


def trier(func):
    """Decorador para intentar cada operación sin bloqueo en la ejecución"""
    @wraps(func)
    def inner_func(self, *args, **kwargs):

        if not self.raise_errors:

            try:
                return func(self, *args, **kwargs)

            except Exception as e:
                print(f"Apareció un {_b('error')} durante '{func.__doc__}' del "
                      f"agregado {_b(self.agregado.name)}.\n\t{_b('Error')}:\t{e}")
        else:
            try:
                return func(self, *args, **kwargs)

            except:
                raise

    return inner_func


class _Lotes:
    """ Clase privada para la ejecución del método 'porLotes' de wasar.Model.
        Las definición de los parámetros de entrada de esta clase figuran en
        el __doc__ de Model.porLotes.
    """
    def __init__(self, Model, agregados='all', 
                set_piezo=True, set_estacion=True, set_other_bd=False,
                piezo_kw=None, estacion_kw=None, other_bd_kw=False,
                mapa=True, plot=True, mapa_kw=None, plot_kw =None,
                save_arrays=True, 
                cwt=False, xwt=False, cwt_kw=None, xwt_kw=None,
                verbose=False, raise_errors=False):

        self.Model = Model
        self.save_arrays = save_arrays

        self.set_piezo = set_piezo
        self.set_estacion = set_estacion
        self.set_other_bd = set_other_bd

        self.mapa = mapa
        self.plot = plot

        self.piezo_kw = piezo_kw
        self.estacion_kw = estacion_kw
        self.other_bd_kw = other_bd_kw
        self.mapa_kw = mapa_kw    
        self.plot_kw = plot_kw

        self.cwt = cwt
        self.xwt = xwt
        self.cwt_kw = cwt_kw
        self.xwt_kw = xwt_kw

        self.verbose = verbose
        self.raise_errors = raise_errors

        # Comprobación del tipo de datos de los argumentos:
        for num, arg in enumerate((save_arrays, mapa, plot,
                                   set_piezo, set_estacion, cwt, xwt, verbose)):
            if not isinstance(arg, bool):
                raise TypeError(f"Alguno de los argumentos no presenta el tipo "
                                 "de datos adecuado.")

        for num, arg in enumerate((mapa_kw, plot_kw, piezo_kw,
                                   estacion_kw, cwt_kw, xwt_kw)):
            if arg and not isinstance(arg, dict):
                raise TypeError(f"El objeto introducido no es un diccionario "
                                 "en alguno de los kwargs.")

        # Identificación de los agregados a estudiar:
        if isinstance(agregados, str) and agregados.lower() == 'all':
            self.agregados = Model.agregados

        elif isinstance(agregados, str):
            _filtro_agregados(agregados, Model.agregados)
            self.agregados = [agregados]

        elif isinstance(agregados, list):
            _filtro_agregados(agregados, Model.agregados)
            self.agregados = agregados

        else:
            raise TypeError(f"{type(agregados)} no es un tipo de datos válido.")

        
        self.main()


    def main(self):

        self.base_folder = self._new_folder('', first=True)

        # Ejecución de todos los agregados -------------------------------------
        try:
            from tqdm import tqdm
            [self.execute(agregado) for agregado in tqdm(self.agregados)]

        except:
            [self.execute(agregado) for agregado in self.agregados]

        # Metadata del Modelo --------------------------------------------------
        if self.verbose:
            with open(f"{self.base_folder }Metadata.txt", 'w') as f:
                info = self.Model._info(_print=False).replace(
                                  "\033[1m",'').replace("\033[0m",'')
                f.write(info)

        print(f"{_b('Fin de la ejecución')}")

    def execute(self, agregado):
        """Funciones que se ejecutan en cada agregado. Bloque código del bucle"""

        try:
            self.agregado = agregado = self.Model.agregado(agregado, _lotes=True)
        except Exception as e:
            print(f"No se pudo analizar el Agregado {agregado}. "
                  f"{_b('Error')}:\n\t{e}")
            return None

        # Generación de una carpeta por agregado.
        path = f"{self.base_folder}{agregado.name}"
        self.path = self._new_folder(path)

        # Ejecución ------------------------------------------------------------
        self.exe_piezo() if self.set_piezo else None
        self.exe_estacion() if self.set_estacion else None
        self.exe_other_bd() if self.set_other_bd else None
        self.exe_save_arrays() if self.save_arrays and (self.set_piezo 
                            or self.set_estacion
                            or self.set_other_bd) else None
        self.exe_images(self.mapa, self.plot)
        self.exe_cwt() if self.cwt else None
        self.exe_xwt() if self.xwt else None
        # ----------------------------------------------------------------------

        if self.verbose:
            with open(f"{self.path}Metadata.txt", 'w') as f:
                info = agregado._info(_print=False).replace("\033[1m",'').replace("\033[0m",'')
                f.write(info)

        plt.close()

    @trier
    def exe_piezo(self):
        """La fijación de la piezometría"""
        piezo_kw = self.piezo_kw
        piezo_kw = dict() if not piezo_kw else piezo_kw
        piezo_kw['_silent'] = True
        self.agregado.set_piezo(**piezo_kw)

    @trier
    def exe_estacion(self):
        """La fijación de la pluviometría"""
        estacion_kw = self.estacion_kw
        estacion_kw = dict() if not estacion_kw else estacion_kw
        estacion_kw['_silent'] = True
        self.agregado.set_estacion(**estacion_kw)

    @trier
    def exe_other_bd(self):
        """La fijación de otro sensor"""
        other_bd_kw = self.other_bd_kw
        other_bd_kw = dict() if not other_bd_kw else other_bd_kw
        other_bd_kw['_silent'] = True
        self.agregado.set_other_bd(**other_bd_kw)

    @trier
    def exe_save_arrays(self):
        """El guardado de las series temporales"""
        kwargs = dict(sep='\t', float_format='%.3f')
        
        if self.set_piezo:
            self.agregado.piezos.to_csv(f"{self.path}Piezometro.txt", **kwargs)
        
        if self.set_estacion:
            self.agregado.estacion.to_csv(f"{self.path}Precipitacion.txt", **kwargs)

        if self.set_other_bd:
            name = self.agregado._info_other_sensors['bd'].name
            self.agregado.other_sensors.to_csv(f"{self.path}{name}.txt", **kwargs)

    @trier
    def exe_images(self, mapa, plot):
        """El cálculo de las imágenes del Agregado"""

        # MAPA
        mapa_kw = dict() if not self.mapa_kw else self.mapa_kw

        if mapa:
            mapa_object = self.agregado.mapa(**mapa_kw)

            # Sin folium
            if isinstance(mapa_object, mpl.axes._axes.Axes):
                mapa_object.figure.savefig(f"{self.path}Agregado.png",
                                           bbox_inches='tight', pad_inches=0.5)
            # Con folium
            else:
                mapa_object.save(f"{self.path}Agregado.html")

        # PLOT
        plot_kw = dict() if not self.plot_kw else self.plot_kw
        own_plot_kwargs = dict(piezo=self.set_piezo,
                               pluvi=self.set_estacion,
                               other_sensors=self.set_other_bd)

        plot_kw = _update_kwargs(own_plot_kwargs, plot_kw)

        if plot:
            self.agregado.plot(**plot_kw).figure.savefig(
                f"{self.path}Series.png", bbox_inches='tight', pad_inches=0.5)

        plt.close()

    @trier
    def exe_cwt(self):
        """El cálculo de la CWT"""

        freq, series, kwargs = self._check_wavelet(self.cwt_kw)

        for num, serie in enumerate(series):

            images = self.agregado.wavelet(freq, serie, **kwargs).image
            savefig_kw = dict(bbox_inches='tight', pad_inches=0.5)

            # Guardar imagen:
            name = serie if isinstance(serie, str) else num

            # CWT
            images[0].savefig(f"{self.path}CWT - {name}.png", **savefig_kw)
            # Power AV
            images[1].savefig(f"{self.path}Power - {name}.png", **savefig_kw)

    @trier
    def exe_xwt(self):
        """El cálculo de la XWT y WTC"""

        freq, series, kwargs = self._check_wavelet(self.xwt_kw)
        assert len(series) == 2, (f"'series' de xwt_kw debe albergar una lista "
                                          "con únicamente dos series temporales.")

        Wavelet = self.agregado.wavelet(freq, *series, **kwargs)

        # Guardar imagen:
        name = [i if isinstance(i, str) else 'Variable' for i in series]
        Wavelet.image.figure.savefig(f"{self.path}Cross - {', '.join(name)}.png",
                                    bbox_inches='tight', pad_inches=0.5)

    def _new_folder(self, name, first=False):
        
        if first:
            path = "Por Lotes"

        else:
            path = name

        if not os.path.exists(path):
            os.mkdir(path)
            
            return f"{path}/"

        else:
            contador=1 
            new_path = f"{path} - copia ({contador})"
            while os.path.exists(new_path):
                contador+=1
                new_path = f"{path} - copia ({contador})"

                if contador==20:
                    raise RuntimeError(
                        "Alcanzado el máximo número de carpetas con el mismo "
                        "nombre. Borra las que no sirvan.")

            os.mkdir(new_path)

            return f"{new_path}/"


    def _check_wavelet(self, param):

        try:
            freq, ts = param['freq'], param['series']

            if isinstance(ts, str):
                ts = [ts]

            elif not isinstance(ts, (str, list)):
                raise TypeError("'series' debe ser str o list.")

        except:
            raise KeyError(f"El diccionario cwt no existe o no contiene las claves "
                            "requeridas: 'freq' y 'series'.")

        kwargs = param.copy()
        [kwargs.pop(i) for i in ('freq', 'series')]

        return freq, ts, kwargs
