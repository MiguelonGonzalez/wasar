"""
Módulo que contiene la clase Wavelet, principalmente accesible a través del
método 'agregado' (devuelve un objeto de tipo 'Agregado') del modelo, aunque 
también se puede usar directamente desde la clase Wavelet de este módulo.

Las herramientas wavelet permiten estudiar las principales frecuencias de series
temporales, tanto univariantes como bivariantes. Entre las principales transformaciones
wavelet se encuentra la CWT (Continuous Wavelet Transform), la XWT (Cross Wavelet
Transform) y la WTC (Wavelet Transform Coherence). Mientras que la primera (CWT)
permite la identificación de periodicidades intermitentes de series unidimensionales,
las dos últimas son de utilidad cuando se quiere estudiar las frecuencias comunes
entre dos series temporales (estudio bivariante).

Este módulo permite calcular las tres transformaciones anterioes, lo cual lo realiza
por medio de R a través del paquete WaveletComp. Tanto los fundamentos teóricos 
sobre los que se apoya el paquete como la documentación del mismo se pueden encontrar
respectivamente en los siguiente enlaces:

(*) http://www.hs-stat.com/projects/WaveletComp/WaveletComp_guided_tour.pdf

(**) https://cran.r-project.org/web/packages/WaveletComp/WaveletComp.pdf

El uso de WaveletComp requiere tener instalado -además del lenguaje de programación
R- el paquete rpy2, el cual actúa como puente entre Python y R.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, tempfile, shutil

from .utils._utils import _update_kwargs
from .plot import dinsar_plot_params

try:
    from rpy2.robjects import r
except (ImportError, ModuleNotFoundError):
    raise ImportError("No se encontrón el paquete rpy2. Prueba a instalarlo de "
                      "la siguiente manera:\npip install rpy2\n"
                      "Más información: https://rpy2.github.io/doc.html")

def get_wavelet_legend():
    """Devuelve una imagen con la leyenda de la XWT y WTC."""
    dinsar_fname = os.path.realpath(__file__)
    fname = dinsar_fname.replace('wavelet.py','utils\\Legend.png')
    
    fig, ax = plt.subplots()
    pic = plt.imread(fname)
    ax.imshow(pic)
    ax.axis('off')
    return ax

class Wavelet:
    """Calcula la CWT cuando se introduce una serie temporal y la XWT y WTC cuando
    se introducen dos. La correcta ejecución de estas operaciones requiere
    que las series temporales introucidas presenten medidas a intervalos de tiempo
    regulares, lo que exige la definición del parámetro 'freq'.

    Parámetros
    ----------

    freq: pandas.DateOffset, pandas.Timedelta or str 
        Frecuencia de muestreo de la serie temporal a la que se remuestreará
        por si se diera el caso de que ésta fuese irregular. No altera los valores
        de la serie si ésta ya presenta un muestreo regular.
        Más info:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    *series: (*args) pandas.Serie o pandas.DataFrame
        Series temporales a analizar. Requiere un mínimo de una y un máximo de dos.
        Si únicamente se introduce una serie, se calcula la CWT (Continuous
        Wavelet Transform). Si se introducen dos, se calcula la XWT (Cross
        Wavelet Transform) y la WTC (Wavelet Transform Coherence) para el periodo
        común entre ambas. En este último caso, el orden de las series
        temporales introducidas importe cuando se quieren estudiar relaciones
        de causa y efecto.

    **kwargs: Otros argumentos disponibles:

        'method' (de pandas.DataFrame.interpolate). Default 'linear'.
            Método de interpolación para el cálculo de los posibles valores NaN
            generados durante el remuestreo. Más info:
            >>> help(pd.DataFrame.resample)
        
        Parámetros para el análisis wavelet:

        _Comunes a las funciones CWT(analyze.wavelet) y XWT-WTC (analyze.coherency)
         de WaveletComp (consultar funcionalidades en **):

        'dt': 12/31
        'dj': 1/20,
        'lowerPeriod': 1
        'upperPeriod': 30
        'loess_span': 0.75
        'n_sim': 100
        'method': 'white.noise'
        'make_pval': 'TRUE'
        'params': 'NULL'

        _Exclusivos para estudio bivariante (XWT-WTC):

        'window_type_t':1,
        'window_type_s':1,
        'window_size_t':5,
        'window_size_s':1/4

    """
    
    def __init__(self, freq, *series, **kwargs):
        """
        kwargs: passed to pandas.Series.interpolate """

        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['axes.linewidth'] = 0.2
        
        self.original_ts = series
        self.freq = freq
        self.kwargs = kwargs
        
        # CALCULAR CWT
        if len(series) == 1:
            A = series[0]
            self.A = self._regularize( self._check_Series(A) )
            self.fig = self._cwt(**kwargs)
        
        # CALCULAR XWT y WTC
        elif len(series) == 2:
            A, B = series[0], series[1]
            self.A, self.B = [self._regularize( self._check_Series(i) ) for i in (A,B)]
            self._overlap_period()
            self.fig = self._xwt(**kwargs)
        
        else:
            raise TypeError("La función está limitada a un mínimo de un argumento "
                            " y a un máximo de dos.")
        

        # Borrar carpeta temporal:
        shutil.rmtree(self.tempdir)

        # A los valores parámetros por defecto de matplotlib en dinsar:
        dinsar_plot_params()

    def _check_Series(self, ts):
        """Comprueba que la serie temporal (ts) sea del tipo pd.Series. Si es de
        tipo pandas.DataFrame, coge la primera columna."""

        if isinstance(ts, pd.Series):
            return ts
        
        elif isinstance(ts, pd.DataFrame):
            return ts.iloc[:,0].squeeze()
        else:
            raise TypeError(f"{type(ts)} no es una Series o un DataFrame de pandas.")
            
    def _regularize(self, ts):
        """Remuestrea la serie temporal a intervalos regulares según la frecuencia
        introducida (self.freq) y rellena los posibles valores NaN generados a
        través de una interpolación (por defecto lineal, aunque modificable a 
        través de los **kwargs de la clase."""

        method = self.kwargs['method'] if 'method' in self.kwargs.keys() else 'linear'

        return ts.resample(self.freq).mean().interpolate(method=method).dropna()
    
    @property
    def series(self):
        """Devuelve la primera serie temporal introducida (A) o las dos (A y B)."""
        try:
            return self.A, self.B
        except:
            return self.A

    @property
    def image(self):
        """ Devuelve la imagen Wavelet como objeto matplotlib.pyplot.figure."""
        return self.fig

    @property
    def legend(self):
        """Devuelve una imagen con la leyenda de la XWT y WTC."""
        return get_wavelet_legend()

    @property
    def used_params(self):
        """Devuelve un diccionario con los parámetros usados en el análisis"""
        return self._all_params

    def _overlap_period(self):
        """Calcula el periodo común entre la dos series temporales. Sobre ese periodo 
        común se analizarán en conjunto ambas series temporales mediante Wavelts."""
        
        common_index = pd.Series(np.intersect1d(self.A.index, self.B.index))
        if not common_index.empty:
            self.A, self.B = self.A.loc[common_index], self.B.loc[common_index]
        else:
            raise RuntimeError("Las series temporales no presentan un periodo de"
                               "tiempo cómun.")
        self._check_index()

    def _check_index(self):
        """Check if both time series have the same index"""
        if not (self.A.index == self.B.index).all():
            raise RuntimeError("Las series temporales no tienen el mismo índice...")
    
    def _cwt(self, **kwargs):
        """
        Genera la CWT de la serie temporal entrante en la clase (A).

        _cwt_params: Diccionario con varios parámetros modificables de la función
        analyze.wavelet, de WaveletComp.

        **kwargs: 
            dt, dj, lowerPeriod, upperPeriod, loess.span, n.sim, method, make.pval,
            params

        Devuelve un objeto de matplotlib.pyplot.figure

        """
        
        # Crear directorio temporal:
        self.tempdir = tempfile.mkdtemp(suffix='_wavelets')

        # Guardar la serie temporal
        self.A.to_csv(f"{self.tempdir}\\ts.txt", header=None, date_format="%Y-%m-%d")

        # Parámetros por defecto
        _cwt_own_params = {
        'dt':12/31,  # Every-12-days frequency.
        'dj':1/20,
        'lowerPeriod':1,
        'upperPeriod':30,
        'loess_span':0.75,
        'n_sim':100,
        'method':'white.noise',
        'make_pval':'TRUE',
        'params':'NULL'
                         }

        # Kwargs handling:

        try:
            kwargs.pop('_precipi')
        except:
            pass

        all_params = self._all_params = _update_kwargs(_cwt_own_params, kwargs)

        # Importación del script de R como texto y definición de variables:
        text = _cwt_func.format(temp_dir=self.tempdir.replace('\\','/'),
                                dt=all_params['dt'],
                                dj=all_params['dj'],
                                lowerPeriod=all_params['lowerPeriod'],
                                upperPeriod=all_params['upperPeriod'],
                                loess=all_params['loess_span'],
                                sim=all_params['n_sim'],
                                method=f"'{all_params['method']}'",
                                make_pval=all_params['make_pval'],
                                params=all_params['params'])

        # EJECUCIÓN EN R -------------------------------------------------------

        r(text)

        # Representación gráfica -----------------------------------------------

        # CWT
        fig1 = plt.figure(figsize=(15,10))
        ax1 = fig1.add_axes([0.15, 0.54, 0.7, 0.52]) 
        ax2 = fig1.add_axes([0.19, 0.45, 0.6, 0.2]) # left, bottom, width, height

        image = plt.imread(f"{self.tempdir}/CWT.png")
        ax1.imshow(image)
        ax1.axis('off')
        self.A.plot(ax=ax2)

        # Average Power
        fig2, ax21 = plt.subplots(figsize=(15,20))
        image = plt.imread(f"{self.tempdir}/Power.png")
        ax21.imshow(image)
        ax21.axis('off')

        return (fig1, fig2)

    def _xwt(self, **kwargs):
        """
        Genera la XWT y WTC de las series temporales entrantes (A y B).

        **kwargs: Diccionario con varios parámetros modificables de la función
                analyze.coherency, de WaveletComp.

        Parámetros posibles:
        dt, dj, lowerPeriod, upperPeriod, loess.span, n.sim, method, make.pval,
        params, window_type_t, window_type_s, window_size_t, window_size_s

        Devuelve un objeto de matplotlib.pyplot.figure

        """
    
        # Crear directorio temporal:
        self.tempdir = tempfile.mkdtemp(suffix='_wavelets')

        # Guardar la serie temporal
        csv_params = dict(header=None, date_format="%Y-%m-%d")
        self.A.to_csv(f"{self.tempdir}\\ts1.txt", **csv_params)
        self.B.to_csv(f"{self.tempdir}\\ts2.txt", **csv_params)

        # Parámetros por defecto
        _xwt_own_params = {
        'dt':1,
        'dj':1/20,
        'lowerPeriod':2,
        'upperPeriod':30,
        'loess_span':0.75,
        'n_sim':100,
        'method':'white.noise',
        'make_pval':'TRUE',
        'params':'NULL',
        'window_type_t':1,
        'window_type_s':1,
        'window_size_t':5,
        'window_size_s':1/4
                         }

        # Kwargs handling:
        try:
            kwargs.pop('_precipi')
        except:
            pass

        all_params = self._all_params = _update_kwargs(_xwt_own_params, kwargs)

        # Importación del script de R como texto y definición de variables:
        text = _xwt_func.format(temp_dir=self.tempdir.replace('\\','/'),
                                dt=all_params['dt'],
                                dj=all_params['dj'],
                                lowerPeriod=all_params['lowerPeriod'],
                                upperPeriod=all_params['upperPeriod'],
                                loess=all_params['loess_span'],
                                sim=all_params['n_sim'],
                                method=f"'{all_params['method']}'",
                                make_pval=all_params['make_pval'],
                                params=all_params['params'],
                                window_type_t={all_params['window_type_t']},
                                window_type_s={all_params['window_type_s']},
                                window_size_t={all_params['window_size_t']},
                                window_size_s={all_params['window_size_s']})


        # EJECUCIÓN EN R:
        r(text)

        # PLOT -----------------------------------------------------------------
    
        fig = plt.figure(figsize=(15,15), dpi=300)
        ax1 = fig.add_axes([0.15, 0.54, 0.7, 0.52]) # left, bottom, width, height
        ax2 = fig.add_axes([0.1635, 0.5, 0.658, 0.142])
        ax_d = ax2.twinx()
        axis = ax2, ax_d
        
        # Importación y pegado de la XWT generada con R
        image = plt.imread(f"{self.tempdir}/XWT.png")
        ax1.imshow(image)

        self.A.plot(ax=ax2, label='A', color='#CC3B54')
        self.B.plot(ax=ax_d, label='B',color='#31669E')

        # Try to plot precipitacion:
        try:
            pp = self.kwargs['_precipi']
            comm = pd.Series(np.intersect1d(self.A.index, pp))
            if not comm.empty:
                pp = pp.loc[comm]

            ax_p = ax2.twinx()
            axis = ax2, ax_d, ax_p
            [ax_p.spines[axis].set_linewidth(0.1) for axis in ['top','bottom','left','right']]
            ax_p.spines["right"].set_position(("axes", 1.07))

            x_min, x_max = self.A.index[0], self.A.index[-1]
            pp.plot(ax=ax_p, label=pp.name, style='--', color='gray',
                    xlim=(x_min, x_max))
            handles3, labels3 = ax_p.get_legend_handles_labels()

            print('Se añadió la precipitación para mejorar la interpretación.')

        except:
            axis = ax2, ax_d
       
        [ax2.spines[axis].set_linewidth(0.1) for axis in ['top','bottom','left','right']]
        [ax_d.spines[axis].set_linewidth(0.1) for axis in ['top','bottom','left','right']]

        [i.tick_params(length=1.5 , width=0.5, pad=0.15) for i in axis]
        ax2.tick_params(which='major', length=4)
        # Rótulos de eje:

        fig.text(0.13, 0.875, r'$\bf{XWT}$',**dict(fontsize=12, rotation=90))
        fig.text(0.13, 0.72, r'$\bf{WTC}$',**dict(fontsize=12, rotation=90))

        # Legends:
        handles1, labels1 = ax2.get_legend_handles_labels()
        handles2, labels2 = ax_d.get_legend_handles_labels()

        try:
            handles, labels = handles1 + handles2 + handles3, labels1 + labels2 + labels3
        except:
            handles, labels = handles1 + handles2, labels1 + labels2

        ax2.legend(handles, labels, loc='lower center', frameon=False,
                    ncol=3, bbox_to_anchor=(0.48,-0.5,0.1,0.1))

        ax1.axis('off')
        fig.subplots_adjust(hspace=0, bottom=0.3)
        ax2.grid(axis='both', which = 'both')

        return fig


#*******************************************************************************
#************--------SCRIPTS WAVELET PARA EJECUTAR DESDE R----------************
#*******************************************************************************

_cwt_func = """
library(WaveletComp)
Sys.setlocale("LC_TIME", "English")

path <- '{temp_dir}'
my.data <- read.delim(paste(path, '/ts.txt', sep=''), sep = ',', header = FALSE)
df <- data.frame(my.data)

# converting to datetime object
df['V1'] <- as.POSIXct(df[['V1']], format = "%Y-%m-%d")

ticks <- seq(head(df$V1,1), tail(df$V1,1),
             by = "month")
labels <- strftime(ticks, "%b-%y")

#tail(df,1)[1,1]
cwt <- analyze.wavelet(
    df, 'V2',
    dt={dt},
    dj={dj},
    lowerPeriod={lowerPeriod},
    upperPeriod={upperPeriod},
    loess.span={loess},
    n.sim={sim},
    method={method},
    make.pval={make_pval},
    params={params}
                      )

#IMAGEN CWT:
png(filename=paste(path,"/CWT.png",sep=''), width=14, height=6,
    units='in', res=1000, pointsize=12)

# To plot the wavelet power spectrum:
lista <- list(lab= "Square root of wavelet power levels",
              mar=4.7, label.digits=2)


power <- wt.image(cwt, n.levels = 250, exponent = 0.5,legend.params = lista,
                  timelab = "", periodlab = "Period",
                  color.key = "quantile", main ='CWT',
                  show.date = TRUE, date.format = "%F %T",
                  spec.time.axis = list(at=ticks, labels = labels, las = 2),
                  timetcl = -0.5, periodtck = 1, periodtcl = NULL)

graphics.off()    # Cerrar Graphical Device 2 R porque se queda bloqueado

#IMAGEN AVERAGE POWER:
png(filename=paste(path,"/Power.png",sep=''), width = 14, height = 6,
    units = 'in', res = 1000, pointsize = 12)

wt.avg(cwt, my.series='cwt', main='Wavelet Power Spectrum')

dev.off()

"""

#########################################################################

_xwt_func = """
library(WaveletComp)
Sys.setlocale("LC_TIME", "English")

path <- '{temp_dir}'

# Abrir coherentemente los archivos de la carpeta 'temp':
my.data1 <- read.delim(paste(path, '/ts1.txt', sep=''), sep = ',',
                       header = FALSE)
df1 <- data.frame(my.data1)

my.data2 <- read.delim(paste(path, '/ts2.txt', sep=''), sep = ',',
                       header = FALSE)

df2 <- data.frame(my.data2)

# converting to datetime object
df1['V1'] <- as.POSIXct(df1[['V1']], format = "%Y-%m-%d")
df2['V1'] <- as.POSIXct(df2[['V1']], format = "%Y-%m-%d")


#ticks <- seq(head(df$V1,1), tail(df$V1,1),
 #            by = "month")

#labels <- format(ticks, "%b-%y")

# Unir DataFrame en uno para poder aplicar WTC.
my.whole.data <- merge(df1,df2, "V1")
colnames(my.whole.data)[1] <- 'date'
names(my.whole.data)[2] <- "x"  # X: Subsidencia
names(my.whole.data)[3] <- "y"  # Y: Piezometría


# CROSS WAVELET TRANSFORM (XWT):
my.wc <- analyze.coherency(my.whole.data, c("x","y"),
         dt={dt},
         dj={dj},
         lowerPeriod={lowerPeriod},
         upperPeriod={upperPeriod},
         loess.span={loess},
         n.sim={sim},
         method={method},
         make.pval={make_pval},
         params={params},
         window.type.t={window_type_t},
         window.type.s={window_type_s},
         window.size.t={window_size_t},
         window.size.s={window_size_s}
                          )


png(filename=paste(path,"/XWT.png",sep=''), width = 14, height = 6, units = 'in',
    res = 1000, pointsize = 12)

################################################################################
# Fusionar ambas transformaciones (XWT y WTC) en una sola imagen:
par.defaults <- par(no.readonly=TRUE)
par(par.defaults)
par(c("PRUEBA TÍTULO"), mfrow=c(2,1), oma=c(0,1.1,0,0)+0.2, mar= c(0,0,0.3,0.3)+0.1)
    # oma = c(6,10,2,10), mar =c)
################################################################################
#                                PLOT
wc.image(my.wc,which.image="wp",
         n.levels = 250,
         exponent = 0.5,
         legend.params = list(lab = "Sqrt of XWT power levels",mar=2,label.digits = 1),
         # periodlab = "Period",
         color.key = "quantile", 
         spec.time.axis = FALSE)
         
# WAVELET COHERENCE:
# Crear 'guardador' en R para guardar la imagen siguiente:

wc.image(my.wc, which.image = "wc",
         color.key = "quantile",
         n.levels = 250,
         legend.params = list(lab="WTC",mar=2,label.digits=0,n.ticks=2)
         )

dev.off()
"""
