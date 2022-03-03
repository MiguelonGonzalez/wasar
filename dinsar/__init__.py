from .model import Model
from .parts import Dataset, Piezometria, Precipitacion, DataBase
from .wavelet import Wavelet, get_wavelet_legend
from .plot import _dinsar_plot_params
from .utils._utils import _importing_folium
import dinsar.example


_is_folium = _importing_folium()

__all__ = ["Model", "parts", "Wavelet", "example"]

_dinsar_plot_params()
