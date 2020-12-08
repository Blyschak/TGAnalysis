""" This module provides a class that is holding the TGA data """
import csv
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import linregress

R = 8.314


class TGAData:
    def __init__(self, input, heating_rate=1., t_units='K'):
        """ Create TGAData instance from the termogravimetric analysis data.
        """

        self.input = self.normalized_input(input)
        self.heating_rate = heating_rate
        self.t_units = t_units

    @staticmethod
    def from_csv(filepath):
        temps = []
        mass_losses = []
        with open(filepath) as fp:
            reader = csv.reader(fp)
            for row in reader:
                temp, mass_loss = row
                temps.append(float(temp))
                mass_losses.append(float(mass_loss))
        return TGAData(
            [np.array(temps), np.array(mass_losses)],
            heating_rate=1 / 6,
        )

    @property
    def temp(self):
        temp, alpha = self.input
        return temp

    @property
    def temp_in_kelvins(self):
        temp, alpha = self.input
        if self.t_units == 'C':
            return temp + 273.
        return temp

    @property
    def alpha(self):
        temp, alpha = self.input
        return alpha

    @staticmethod
    def normalized_input(input):
        temp, mass_loss = input
        while True:
            filter_map = [temp[i] > temp[i - 1] for i in range(1, temp.size)] + [True]
            if all(filter_map):
                break
            temp = temp[filter_map]
            mass_loss = mass_loss[filter_map]

        min_mass = np.min(mass_loss)
        max_mass = np.max(mass_loss)
        return np.array([
            temp,
            (np.array(max_mass) - mass_loss) / np.array(max_mass - min_mass)
        ])

    def range(self, t_min, t_max):
        boolean_mask = (t_min <= self.temp) & (self.temp <= t_max)
        data_range = np.array([self.temp[boolean_mask], -self.alpha[boolean_mask]])
        return TGAData(data_range, self.heating_rate, t_units=self.t_units)

    def interpolated(self):
        return UnivariateSpline(self.temp, self.alpha, k=5, s=10000)

    def _derivative(self):
        interpolation = self.interpolated()
        return interpolation.derivative(n=1)

    def derivative(self):
        return savgol_filter(self.alpha, window_length=7, polyorder=2, deriv=1) / \
               savgol_filter(self.temp, window_length=7, polyorder=2, deriv=1)

    def derivative_in_kelvin(self):
        return savgol_filter(self.alpha, window_length=7, polyorder=2, deriv=1) / \
               savgol_filter(self.temp_in_kelvins, window_length=7, polyorder=2, deriv=1)

    def temp_for_alpha_interpolated(self, alpha):
        return UnivariateSpline(self.alpha, self.temp)(alpha)

    def ranges(self):
        der = self._derivative()(self.temp)

    @classmethod
    def from_table(cls, temps, alphas, t_units='K', heat_rate=1.):
        return TGAData([np.array(temps), np.array(alphas)],
                       t_units=t_units,
                       heating_rate=heat_rate)


@dataclass
class ModelFittingResult:
    Ea: float
    lnA: float
    rvalue: float
    stderr: float
    model: Any

    @property
    def delta_Ea(self):
        return self.stderr

    @property
    def delta_lnA(self):
        return (self.stderr / self.Ea) * self.lnA


class SimpleOrderModel:
    def __init__(self, n):
        self.n = n

    @property
    def name(self):
        return f'{self.n} order model'

    def __call__(self, alpha):
        return (1 - alpha) ** self.n


class TwoDimDiffusionModel:
    @property
    def name(self):
        return f'2-dim diffusion model'

    def __call__(self, alpha):
        return -1 / np.log(1 - alpha)


class ThreeDimDiffusionModel:
    @property
    def name(self):
        return f'3-dim diffusion model'

    def __call__(self, alpha):
        return 1.5 * ((1 - alpha) ** (2 / 3)) / (1 - (1 - alpha) ** (1 / 3))


class ModelFitting:
    def __init__(self):
        pass

    def fit_model(self, tga, model):
        slope, intercept, rvalue, pvalue, stderr = linregress(*self.fitting_xy(tga, model))
        # print(slope, intercept, rvalue, pvalue, stderr)
        return ModelFittingResult(Ea=slope, lnA=intercept, model=model,
                                  rvalue=rvalue, stderr=stderr)

    def fit_model(self, dtg, model):
        slope, intercept, rvalue, pvalue, stderr = linregress(*self.fitting_xy(tga, model))
        # print(slope, intercept, rvalue, pvalue, stderr)
        return ModelFittingResult(Ea=slope, lnA=intercept, model=model,
                                  rvalue=rvalue, stderr=stderr)

    @staticmethod
    def models():
        yield SimpleOrderModel(1)
        yield SimpleOrderModel(2)
        yield SimpleOrderModel(3)
        yield TwoDimDiffusionModel()
        yield ThreeDimDiffusionModel()

    def fit(self, tga):
        results = {}
        for model in self.models():
            results[model.name] = self.fit_model(tga, model)
        return results

    @staticmethod
    def fitting_xy(tga, model):
        # [ln(da/dT) + ln(beta) - ln(model(a))] vs -1/RT
        #
        ##########################################################################
        # [ln(da/dT) + ln(beta) - ln(model(a))] = INTERCEPT + SLOPE * (-1/RT)
        # INTERCEPT = lnA
        # SLOPE = Ea
        ##########################################################################
        #
        X = -1 / (R * tga.temp_in_kelvins)
        Y = np.log(tga.derivative_in_kelvin() * tga.heating_rate / model(tga.alpha))
        # print(X)
        # print(Y)
        # print(tga.alpha)
        # print(tga.derivative_in_kelvin())
        # filter out NaN values that might happen with real data
        filter_mask = ~np.isnan(Y) & ~np.isinf(Y)
        fitting_tuple = (X[filter_mask], Y[filter_mask])
        return fitting_tuple
