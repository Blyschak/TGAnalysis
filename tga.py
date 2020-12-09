""" This module provides a class that is holding the TGA data """

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import linregress

R = 8.314


@dataclass(frozen=True)
class TemperatureRange:
    T_first: float
    T_second: float


class TGA:
    """ This holds the TGA analysis data. """

    def __init__(self,
                 temp,
                 mass_loss,
                 mass_loss_derivative,
                 heating_rate=1.,
                 t_units='K'):
        """ Create TGAData instance from the termogravimetric analysis data """

        self.temp = np.array(temp)
        self.mass_loss = np.array(mass_loss)
        self.mass_loss_derivative = np.array(mass_loss_derivative)
        self.heating_rate = heating_rate
        self.t_units = t_units

        min_mass = np.min(self.mass_loss)
        max_mass = np.max(self.mass_loss)

        # convert mass loss to alpha of conversion
        self.alpha = (np.array(max_mass) - self.mass_loss) / \
                     np.array(max_mass - min_mass)

        # convert DTG data to d(alpha)/dT
        self.dtg = - self.mass_loss_derivative / \
                   np.array(max_mass - min_mass)

    @property
    def temp_in_kelvins(self):
        """ Returns temperature array in K. """

        if self.t_units == 'C':
            return self.temp + 273.
        return self.temp

    def range(self, t_min, t_max):
        """ Returns an object of TGA limited to a given range of temperature. """

        boolean_mask = (t_min <= self.temp) & (self.temp <= t_max)
        return TGA(self.temp[boolean_mask],
                   self.mass_loss[boolean_mask],
                   self.mass_loss_derivative[boolean_mask],
                   self.heating_rate, t_units=self.t_units)


@dataclass
class ModelFittingResult:
    """ Model fitting results object. """

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


""" Models """


class SimpleOrderModel:
    """ (1-alpha)**n model """

    def __init__(self, n):
        self.n = n

    @property
    def name(self):
        return f'{self.n} order model'

    def __call__(self, alpha):
        return (1 - alpha) ** self.n


class TwoDimDiffusionModel:
    """ Two dimensional diffusion model """

    @property
    def name(self):
        return f'2-dim diffusion model'

    def __call__(self, alpha):
        return -1 / np.log(1 - alpha)


class ThreeDimDiffusionModel:
    """ Three dimensional diffusion model """

    @property
    def name(self):
        return f'3-dim diffusion model'

    def __call__(self, alpha):
        return 1.5 * ((1 - alpha) ** (2 / 3)) / (1 - (1 - alpha) ** (1 / 3))


class ModelFitting:
    """ Model fitting of TGA data. """

    def __init__(self):
        pass

    def fit_model(self, tga, model):
        slope, intercept, rvalue, pvalue, stderr = linregress(*self.fitting_line(tga, model))
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
    def fitting_line(tga, model):
        # [ln(da/dT) + ln(beta) - ln(model(a))] vs -1/RT
        #
        ##########################################################################
        # [ln(da/dT) + ln(beta) - ln(model(a))] = INTERCEPT + SLOPE * (-1/RT)
        # INTERCEPT = lnA
        # SLOPE = Ea
        ##########################################################################
        #
        X = -1 / (R * tga.temp_in_kelvins)
        Y = np.log(tga.dtg * tga.heating_rate / model(tga.alpha))

        # filter out NaN values that might happen with real data
        filter_mask = ~np.isnan(Y) & ~np.isinf(Y)
        fitting_tuple = (X[filter_mask], Y[filter_mask])

        return fitting_tuple
