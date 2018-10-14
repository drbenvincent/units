import numpy as np


class Param:

    def __init__(self, name, units=None, true_value=None, xi=None):
        self.name = name
        self.units = units
        self.true_value = true_value
        self.xi = xi

    @property
    def label(self):
        return f'{self.name} ({self.units})'


class ParamSpace2D:

    def __init__(self, param_list):
        self.params = param_list
        self.grid_size = [param.xi.size for param in self.params]

    def grid_evaluate(self, func):
        """Evaluate the supplied function over the parameter grid"""
        self.LL = np.empty(self.grid_size)
        row_vec = self.params[0].xi
        col_vec = self.params[1].xi

        for column, col_value in enumerate(col_vec):
            for row, row_value in enumerate(row_vec):
                params = [row_value, col_value]
                self.LL[row, column] = func(params)

        params_ml = self.arg_max_2D(self.LL, col_vec, row_vec)
        self.params[0].mle = params_ml[0]
        self.params[1].mle = params_ml[1]

    def plot(self, ax, xscale='linear'):
        row_vec = self.params[0].xi
        col_vec = self.params[1].xi

        # log likelihood surface
        L = np.exp(self.LL)
        mesh = ax.pcolormesh(col_vec, row_vec, L, cmap='gray_r')

        if xscale is 'log':
            ax.set_xscale('log')  # xscale('log')
            
        ax.set_ylabel(self.params[0].label)
        ax.set_xlabel(self.params[1].label)

        # true parameter crosshairs
        ax.axhline(y=self.params[0].true_value)
        ax.axvline(x=self.params[1].true_value)

    @staticmethod
    def arg_max_2D(matrix, col_vec, row_vec):
        max_indicies = np.unravel_index(matrix.argmax(), matrix.shape)
        row_value = row_vec[max_indicies[0]]
        col_value = col_vec[max_indicies[1]]
        return (row_value, col_value)

    @property
    def true_params(self):
        return [param.true_value for param in self.params]
