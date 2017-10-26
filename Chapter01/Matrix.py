import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
#plotly.tools.set_credentials_file(username='rezacsedu', api_key='dpVqkr9iHl7kiQKea9mG')

import numpy as np
import pandas as pd
import scipy

matrix1 = np.matrix(
    [[1, 5],
     [6, 2]]
)

matrix2 = np.matrix(
    [[-1, 4],
     [3, -6]]
)

matrix_sum = matrix1 + matrix2
print(matrix_sum)

colorscale = [[0, '#EAEFC4'], [1, '#9BDF46']]
font=['#000000', '#000000']

table = FF.create_annotated_heatmap(matrix_sum.tolist(), colorscale=colorscale, font_colors=font)
py.iplot(table, filename='matrix-sum')

