from nose import with_setup, tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from . import holtwinters

class TestHoltWinters:
  @classmethod
  def setup_class(cls):
    filepath = os.path.join(os.path.dirname(__file__), '../data', 'monthly_demand.csv')
    cls.data = pd.read_csv(filepath)['demand']

  @classmethod
  def teardown_class(cls):
    pass

  def test_holtwinters(self):
    forecast, smooth, rmse = holtwinters.multiplicative(list(self.data.values), 12, 12)
    tools.assert_true(2.75 < rmse < 2.76)
    tools.assert_true(np.allclose(forecast,
      [73.12, 64.8, 78.82, 88.71, 88.05, 102.06,
       101.05, 99.9, 99.55, 86.03, 79.05, 61.14],
       1e-04))

  def plot(self):
    forecast, smooth, rmse = holtwinters.multiplicative(list(self.data.values), 12, 12)
    predictions = np.hstack([smooth, forecast])
    plt.plot(self.data, 'b')
    plt.plot(predictions, 'g--')
    plt.show()
