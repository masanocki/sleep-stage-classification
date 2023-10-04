import numpy as np
import mne
import matplotlib.pyplot as plt
import data_prep as dp

data = dp.read_file("datasets/recordings/SN001.edf", data_type="raw")
mne.viz.plot_raw(data, block=True)
plt.show()
