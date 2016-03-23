import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


image = mpimg.imread('./belle-de-nuit-bicolore.jpg')
print(image)
imgplot = plt.imshow(image)
plt.show(imgplot)