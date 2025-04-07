import pandas as pd
from PIL import Image
import numpy as np

df = pd.read_csv("readData\\train.csv")

print(df.head())

imgArr = df.iloc[1, 1:].to_numpy().reshape(28, 28).astype(np.int8)

img = Image.fromarray(imgArr)
img.show()