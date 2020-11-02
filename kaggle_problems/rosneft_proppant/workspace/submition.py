import pandas as pd
from common import COEF, r2prop_size
import os

class Submition:
    def __init__(self, img_number, filename):
        self.img_number = img_number
        self.filename = filename
        self.write_header = not os.path.exists(self.filename)

    def submit(self, circles):
        diam = [circle[2] for circle in circles]
        sb = pd.DataFrame(data={
            "ImageId": [self.img_number] * len(diam),
            "prop_size": diam})

        sb.to_csv(self.filename, index=False, mode='a', header=self.write_header)