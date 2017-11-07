# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import time
np.array(range(8)).reshape(2,4).max(axis=0)

a=pd.DataFrame({'q':range(5),'w':range(5)})
b=pd.DataFrame({'e':range(5),'r':range(5)})
c=pd.DataFrame()
c=pd.concat([c,a])
print(c)
print(pd.concat([c,b]).shape[0])