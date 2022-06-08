import pandas as pd
from utopya.plotting import register_operation

register_operation(name="pd.Index", func=pd.Index)
