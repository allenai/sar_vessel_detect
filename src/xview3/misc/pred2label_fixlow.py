import pandas as pd
import sys

fname = sys.argv[1]

df = pd.read_csv(fname)
df.loc[df.confidence == 'LOW', 'score'] = 1.0
df.to_csv(fname, index=False)
