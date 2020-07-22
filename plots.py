import matplotlib.pyplot as plt
import pandas as pd


weights = [14.5589288,  -10.91125129, 134.64405279,  -2.04189656,  60.60180254,
   5.08663998,  27.57215712,  11.22929793,  58.56049503,  69.74105026,
  -6.38408481]


df = pd.DataFrame({'weights':weights})
df['positive'] = df['weights'] > 0
df['features'] = range(1,12)
df.set_index("features",drop=True,inplace=True)



df['weights'].plot(x='features',
                   kind='bar',
                   color=df.positive.map({True: 'g', False: 'r'}))

plt.show()