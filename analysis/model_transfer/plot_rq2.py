from plotnine import *
import pandas as pd
import numpy as np

df = pd.read_csv('rq2_f1.csv')

df = df.astype({"num train examples": str, "num dev examples": str, "matches": int})

datavals = [1, 5,10, 15, 25, 50, 100, 150, 250, 500]
datavals = [str(x) for x in datavals]

df["num train examples"] = pd.Categorical(df["num train examples"], categories=datavals)
df["num dev examples"] = pd.Categorical(df["num dev examples"], categories=datavals)
df["label"] = df["average f1"].round(decimals=1)
df["stdev"] = df["average std"].round(decimals=1)

p = (ggplot(df, aes('num train examples', 'num dev examples', fill='average f1'))
     + labs(x="# training documents", y="# dev documents", fill="Avg. F1")
     + scale_fill_gradient(low="white", high="#02841c", limits=[70, 85])
     + geom_tile(aes(width=.95, height=.95))
     + geom_text(aes(label='label'), size=9, nudge_y = 0.1)
     + geom_text(aes(label='stdev'), nudge_x = -0.15, nudge_y = -0.3, size=4.5, format_string="\u00B1{}")
     + geom_text(aes(label='matches'), nudge_x = 0.30, nudge_y = -0.3, size=4.5)
     + facet_wrap("model")
     + theme_bw()
     )

p.save(filename = 'rq2_f1.pdf', height=4, width=8, units = 'in', dpi=200)
