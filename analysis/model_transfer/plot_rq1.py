from plotnine import *
from mizani.palettes import hue_pal
import pandas as pd
import numpy as np


palette = hue_pal(0.01, 0.6, 0.65, color_space='hls')
pal = palette(8)

name = 'rq5_f1'
df = pd.read_csv(f'{name}.csv')
# size = "small"
size = "big"

if size != "small":
    height = 7
else:
    height = 2

# Need to do some conversions first

new_df = pd.DataFrame(columns=["exp", "size", "score", "model", "type", "linetype"])

df.astype({'Number of training examples': np.float64})

def calc_type(model):
    if "SpanBERT" in model or "XLM" in model:
        return 2
    else:
        return 1

def fix(model):
    d = {
        "XLM-R-large": 1,
        "SpanBERT-large": 2,
        "SpanBERT-base-on": 3,
        "SpanBERT-large-on": 4,
        "Transfer-on": 5,
        "Transfer-pc": 6,
        "Transfer-en": 7
    }
    return d[model]

if name == "rq1_f1":
    model_labels = ["SpanBERT (L)", "SpanBERT-On (b)", "SpanBERT-On (L)",
                    "Transfer (on)", "Transfer (pc)"]
    old_linetypes = ["dashed", "dashed", "dashed", "solid", "solid"]
elif name == "rq5_f1":
    model_labels = ["XLM-R (L)", "Transfer (en)"]
    old_linetypes = ["dashed", "solid"]
                 
old_palette = [pal[3], pal[1], pal[0], pal[2],  pal[6], "blue", "pink"]
# old_palette = ["orange", pal[1], pal[0], pal[2],  pal[6], "blue", "pink"]



for i, row in df.iterrows():
    if row["Number of training examples"] == 0:
        continue
    new_row1 = {"exp": row["exp"], "size": row["Number of training examples"],
                "model": fix(row["model"]), "score": row["Average F1"], "type": "1. Coref (pred. ment.)", "linetype": calc_type(row["model"]) }
    new_df = new_df.append(new_row1, ignore_index=True)
    if size != "small":
        new_row2 = {"exp": row["exp"], "size": row["Number of training examples"],
                    "model": fix(row["model"]), "score": row["Mention F1"], "type": "2. Mentions",  "linetype": calc_type(row["model"])}
        new_row3 = {"exp": row["exp"], "size": row["Number of training examples"],
                    "model": fix(row["model"]), "score": row["Average F1 (gold mentions)"], "type": "3. Coref (gold ment.)",  "linetype": calc_type(row["model"])}
        new_df = new_df.append(new_row2, ignore_index=True)
        new_df = new_df.append(new_row3, ignore_index=True)

new_df = new_df.dropna()
# new_df = new_df[new_df['exp'] == "1. LitBank"]  

def relabel_facet(f):
    return f[3:]

p = (ggplot(new_df, aes(x='size',y='score', group='model'))
     + labs(x='# training documents', y='F1', color="Init. method")
     + geom_line(aes(color="model", linetype="model"), size=1.0)
     + scale_x_log10()
     + scale_y_continuous()
     # + scale_linetype_manual(values=["solid", "dashed"], name="Init. method",
     #                         labels=["1", "2,"])
     + scale_color_manual(values = old_palette,
                          name="Initialization method", labels=model_labels)
     + scale_linetype_manual(values = old_linetypes,
                             name="Initialization method", labels=model_labels)
     + facet_grid(("type", "exp"), scales="free",
                  labeller=labeller(cols=relabel_facet,
                                    rows=relabel_facet))
     + theme_bw()
     + theme(panel_spacing_x=0.25,
             axis_title_x=element_text(size=16),
             axis_title_y=element_text(size=16),
             legend_position="top",
             legend_title_align="center",
             legend_text=element_text(size=12),
             legend_title=element_text(size=14),
             legend_entry_spacing_x=10.0,
             strip_text_x=element_text(size=11),
             strip_text_y=element_text(size=11),
             
     )
)

p.save(filename = f'{name}_{size}.pdf', height=height, width=12, units = 'in', dpi=400)
