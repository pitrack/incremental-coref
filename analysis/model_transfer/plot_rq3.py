from plotnine import *
import pandas as pd
import numpy as np
from mizani.palettes import hue_pal

name="rq6_f1"
df = pd.read_csv(f'{name}.csv')

palette = hue_pal(0.01, 0.6, 0.65, color_space='hls')
pal = palette(4)
old_palette = [pal[1], pal[3], pal[0]]
model_labels = ["Transfer (on)", "Transfer (pc)", "Transfer (en)"]
# if name == "rq1_f1":
#     old_palette = [pal[1], pal[3], pal[0]]
# else:
#     old_palette = [pal[0], pal[1], pal[3]]
def fix(model):
    d = {
        "Transfer-on": "1",
        "Transfer-pc": "2",
        "Transfer-en": "3"
    }
    return d[model]
# if name == "rq1_f1":
#     model_labels = ["Transfer (on)", "Transfer (pc)"]#, "Transfer (en)"]
# if name == "rq5_f1":
#     model_labels = ["Transfer (en)"]
# Need to do some conversions first

new_df = pd.DataFrame(columns=["exp", "size", "score", "model", "type", "std", "dataset"])

df.astype({'Number of training examples': np.float64})
for i, row in df.iterrows():
    if "Transfer" not in row["model"]:
        continue
    new_row1 = {"exp": row["exp"], "size": row["Number of training examples"],
                "model": fix(row["model"]), "score": row["Average F1"], "type": "1. Avg. F1 (pred. ment.)" + row["model"],
                "std": 0, "eval set":  "target"}
    new_row2 = {"exp": row["exp"], "size": row["Number of training examples"],
                "model": fix(row["model"]), "score": row["Average (source) F1"], "type": "2. Avg. F1 (orig. data)" + row["model"],
                "std": row["Average (source) std"], "eval set": "source"}
    new_df = new_df.append(new_row1, ignore_index=True)
    new_df = new_df.append(new_row2, ignore_index=True)

new_df.fillna(0, inplace=True)

def relabel_facet(f):
    return f[3:]

p = (ggplot(new_df, aes(x='size',y='score', color='model', group='type'))
     + labs(x='# training documents', y='Avg. F1', color='Init. method', linetype="Eval. set")
     + geom_line(aes(linetype='eval set'))
     + scale_x_log10()
     + scale_y_continuous()
     # + facet_grid((".", ["exp"]), scales="free",
     #              labeller=labeller(cols=relabel_facet,
     #                                rows=relabel_facet))
     + facet_wrap("exp", nrow=2, scales="free",
                  labeller=labeller(cols=relabel_facet,
                                    rows=relabel_facet))
     
     + scale_color_manual(values = old_palette,
                          name="Init. method", labels=model_labels)
     + theme_bw()
     + theme(panel_spacing_x=0.25,
             axis_title_x=element_text(size=16),
             axis_title_y=element_text(size=16),
             subplots_adjust={'hspace': 0.4},
             legend_position=(0.85, 0.75),
     )
)

print(f"Saved to rq3_{name}.pdf")
p.save(filename = f'rq3_{name}.pdf', height=4, width=10, units = 'in', dpi=400)
