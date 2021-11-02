from plotnine import *
import pandas as pd
import numpy as np
from mizani.palettes import hue_pal
df = pd.read_csv('rq4_f1.csv')

palette = hue_pal(0.01, 0.6, 0.65, color_space='hls')
pal = palette(4)
old_palette = [pal[1], pal[3], pal[0]]
def fix(model):
    d = {
        "Transfer-on": "1",
        "Transfer-pc": "2",
        "Transfer-en": "3"
    }
    return d[model]


palette = hue_pal(0.01, 0.6, 0.65, color_space='hls')
pal = palette(8)

new_df = df.dropna()
new_df = new_df.astype({'Number of training examples': str})

model_labels = ["SpanBERT (L)", "SpanBERT-On (L)",
                "Transfer (on)", "Transfer (pc)"]
old_linetype = ["dashed", "dashed", "solid", "solid"]
old_palette = [pal[3], pal[1], pal[2], pal[6], "blue", pal[0], "pink"]

def relabel_facet(f):
    if f == "1solid":
        return "Low"
    elif f == "2dashed":
        return "Medium"
    elif f == "3dotted":
        return "All"
    else:
        return f[3:]

    
p = (ggplot(new_df, aes(x='layers', y='Average F1', color='model', group='type'))
     + labs(x='Top k layers are trainable', y='Coref Avg. F1', color='model', linetype="line type")
     + geom_line(aes(linetype="model"), size=1.0)
     + scale_y_continuous()
     + scale_x_continuous(breaks=[0,6,12,18,24])
     + facet_grid(("line type", "exp"), scales="free",
                  labeller=labeller(cols=relabel_facet,
                                    rows=relabel_facet))
     + scale_color_manual(values = old_palette,
                          name="Initialization method", labels=model_labels)
     + scale_linetype_manual(values = old_linetype,
                          name="Initialization method", labels=model_labels)
     + theme_bw()
     + theme(panel_spacing_x=0.25,
             axis_title_x=element_text(size=16),
             axis_title_y=element_text(size=16),
             legend_position="top",
             legend_text=element_text(size=12),
             legend_title=element_text(size=14),
             legend_entry_spacing_x=10.0,
             legend_entry_spacing_y=3.0
     )
     + guides(color=guide_legend(nrow=2))
)

p.save(filename = 'rq4_f1.pdf', height=5, width=5, units = 'in', dpi=400)
