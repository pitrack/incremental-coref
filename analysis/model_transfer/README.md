# Plotting scripts

These are plotting scripts for "Moving on from OntoNotes: Coreference Resolution Model Transfer." The scripts assume data is already correctly formatted as CSVs, and there are also some renaming code hacks/laziness from the raw files as the same file was used to generate multiple plots (usually to change size). These are provided mostly as a reference and not intended to be re-run exactly.

These all use the [plotnine](https://plotnine.readthedocs.io/en/stable/) library.

## Figure 1, Figure 2, Figure 6

This uses the `plot_rq1.py` and assumes a csv file `rq1_f1.csv` (for Fig. 1) and `rq5_f1.csv` (for Fig. 2) which has these columns:

```
exp,model,Number of training examples,Average F1,Mention F1,Average F1 (gold mentions),Average (source) F1,Average (source) std
1. LitBank,SpanBERT-large,5,49.8,79.8,62.8,,
```
There are some tweaks in the code based on what exactly needs to be plot (and how big). "small" only plots the first row.

There is a setting I couldn't figure out, and so I needed to go into [`plotnine/facets/facet_grid.py#144`](https://github.com/has2k1/plotnine/blob/master/plotnine/facets/facet_grid.py#L144) and add the following to get independent y-axis labels.

```
layout['SCALE_Y'] = layout['PANEL']
layout['AXIS_Y'] = True
```

## Figure 3

This relies on `dev_analysis.py` and `process_dev_analysis.py` to generate the results while `plot_rq2.py` plots the figure. The csv should contain entries like this

```
model,num train examples,num dev examples,average f1,average std,matches
SpanBERT-On (L),1,1,32.575,21.699,4
SpanBERT-On (L),1,5,44.959,18.213,11
SpanBERT-On (L),1,10,50.941,9.708,7
```

## Figure 4

This relies on a `rq6_f1.csv` file and `plot_rq3.py`. Again, we need to make a modification to [`plotnine/facets/facet_wrap.py#L74`](https://github.com/has2k1/plotnine/blob/master/plotnine/facets/facet_wrap.py#L74) and change ` _id = np.arange(1, n+1)` to `_id = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]` so that panel 6 is skipped and so the legend can be placed in the top-right. The format should look the same as the files for Figure 1, 2.

## Figure 5, Figure 7

This uses `plot_rq4.py` (Fig 5) or `plot_rq4_large.py` (Fig 7) with `rq4_f1.csv` containing entries like this:

```
exp,model,layers,layer percent,Number of training examples,Average F1,line type,type
1. LitBank,SpanBERT-large,0,0,10,28.4,1solid,1solidSpanBERT-large
1. LitBank,SpanBERT-large,6,25,10,51.9,1solid,1solidSpanBERT-large
```




