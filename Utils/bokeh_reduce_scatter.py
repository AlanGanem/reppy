# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:12:48 2020

@author: User Ambev
"""

from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, NumeralTickFormatter
from bokeh.models import HoverTool
import matplotlib as mpl
import pandas as pd
import numpy as np


# IMPLEMENT NODE SIZE OPTION

def bidim_colormap(array, cmp):
    if not cmp.__class__ == list:
        raise TypeError('color_map (cmp) must be list')
    if len(cmp) == 1:
        cmp += cmp

    dim1 = np.array(
        [[int(r), int(g), int(b)] for r, g, b, _ in 255 * getattr(mpl.cm, cmp[0])(mpl.colors.Normalize()(array[:, 0]))])
    dim2 = np.array(
        [(int(r), int(g), int(b)) for r, g, b, _ in 255 * getattr(mpl.cm, cmp[1])(mpl.colors.Normalize()(array[:, 1]))])

    rgb_array = (dim1 + dim2) / 2
    rgb_array = rgb_array.astype(int)

    colors = ["#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]) for rgb in rgb_array]
    return colors

def color_map(array, cmp):
    rgb_array = np.array([[int(r), int(g), int(b)] for r, g, b, _ in 255 * getattr(mpl.cm, cmp)(mpl.colors.Normalize()(array))])
    rgb_array = rgb_array.astype(int)
    colors = ["#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]) for rgb in rgb_array]
    return colors

def bokeh_reduce_scatter(
        df,
        file_name,
        file_title,
        colors_column=None,
        radii_column=None,
        fill_alpha = 0.5,
        nonselection_alpha=0.5,
        hover_info=None,
        select_tools=['wheel_zoom', 'crosshair', 'undo', 'redo', 'box_select', 'lasso_select', 'poly_select', 'tap',
                      'reset', 'box_zoom'],
        x_axis_label='X',
        y_axis_label='y',
        plot_height=800,
        plot_width=1200,
        mpl_color_map=['viridis'],
        plot_title='plot_title',
        toolbar_location='below',
        line_color=None,
):
    # create 2Dcolormap if not color array provided
    if not colors_column:
        df['filling_colors'] = bidim_colormap(df[[x_axis_label, y_axis_label]].values, cmp=mpl_color_map)
    else:
        df['filling_colors'] = df[colors_column]

    radiiarg = {}
    if radii_column:
        radiiarg = {'radius': radii_column}

    output_file(file_name + '.html',
                title=file_title)

    data_cds = ColumnDataSource(df)

    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        title=plot_title,
        toolbar_location=toolbar_location,
        tools=select_tools,

    )

    fig.scatter(
        x=x_axis_label,
        y=y_axis_label,
        source=data_cds,
        # color=colors,
        selection_color='deepskyblue',
        nonselection_color='lightgray',
        nonselection_alpha=nonselection_alpha,
        fill_alpha = fill_alpha,
        fill_color='filling_colors',
        line_color=line_color,
        **radiiarg
    )

    # Add the HoverTool to the figure
    if hover_info:
        if hover_info.__class__ not in [list, tuple, set]:
            hover_info = [hover_info]
        tooltips = [(i, '@{}'.format(i)) for i in hover_info]
        fig.add_tools(HoverTool(tooltips=tooltips))
    # Visualize
    show(fig)
    return