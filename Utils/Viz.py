import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, NumeralTickFormatter, Column, HoverTool, Row
from bokeh.models.widgets import DataTable, TableColumn
import matplotlib as mpl
import pandas as pd
import numpy as np
import typing

from .DfScaler import DfScaler

def simple_color_map( data,var_type,numerical_scale_method = 'RobustScaler',numerical_cmp = 'viridis'):
    if var_type == 'categorical':
        cmp = 'Paired'
        classes = data.unique()
        class_enumerator = {v: i for i, v in enumerate(set(classes))}
        array = np.array([class_enumerator[cl] for cl in data]).flatten()
        rgb_array = np.array(
            [[int(r), int(g), int(b)] for r, g, b, _ in 255 * getattr(mpl.cm, cmp)(mpl.colors.Normalize()(array))]
        )
        rgb_array = rgb_array.astype(int)
        colors = np.array(["#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]) for rgb in rgb_array])
        return colors

    elif var_type == 'numerical':
        
        if numerical_scale_method:
            if data.__class__ != pd.DataFrame:
                data = pd.DataFrame(data)
                data.columns = ['color_col']
            data = scale_df(data, columns = data.columns, method=numerical_scale_method)
            data = scale_df(data, columns=data.columns, method='MinMaxScaler')

        cmp = numerical_cmp
        rgb_array = np.array(
            [[int(r), int(g), int(b)] for r, g, b, _ in 255 * getattr(mpl.cm, cmp)(mpl.colors.Normalize()(data.values.flatten()))]
        )
        colors = np.array(["#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]) for rgb in rgb_array])
        return colors
    else:
        raise ValueError('var_type should be one of ["numerical","categorical"]')

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

def scale_df(df, columns,method = 'RobustScaler'):

    scaler = DfScaler(method = method, columns = columns)
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    return scaled_df

def bokeh_reduce_scatter(
        df,
        file_name,
        file_title,
        x_axis_label,
        y_axis_label,
        colors_column=None,
        radii_column=None,
        colors_scale_method = None,
        radii_scale_method = None,
        max_radii = 0.5,
        fill_alpha = 0.5,
        nonselection_alpha=0.5,
        hover_info=None,
        select_tools=['wheel_zoom', 'crosshair', 'undo', 'redo', 'tap',
                      'reset', 'box_zoom','pan','lasso_select'],
        plot_height=800,
        plot_width=1200,
        table_height = 100,
        table_width = 1200,
        mpl_color_map=['viridis'],
        plot_title='plot_title',
        toolbar_location='below',
        line_color=None,
        x_range = None,
        y_range = None,
        data_table_columns = None
):
    # create 2Dcolormap if not color array provided
    if not colors_column:
        df['filling_colors'] = bidim_colormap(df[[x_axis_label, y_axis_label]].values, cmp=mpl_color_map)
    elif df[colors_column].dtypes == 'O':
        df['filling_colors'] = df[colors_column]
    else:
        df['filling_colors'] = simple_color_map(df[colors_column], var_type = 'numerical', numerical_scale_method='MinMaxScaler')

    if colors_scale_method:
        df = scale_df(df,['filling_colors'],method = 'RobustScaler')
    if radii_scale_method:
        df = scale_df(df, [radii_column], method='RobustScaler')

    # SET RADIUS ARGS
    radiiarg = {}
    if radii_column:
        radiiarg = {'radius': radii_column}
        df.loc[:,radii_column] = max_radii*df[radii_column]/df[radii_column].max()

    #CREATES OUTPUT FILE
    output_file(file_name + '.html',
                title=file_title)
    #CREATES DATA SOURCE
    data_cds = ColumnDataSource(df)
    #CREATES DATA TABLE SOURCE
    if data_table_columns:
        dt_columns = [TableColumn(field=i, title=i) for i in data_table_columns]
        data_table = DataTable(source=data_cds, columns=dt_columns, width=table_width, height=table_height,fit_columns=True)

    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        title=plot_title,
        toolbar_location=toolbar_location,
        tools=select_tools,
        x_range = x_range,
        y_range = y_range,
        output_backend="webgl"
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
    if data_table_columns:
        layout = Column(fig, data_table)
    else:
        layout = fig
    show(layout)
    return