######### viz
def create_scatter(data, graph)
    df_dict = {
        'Description': index2word,
        'X':n2v_2d[:,0].flatten(),
        'y':n2v_2d[:,1].flatten(),
        'classes':classes
        }


df = pd.DataFrame(df_dict)

df['colors'] = colors
df['radius'] = 0.1

dir(mpl.cm)

bokeh_reduce_scatter(
    df,
    nonselection_alpha = 0.5,
    radii_column='radius',
    colors_column = 'colors',
    hover_info = ['Description'],
    file_name = 'node2vec_umap',
    file_title = 'node2vec_umap',
    x_axis_label='X',
    y_axis_label='y',
    plot_height=800,
    plot_width=1200,
    mpl_color_map = ['inferno'],
    plot_title='node2vec map',
    toolbar_location='below',
)



class ColorLab:
    @classmethod
    def scale_values(cls):
        return

    @classmethod
    def scale_values(cls):

        return

    @classmethod
    def color_map(cls, data, var, var_type = 'categorical'):
        if var_type == 'categorical':
            class_enumerator = {v: i for i, v in enumerate(set(classes))}
            class_num = [class_enumerator[cl] for cl in classes]
        return

colors = color_map(class_num, 'Paired')


fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(x1,x2, s=0.3, c=np.array(train_labels), cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(classes)
plt.title('umap transform')