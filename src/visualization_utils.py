import matplotlib.pyplot as plt


def plot_datapoint(window, k):
    plt.style.use("ggplot")
    fig = plt.figure(figsize=[16, 10])
    title = fig.suptitle(f"Plot of class {k}")
    title.set_fontsize(20)
    ax = fig.add_subplot()
    x_axis = ax.xaxis
    y_axis = ax.yaxis
    x_axis.set_label_text("time")
    y_axis.set_label_text("Amplitude")
    line_plot, = ax.plot(window)
    line_plot.set_linewidth(2)
    line_plot.set_color("C1")
    return fig


def plot_multiple_samples(data, rows, columns):
    """

    :param data: size should be [rows, columns, 1, window_size]
    :param rows:
    :param columns:
    :return:
    """
    plt.style.use("ggplot")
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(16, 10))

    for c in range(columns):
        for r in range(rows):
            ax = axes[r, c]
            window = data[r, c, 0, :]
            line_plot, = ax.plot(window)
            line_plot.set_linewidth(2)
            line_plot.set_color("C1")

    for c in range(columns):
        ax = axes[0, c]
        ax.set_title(f"Plot of class {c}")

    fig.tight_layout()
    return fig
