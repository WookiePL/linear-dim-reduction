from datetime import datetime


def make_num_binary(item):
    if not (item == 0):
        return 1
    else:
        return 0


def save_plot_as_png_file(plt):
    plt2 = plt
    date = str(datetime.now().strftime('%Y-%m-%d %H%M%S'))

    plt2.savefig(date + ".png")

