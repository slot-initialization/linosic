import numpy as np
import matplotlib.pyplot as plt
import numpy



def plot(x_axis, y_axis, labels, amount, title='Plot'):
    plt.clf()
    for i in range(amount):
        plt.plot(x_axis[i], y_axis[i], label=labels[i])#), marker='+')
    plt.legend()
    plt.title(title)
    return plt


def read_eval_result(path):
    lines = []
    with open(path, 'r') as txt_file:
        for line in txt_file:
            lines.append(ast.literal_eval(line))
        # lines = txt_file.read().split('\n')[:-1]
    return lines


def get_eval_res_col(col_nums, lines):
    cols = []
    for col in col_nums:
        cols.append([l[col] for l in lines])
    arg_sort = list(np.argsort(cols[0]))
    cols = [[col[arg] for arg in arg_sort] for col in cols]
    return cols


def stack_eval_res_col(col_nums, paths, labels, title):
    x_axis_stack, y_axis_stack = [], []
    amount = len(paths)
    for i in range(amount):
        lines = read_eval_result(paths[i])

        x_axis, y_axis = get_eval_res_col(col_nums, lines)
        x_axis_stack.append(x_axis)
        y_axis_stack.append(y_axis)
    save_fig = plot(x_axis_stack, y_axis_stack, labels, amount, title)
    return save_fig


