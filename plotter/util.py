import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Plot loss function
def plot_loss(loss_history, display=False, path: Path = None):
    fig, axe = plt.subplots(figsize=[6, 5])  # inches

    axe.plot(loss_history)
    axe.set_yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plot_display(display, path)

    plt.close()


# Plot the results
def plot_result(input_train, output_train, press_predict, display=False, path: Path = None):
    plt.scatter(input_train, output_train[0], label='circ data')
    plt.plot(input_train, press_predict, color='red', label='fitted circ')
    plt.legend()

    plot_display(display, path)

    plt.close()


def plot_something(input_train, output_train, terms, lowers, uppers, display=False, path: Path = None):
    plt.rcParams['xtick.major.pad'] = 14  # set plotting parameters
    plt.rcParams['ytick.major.pad'] = 14
    # Plot the contributions of each term to the output of the model
    fig, axt = plt.subplots(figsize=(12.5, 8.33))
    num_terms = terms
    cmap = plt.get_cmap('jet_r', num_terms)  # define the colormap with the number of terms from the full network
    # this way, we can use 1 or 2 term models and have the colors be the same for those terms
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # ax.set_xticks([1, 1.02, 1.04, 1.06, 1.08, 1.1])
    # axt.set_xlim(1, 2.0)
    # ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    # axt.set_ylim(0, 20.0)
    # colormap
    for i in range(terms):
        lower = lowers[:, i]
        upper = uppers[:, i]
        axt.fill_between(input_train, lower.flatten(), upper.flatten(), lw=0, zorder=i + 1, color=cmaplist[i],
                         label=str(i + 1))
        axt.plot(input_train, upper, lw=0.4, zorder=34, color='k')

    axt.scatter(input_train, output_train[0], s=200, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
    plt.title('contributions w2_x')
    plt.tight_layout(pad=2)
    plt.legend()

    plot_display(display, path)

    plt.close()


def plot_display(display=False, path: Path = None):
    if path is not None:
        plt.savefig(path)

    if display:
        plt.show()
