import os
import matplotlib.pyplot as plt


def get_fig_name(path):
    """Get unique figure name"""
    i = 0
    while True:
        f = os.path.join(path, str(i))
        if not os.path.exists(f):
            return f


class Plotter(object):

    """Class for interactively plotting eval scores parent process

    :class:`Plotter` plots completed jobs during asynchronous job completion
    interactive in parent process. The resulting figure can be saved to file.

    Args:
        path (str): path to log files
        save (bool): save figure
        semilogy (bool): use semilogy instead of plot
        fig_kwargs (dict): figure kwargs
        plot_kwargs (dict): plot kwargs
        save_kwargs (dict): savefig kwargs
    """

    def __init__(self, path, save, semilogy,
                 fig_kwargs, plot_kwargs, save_kwargs):
        self.path = path
        self.save = save
        self.semilogy = semilogy
        self.fig_kwargs = fig_kwargs
        self.plot_kwargs = plot_kwargs
        self.save_kwargs = save_kwargs
        self._plot_count = 0
        self._cm = [plt.cm.rainbow(i) for i in [i / 10 for i in range(1, 21)]]

        plt.ion()

        f, ax = plt.subplots(**fig_kwargs)
        self._f = f
        self._ax = ax

    def plot(self, jid):
        x, y = self.load_log(os.path.join(self.path, '{}.log'.format(jid)))
        if len(x) < 2:
            return

        color = self._cm[self._plot_count]
        f = self._ax.plot if not self.semilogy else self._ax.semilogy
        f(x, y, c=color, **self.plot_kwargs)

        plt.pause(0.1)
        self._plot_count = (self._plot_count + 1) % 10

        if self.save:
            fig_name = get_fig_name(self.path)
            plt.savefig(fig_name, **self.save_kwargs)

    def load_log(self, log):
        """Load eval scores for given log"""
        # TODO: break this out into eval or utils
        x = []
        y = []
        with open(log, 'r') as olog:
            for line in olog:
                if line.startswith('EVAL:'):
                    _, s, v = line.split(':')
                    s = int(s)
                    v = float(v)
                    x.append(s)
                    y.append(v)
        return x, y
