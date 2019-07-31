
import numpy as np
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, xlabel, split_name, title_name, x, y, markers=False):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name,
                markers=markers,
                markersymbol='plus'
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


class VisdomScatterPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name

    def plot(self, title_name, x, y, legends=None):

        self.viz.scatter(X=x,
                         Y=y,
                         env=self.env,
                         opts=dict(
                             title=title_name,
                             legends=[legends],
                             markersymbol='dot'
                         ))


class VisdomHeatmap():
    """Plots to Visdom"""

    def __init__(self, env_name='main', port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name

    def plot(self, title_name, x, class_list):

        self.viz.heatmap(
            X=np.flipud(x),
            env=self.env,
            opts=dict(
                title=title_name,
                columnnames=class_list,
                rownames=list(reversed(class_list))
            )
        )