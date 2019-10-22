
import numpy as np
from visdom import Visdom


class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, server, env_name='main', port=8097):
        self.viz = Visdom(server, port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, ylabel, xlabel, legend, title_name, x, y, markers=False):
        if title_name not in self.plots.keys():
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[legend],
                title=title_name,
                xlabel=xlabel,
                ylabel=ylabel,
                markers=markers,
                markersymbol='plus'
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[title_name], name=legend, update ='append')

    def plot_curve(self,
                   ylabel: str,
                   xlabel: str,
                   legend: list,
                   title_name: str,
                   x: np.array,
                   y: np.array,
                   markers=False):

        if title_name in self.plots.keys():
            # Reset plot
            self.viz.close(self.plots[title_name], env=self.env)

        self.plots[title_name] = self.viz.line(X=x, Y=y, env=self.env, opts=dict(
            title=title_name,
            xlabel=xlabel,
            ylabel=ylabel,
            markers=markers,
            markersymbol='plus'
        ))

    def stem_plot(self,
                  title_name: str,
                  ylabel: str,
                  xlabel: str,
                  legend: list,
                  y: np.array,
                  x: np.array):

        if title_name in self.plots.keys():
            # Reset plot
            self.viz.close(self.plots[title_name], env=self.env)

        self.plots[title_name] = self.viz.stem(
            X=x,
            Y=y,
            env=self.env,
            opts=dict(legend=legend,
                      title_name=title_name,
                      xlabel=xlabel,
                      ylabel=ylabel)
        )

    def scatter_plot(self,
                     title_name: str,
                     x: np.array,
                     y: np.array,
                     legends=None):

        if title_name in self.plots.keys():
            # Reset plot
            self.viz.close(self.plots[title_name], env=self.env)

        y_plot = y.copy()
        y_plot += 1
        self.plots[title_name] = self.viz.scatter(X=x,
                                                  Y=y_plot,
                                                  env=self.env,
                                                  opts=dict(
                                                      title=title_name,
                                                      legends=[legends],
                                                      markersymbol='dot'
                                                  )
        )

    def vis_1d_distribution(self,
                            pts: np.array,
                            name: str) -> None:
        step = 0.05
        max_distance = max(pts) + step
        bins = np.arange(0.0, max_distance, step)
        hist, _ = np.histogram(pts, bins=bins)
        self.stem_plot(name, 'Samples number', name,
                          ['dist'], bins[1:], hist)

    def plot_exist(self, title_name):
        return title_name in self.plots.keys()





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