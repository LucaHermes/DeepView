from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

import numpy as np


class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    This class adjusts the source code taken from:
        https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to interact with.
        collection : `matplotlib.collections.Collection` subclass
            Collection you want to select from.
        alpha_other : 0 <= float <= 1
            To highlight a selection, this tool sets all selected points to an
            alpha value of 1 and non-selected points to *alpha_other*.
        """
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.xys = collection
        self.Npts = len(self.xys)
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        """
        Draws the shape around the set of points in the DeepView Image which are then put into a list

        Parameters
        ----------
        verts: np.ndarray
            List of vertices in the DeepView Image to be visualized by the shape visualization
        """
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        return self.ind

    def disconnect(self):
        """Disconnect all events created by this widget."""
        self.lasso.disconnect_events()
        self.canvas.draw_idle()