import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .utils import CurvedText, curved_labels
from .plot_helper import cat_filter, get_meta_data, metric_display, cat_display
import arabic_reshaper
from bidi.algorithm import get_display


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)
        self.counts = area
        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[: len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[: len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors, fontsize=10, show_counts=False):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(
                *self.bubbles[i, :2],
                labels[i],
                horizontalalignment="center",
                verticalalignment=show_counts and "bottom" or "center",
                fontfamily="arial",
                fontsize=fontsize,
            )
            if show_counts:
                ax.text(
                    *self.bubbles[i, :2],
                    self.counts[i],
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontfamily="arial",
                    fontsize=fontsize,
                )

        return ax


def dot_plot(df, showcmap=True, legend=False, clip=None, size=200, **kwargs):
    # cut,unbalance
    norm = None
    if clip:
        allvals = df.values.flatten()
        allvals = allvals[~np.isnan(allvals)]
        if clip.get("method"):
            Q3, Q1 = np.percentile(allvals, [75, 25])
            IQR = Q3 - Q1
            if clip.get("method") == "unbalanced":
                norm = mcolors.TwoSlopeNorm(vmin=1, vcenter=np.median(allvals))
            elif clip.get("method") == "cut_outliers":
                norm = mcolors.Normalize(
                    vmin=Q1 - 1.5 * IQR > 1 and Q1 - 1.5 * IQR or 1,
                    vmax=Q3 + 1.5 * IQR <= max(allvals)
                    and Q3 + 1.5 * IQR
                    or max(allvals),
                )
            # elif clip.get('method')=='outlier':
            else:
                raise ValueError(f"Invalid method {clip.get('method')}")
        else:
            if clip.get("vcenter"):
                norm = mcolors.TwoSlopeNorm(
                    vmin=clip.get("vmin", 1),
                    vcenter=clip.get("vcenter"),
                    vmax=clip.get("vmax"),
                )
            else:
                norm = mcolors.Normalize(
                    vmin=clip.get("vmin", 1), vmax=clip.get("vmax", max(allvals))
                )

    x = df.index
    y = df.columns
    X, Y = np.meshgrid(x, y)
    xy = pd.DataFrame({"X": X.flatten(), "Y": Y.flatten()})
    figsize = kwargs.get("figsize")
    if not figsize:
        width = (
            len(max(df.index.astype(str), key=len)) * len(df.index) // 10
        ) + 2.5 * showcmap
        height = len(y) // 3
        if width < 10:
            width = 10
        if height < 10:
            height = 10
        figsize = (width, height)
    fig, ax = plt.subplots(1, figsize=figsize)

    scatter = ax.scatter(
        xy.X, xy.Y, cmap="Greens", c=df.T.values.flatten(), s=size, norm=norm
    )
    if legend:
        kw = dict(
            prop="sizes",
            num=5,
            color=scatter.cmap(0.7),
            alpha=0.6,
            func=lambda s: np.sqrt(s),
        )
        legend2 = ax.legend(
            *scatter.legend_elements(**kw), title=kwargs.get("legend_title"), loc="best"
        )

    # loc=(1.15,0.9)
    # plt.legend(*im.legend_elements("sizes", num=2))
    if showcmap:
        plt.colorbar(scatter)
    ax.set_yticks(df.columns)
    plt.xticks(rotation=kwargs.get("rot", 0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if kwargs.get("hide_axis"):
        ax.spines["left"].set_alpha(False)
        ax.spines["bottom"].set_alpha(False)
    else:
        ax.spines["left"].set_alpha(0.2)
        ax.spines["bottom"].set_alpha(0.2)
    ax.tick_params(axis="both", length=0)

    return ax


outer_colors = np.vstack(
    (
        plt.colormaps["tab20c"](np.arange(5) * 4),
        plt.colormaps["tab20b"](np.arange(5) * 4),
    )
)
inner_colors = np.vstack(
    (
        plt.colormaps["tab20c"]([1, 2, 5, 6, 9, 10, 13, 14, 17, 18]),
        plt.colormaps["tab20b"]([1, 2, 5, 6, 9, 10, 13, 14, 17, 18]),
    )
)


def lighten(color, factor):
    return [color[i] + (1 - color[i]) * factor for i in range(3)] + [1.0]


def get_pie_vals(df, cols, order):
    mux = pd.MultiIndex.from_product([df[col].unique() for col in cols], names=cols)
    if len(cols) > 1:
        mux = mux.sort_values()
    data = (
        df[cols]
        .value_counts(sort=False)
        .reindex(mux, fill_value=0)
        .reset_index()
        .groupby(cols[0])[0]
        .agg(list)[order]
    )
    return data.explode().values


def label_corr(labels, vals):
    zeros = [i for i, x in enumerate(vals) if x == 0]
    for zero in zeros:
        labels[zero] = ""
    return labels


def rotate_text(pie, names, values, fit=False):
    if names:
        plt.setp(pie[1], rotation_mode="anchor", ha="center", va="center")
        for tx in pie[1]:
            rot = tx.get_rotation()
            tx.set_rotation(rot + 90 + (1 - rot // 180) * 180)
    if values:
        plt.setp(pie[2], rotation_mode="anchor", ha="center", va="center")
        for tx2, tx1 in zip(pie[2], pie[1]):
            rot = tx1.get_rotation()
            tx2.set_rotation(rot + fit * (90 + (1 - rot // 180) * 180))


def sunburst(
    data,
    metric,
    hrchy_levels=["sender_name"],
    filters={},
    subset=None,
    label_value=True,
    label_name=True,
    **kwargs,
):

    unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize"]))
    if unexpected_kwargs:
        raise TypeError(f"got an unexpected keyword argument {unexpected_kwargs.pop()}")

    if not isinstance(hrchy_levels, list):
        raise TypeError(
            f"got a type of {type(hrchy_levels)} for hrchy_levels, expected a list"
        )
    if not isinstance(label_value, (bool, list)):
        raise TypeError(f"expected list or bool , got {type(label_value)}")
    if isinstance(label_value, list) and len(label_value) != len(hrchy_levels) + 1:
        raise ValueError(
            f"expected a list of length {len(hrchy_levels)+1} for label_value, got {len(label_value)}"
        )

    if not isinstance(label_name, (bool, list)):
        raise TypeError(f"expected list or bool , got {type(label_name)}")

    if isinstance(label_name, list) and len(label_name) < len(hrchy_levels) + 1:
        raise ValueError(
            f"expected a list of length {len(hrchy_levels)+1} for label_name got {len(label_name)}"
        )
    if not isinstance(filters, dict):
        raise TypeError(f"expected dict, got {type(filters)}")

    if isinstance(label_value, bool):
        label_value = [label_value] * 4
    if isinstance(label_name, bool):
        label_name = [label_name] * 4
    df = data.copy()
    l0_label = "Chats"
    if df.sender_name.nunique() > 2:
        meta_data = get_meta_data()
        df.sender_name = df.sender_name.map(
            lambda name: name == meta_data["user"] and "You" or "Others"
        )
    if df.chat.nunique() < 2:
        df.chat = df.sender_name
        if "sender_name" in hrchy_levels:
            hrchy_levels.remove("sender_name")
        l0_label = "Senders"
    cat_mapper = {}
    for cat in hrchy_levels:
        cat_mapper.update(dict.fromkeys(df[cat].unique(), cat))
    if "chat" in hrchy_levels:
        raise Exception(
            "chat should not included in hrchy_levels, it is included by default"
        )
    levels = ["chat"] + hrchy_levels
    if not filters:
        filters = {}
    for cat in levels:
        if cat not in filters.keys():
            if cat == "chat":
                df, n_chat = cat_filter(
                    df, cat="chat", n=5, others=False, metric=metric, aggr="count"
                )
            else:
                df, n = cat_filter(df, cat=cat, metric=metric, n=2, others=True)
                if n < 2 and cat in levels:
                    levels.remove(cat)

    if filters:
        for cat, f in filters.items():
            df, n = cat_filter(df, cat=cat, metric=metric, **f)
            if cat == "chat":
                n_chat = n
            if n < 2 and cat in levels:
                levels.remove(cat)

    if subset:
        if isinstance(subset, list):
            for sub in subset:
                if not isinstance(sub, list):
                    sub = [sub]
                for subsub in sub:
                    if subsub not in cat_mapper.keys():
                        raise ValueError(
                            f"{subsub} is not a valid name of subset; valid values are {cat_mapper.keys()}"
                        )
                df = df[df[cat_mapper[sub[0]]].isin(sub)]
                if len(sub) == 1 and cat_mapper[sub[0]] in levels:
                    levels.remove(cat_mapper[sub[0]])

        elif isinstance(subset, str):
            if subset not in cat_mapper.keys():
                raise ValueError(
                    f"{subset} is not a valid name of subset; valid values are {cat_mapper.keys()}"
                )
            df = df[df[cat_mapper[subset]] == subset]
            if cat_mapper[subset] in levels:
                levels.remove(cat_mapper[subset])
        else:
            raise TypeError(f"expected list or str, got {type(subset)}")
        n_chat = df.chat.nunique()

    size = 1 / (len(levels) + 1)
    radius = len(levels) * n_chat > 10 and len(levels) * n_chat or 10
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (radius, radius)))

    if n_chat > 10:
        raise Exception("Too much data to be fitted to a single starburst chart")

    ax.pie(
        [1],
        labels=[l0_label],
        labeldistance=0,
        textprops={
            "fontsize": "15",
            "fontweight": "bold",
            "ha": "center",
            "va": "center",
        },
        radius=size,
    )
    if len(levels) > 0:

        order = df[levels[0]].value_counts().index
        vals = get_pie_vals(df, levels[:1], order)
        names = list(
            map(lambda name: get_display(arabic_reshaper.reshape(name)), order)
        )
        pie = ax.pie(
            vals,
            labels=None,
            radius=2 * size,
            colors=outer_colors[:n_chat],
            wedgeprops=dict(width=size, edgecolor="w"),
            autopct=label_value[0]
            and (lambda x: "{:.0f}".format(x * len(df) / 100))
            or None,
            textprops={"fontsize": 10, "ha": "center", "va": "center"},
            labeldistance=0.7,
            pctdistance=0.7,
            rotatelabels=True,
            counterclock=False,
            startangle=-270,
        )
        if label_name[0]:
            curved_labels(ax, vals, names, labeldistance=size * 2 * 0.85)
        rotate_text(pie, False, label_value[0], True)

    if len(levels) > 1:
        vals = get_pie_vals(df, levels[:2], order)
        l2_labels = sorted(df[levels[1]].unique())
        labels_ = label_corr(np.tile(l2_labels, n_chat), vals)
        l2_colors = inner_colors[: len(labels_)]
        pie = ax.pie(
            vals,
            labels=labels_ if label_name[1] else None,
            radius=3 * size,
            colors=l2_colors,
            wedgeprops=dict(width=size, edgecolor="w"),
            labeldistance=0.9,
            pctdistance=0.8,
            rotatelabels=True,
            counterclock=False,
            startangle=-270,
            autopct=label_value[1]
            and (lambda x: x and "{:.0f}".format(x * len(df) / 100) or None)
            or None,
            textprops={"fontsize": 10},
        )
        plt.setp(pie[1], rotation_mode="anchor", ha="center", va="center")
        rotate_text(pie, True, label_value[1])

    if len(levels) > 2:
        vals = get_pie_vals(df, levels[:3], order)
        l3_labels = [label.title() for label in sorted(df[levels[2]].unique())]
        labels_ = label_corr(np.tile(l3_labels, len(l2_labels) * n_chat), vals)

        if len(l3_labels) > 3 or len(l2_labels) * n_chat > 20:
            l3_colors = np.repeat(l2_colors, len(l3_labels), axis=0)
        else:
            l3_colors = [
                lighten(
                    inner_colors[i // len(l3_labels)],
                    0.25 + 0.25 * (i % len(l3_labels)),
                )
                for i in range(len(labels_))
            ]
        pie = ax.pie(
            vals,
            labels=labels_ if label_name[2] else None,
            radius=4 * size,
            colors=l3_colors,
            counterclock=False,
            startangle=-270,
            wedgeprops=dict(width=size, edgecolor="w", alpha=1),
            textprops={"fontsize": 10, "ha": "center", "va": "center"},
            labeldistance=0.9,
            pctdistance=0.8,
            rotatelabels=True,
            autopct=label_value[2]
            and (lambda x: x and "{:.0f}".format(x * len(df) / 100) or None)
            or None,
        )
        rotate_text(pie, False, label_value[2])

    if len(levels) > 3:
        vals = get_pie_vals(df, levels[:4], order)
        l4_labels = [label.title() for label in sorted(df[levels[3]].unique())]
        labels_ = label_corr(
            np.tile(l4_labels, len(l3_labels) * len(l2_labels) * n_chat), vals
        )
        if len(l4_labels) > 4:
            l4_colors = np.repeat(l3_colors, len(l4_labels), axis=0)
        else:
            colors = [
                lighten(
                    l3_colors[i // len(l4_labels)], 0.25 + 0.15 * (i % len(l4_labels))
                )
                for i in range(len(labels_))
            ]
        pie = ax.pie(
            vals,
            labels=labels_ if label_name[2] else None,
            radius=1,
            colors=colors,
            counterclock=False,
            startangle=-270,
            wedgeprops=dict(width=size, edgecolor="w", alpha=1),
            textprops={"fontsize": 10, "ha": "center", "va": "center"},
            labeldistance=0.91,
            pctdistance=0.85,
            rotatelabels=True,
            autopct=label_value[2]
            and (lambda x: x and "{:.0f}".format(x * len(df) / 100) or None)
            or None,
        )
        rotate_text(pie, False, label_value[3])
    hrchy = ""
    for cat in levels:
        hrchy += cat_display[cat][:-1] + ", "
    plt.title(
        f"{metric_display[metric]} proportion by {hrchy[:-2]}",
        fontsize=12,
        fontweight="bold",
    )
    return ax
