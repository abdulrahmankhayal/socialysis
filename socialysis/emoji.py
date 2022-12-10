import numpy as np
import pandas as pd
import emoji
import matplotlib.pyplot as plt
from imojify import imojify
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from .plot_helper import (
    cat_Vs_counts,
    dynamic_sort,
    make_pretty,
    cat_filter,
    dt_filter,
    freq_mapper,
    offset_image,
    title_freq,
    valid_cum_aggr,
)
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
from pandas.util._decorators import doc


def explode_emoji(txt):
    emojis = emoji.emoji_list(txt)
    return [emojis[i]["emoji"] for i in range(len(emojis))]


def emo_bar(s, n, label=True, **kwargs):
    emjis = s[:n].index
    values = s[:n].values

    fig, ax = plt.subplots(
        figsize=kwargs.get("figsize", (1.25 * n > 15 and 1.25 * n or 15, 10))
    )

    ax.bar(range(len(emjis)), values, width=0.9, align="center")
    ax.set_xticks(range(len(emjis)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="major", pad=26)
    tol = values.max() * 0.02
    for i, e in enumerate(emjis):
        offset_image([i, 0], e, ax, zoom=0.04)
        if label:
            ax.text(
                i,
                values[i] + tol,
                values[i],
                ha="center",
                rotation=kwargs.get("label_rot", 0),
            )
    if label:
        ax.set_ylim(top=tol * 55)
    plt.title(f"Top {n} used emoji combinations", fontsize=12, fontweight="bold")
    return ax


def emoji_filter(df, include=[], exclude=[], keep_var=True, strip_others=False):
    if include and exclude:
        raise Exception(f"include and exclude are mutually exclusive")
    if include:
        if isinstance(include, str):
            include = [include]
        df = df[
            (keep_var and df.emoji.str.contains("|".join(include)))
            | df.emoji.isin(include)
        ]
        s = df.set_index("emoji").squeeze()
        if not strip_others or not keep_var:
            return s

        include_set = set(include)
        s.index = s.index.map(
            lambda txt: [
                emoji["emoji"] if emoji["emoji"] in include_set else ""
                for emoji in emoji.emoji_list(txt)
            ]
        ).map("".join)
        return s.groupby(level=0).sum().sort_values(ascending=False)
    return (
        df.replace(exclude, "", regex=not keep_var)
        .replace("", np.NaN)
        .dropna()
        .set_index("emoji")
        .squeeze()
    )


@doc(
    descr="""The class that enables you to deal with chat Reacts data.
It contains a variety of Methods and Attributes that aim
to help you better understand the data about your chat Reacts.

It handels the Reacts of one single chat.""",
    other_attrs="",
    other_methods="",
    klass="react",
    other="",
)
class chat_emoji:
    """
    {descr}

    {other}Attributes
    ----------
    data : pandas.DataFrame
        The {klass}s row data
    count : int
        The count of all {klass}s
    emoji_list : pandas.Series
        List of all used {klass}s and their counts
    top_emoji : str
        The most used {klass}
    {other_attrs}

    {other}Methods
    -------
    most_used_emojis(...):
        Make bar chart of the most used {klass}s and their counts.
    emoji_timeline(...)
        Make timeline line chart/s for the Top used {klass}s.
    most_used_timeline(...)
        Make line chart showing the most used {klass} for every time period and its count.
    {other_methods}

    """

    def __init__(self, metric, passed_data):
        self._metric = metric
        metric = self._metric == "reactions" and self._metric or "emoji_single"
        self.data = passed_data
        self.count = self.data[metric].count()
        self.emoji_list = self.data[metric].value_counts()
        self.top_emoji = len(self.emoji_list) and self.emoji_list.index[0] or ""

    def most_used_emojis(
        self, n=10, show_sender=True, stacked=False, label=True, legend=True, **kwargs
    ):
        """
        Make bar chart of the most used Rects/Emojis and their counts.

        Parameters
        ----------
        n : int, default 10
            The number of emojis/reacts to show.
        show_sender : bool, default True
            Whether to show emoji/react sender or not.
            When True, 2 bars for each emoji are ploted, one for emojis you sent,
            and one for emojis your friends sent.
        stacked : bool, default False
            If show_sender, whether to stack senders' bars together or not.
        label : bool, default True
            Whether to show the text(annotation) of the bar values (heights).
        legend : bool, default True
            Whether to show the plot legend or not
            Only has effect when `show_sender=True`.
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.
        Returns
        -------
        matplotlib.axes.Axes
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        df = self.data.copy()
        metric = self._metric == "reactions" and self._metric or "emoji_single"
        df, n = cat_filter(df, n, cat=metric, metric="chat")
        return cat_Vs_counts(
            df,
            "chat",
            cat=metric,
            show_sender=show_sender,
            n=n,
            legend=legend,
            label=label,
            stacked=stacked,
            emoji=True,
            **kwargs,
        )

    def emoji_timeline(
        self,
        n=5,
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include=[],
        exclude=[],
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        subplots=False,
        **kwargs,
    ):
        """
        Make timeline line chart/s for the Top used Rects/Emojis.

        Parameters
        ----------
        n : int, default 5
            The number of emojis/reacts to show.
            Has no effect when `include` parameter is used.
        include : list-like objects, default []
            list of emojis/reacts to include.
        exclude : list-like objects, default []
            list of emojis/reacts to exclude.
        cumulative : bool, default False
            Whether to apply a cumulative aggregation or not.
            Used along with `window` paramter to select a moving or a running aggreagtion.
        cum_aggr : {"count","nunique","sum","mean","max","min","var","std"}, default 'sum'
            The aggregation to be applied for the cumulation.
        window : int or 'default', default 'default'
            Used to define the window within it the aggreagtion is done.
            By default it is the length of the data which calculate running aggregates
            Choose a suitable window length to calculate moving aggregates.
        dt_disc : bool, default False
            when True, datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range` to manipulate datetime
        dt_hrchy : {'minute','hour','day','week','month','quarter','year'}, default 'month'
            used to select the hierarchy(frequancy) of the datetime variable
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'
            use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        subplots : bool, default False
            used to create subplots, one per emoji/react for every selected emoji/react.
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        if dt_hrchy not in freq_mapper.keys():
            raise ValueError(
                f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
            )
        df = self.data.copy()
        metric = self._metric == "reactions" and self._metric or "emoji_single"
        if include or exclude:
            df, n = cat_filter(
                df,
                n=n,
                cat=metric,
                include=include,
                exclude=exclude,
                groups=[],
                others=False,
                asc=False,
                sort="fixed",
                metric="sender_name",
            )
        # if n>df[metric].nunique():
        #   raise
        if dt_range:
            df = dt_filter(df, dt_range)
        groupping_params = {
            True: "timestamp_ms",
            False: pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]),
        }
        if dt_disc:
            df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)

        df = df[["timestamp_ms", metric]]
        df = (
            df.groupby(groupping_params[dt_disc])[metric]
            .value_counts()
            .to_frame()
            .rename(columns={metric: "counts"})
            .reset_index()
            .pivot(index="timestamp_ms", columns=metric, values="counts")
            .T
        )
        df = df.sort_index(key=df.sum(1).get, ascending=False)[:n].T.fillna(0)
        if cumulative:
            if cum_aggr not in valid_cum_aggr:
                raise ValueError(
                    f"{cum_aggr} is not a valid aggregation function for cumulation; valid aggr are {valid_cum_aggr}"
                )
            if window == "default":
                window = len(df)
            df = df.rolling(window=window, min_periods=1).agg(cum_aggr)
        if n > len(df.columns):
            n = len(df.columns)
        if len(df.columns) < 2:
            subplots = False
        if subplots:
            fig, axes = plt.subplots(
                nrows=n,
                ncols=1,
                figsize=kwargs.get("figsize", (15, 10)),
                sharex=True,
                sharey=True,
            )
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
        else:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=kwargs.get("figsize", (15, 10))
            )
        x = df.index
        for i in range(n):
            if subplots:
                ax = axes[i]
            y = df[df.columns[i]]
            image = plt.imread(imojify.get_img_path(y.name))
            image_box = OffsetImage(image, zoom=0.05)
            for x0, y0 in zip(x, y):
                ab = AnnotationBbox(image_box, (x0, y0), frameon=False,)
                ax.add_artist(ab)

            ax.plot(x, y, label="")
        type_title = metric == "reactions" and "reacts" or "emojis"
        cum_subtitle_start = cumulative and "Cumulative" or ""
        cum_subtitle_end = (
            cumulative
            and (
                window == len(df)
                and f"(Running {cum_aggr.upper()})"
                or f"(Moving {cum_aggr.upper()} with window = {window})"
            )
            or ""
        )
        if subplots:
            fig.suptitle(
                f"{cum_subtitle_start} {title_freq[dt_hrchy].title()} Timeline of Top {n} used {type_title} {cum_subtitle_end}",
                y=0.92,
                fontsize=12,
                fontweight="bold",
            )
        else:
            plt.title(
                f"{cum_subtitle_start} {title_freq[dt_hrchy].title()} Timeline of Top {n} used {type_title} {cum_subtitle_end}",
                fontsize=12,
                fontweight="bold",
            )
        return ax

    def most_used_timeline(
        self, dt_disc=False, dt_hrchy="month", dt_range="last year", **kwargs
    ):
        """
        Make a single line chart showing the most used emoji/react
        for every time period and its count.
        
        The data points of the line chart are the count of the most
        used emoji/react in each selected dt_hrchy, and its markers are these emojis.


        Parameters
        ----------
        dt_disc : bool, default False
            when True, datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range` to manipulate datetime
        dt_hrchy : {'minute','hour','day','week','month','quarter','year'}, default 'month'
            used to select the hierarchy(frequancy) of the datetime variable
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'
            use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
        Returns
        -------
        matplotlib.axes.Axes 
        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        if dt_hrchy not in freq_mapper.keys():
            raise ValueError(
                f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
            )
        df = self.data.copy()

        metric = self._metric == "reactions" and self._metric or "emoji_single"
        if dt_range:
            df = dt_filter(df, dt_range)
        groupping_params = {
            True: ["timestamp_ms"],
            False: [pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy])],
        }
        if dt_disc:
            df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
        df = (
            df.groupby(groupping_params[dt_disc])[metric]
            .agg(lambda x: Counter("".join(x)).most_common(1)[:1])
            .explode()
            .dropna()
            .apply(lambda loc: pd.Series(loc))
            .rename(columns={0: metric, 1: "counts"})
        )
        x = df.index
        y = df["counts"]
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (15, 10)))

        for x0, y0 in zip(x, y):
            image = plt.imread(imojify.get_img_path((df.loc[x0][metric])))
            image_box = OffsetImage(image, zoom=0.05)
            ab = AnnotationBbox(image_box, (x0, y0), frameon=False,)
            ax.add_artist(ab)
        ax.set_ylim(bottom=0, top=max(y) * 1.1)
        ax.plot(x, y, label="")
        type_title = metric == "reactions" and "react" or "emoji"
        plt.title(
            f"The most used {type_title} over {dt_hrchy}s",
            fontsize=12,
            fontweight="bold",
        )
        return ax


@doc(
    chat_emoji,
    descr="""The class that enables you to deal with all chats Reacts data
It contains a variety of Methods and Attributes that aim
to help you better understand the data about your Reacts across all chats.""",
    other_attrs="",
    other_methods="""emoji_Vs_chat(...)
    Make a bar chart describing your Top reacts across Top chats.
single_emoji_per_chats(emoji,n=10,**kwargs)
    Make a bar chart showing your react across chats.
    """,
    klass="react",
    other="",
)
class all_emoji(chat_emoji):
    def emoji_Vs_chat(
        self,
        n_emoji=5,
        include_emoji=[],
        exclude_emoji=[],
        emoji_groups={},
        n_chats=5,
        include_chat=[],
        exclude_chat=[],
        chat_groups={},
        others=False,
        return_data=False,
        style=False,
        label=True,
        legend=True,
        sort="fixed",
        **kwargs,
    ):
        """
        Make a bar chart describing your Top emojis/reacts across Top chats.

        Parameters
        ----------
        n_emoji : int, default 5
            Max number of emojis/reacts to show.
            Has no effect when include_emoji is used.
        include_emoji : list-like object, default []
            List of the only emojis/reacts to show.
        exclude_emoji : list-like object, default []
            List of emojis to exclude.
        emoji_groups : dict, default {}
            Used to create emoji/reacts groups like: 
            {'😂':'Happy', '😄':'Happy','😓':'Sad', '😢':'Sad' }.
            When used, you must specify the group to which every emoji 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        n_chats : int, default 5
            Number of chats to show.
            Has no effect when include_chat is used or sort='dynamic'.
        include_chat : list-like object, default []
            List of the only chats to include.
        exclude_chat : list-like object, default []
            List of chats to exclude.
        chat_groups : dict, default {}
            Used to make chat groups the same way as emoji_groups. 
        others : bool, default False
            when True, group any chat/emojis other than the selected n 
            or the included ones and label them as 'Others'
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table.
        label : bool, default True
            whether to show the text(annotation) of the bar values (heights)
        legend : bool, default True
            whether to show the plot legend or not.
        sort : {'fixed', 'dynamic'}, default 'fixed'
            Used to select a 'fixed' `n` chats for all emojis,
            who are The Top people generally,
            or select them 'dynamic'ally , each emoji with its own Top `n` chats.
            Has no effect when include is used.
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.
        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        metric = self._metric == "reactions" and self._metric or "emoji_single"

        df = self.data.copy()
        # emoji filter
        df, n_emoji = cat_filter(
            df,
            n=n_emoji,
            cat=metric,
            include=include_emoji,
            exclude=exclude_emoji,
            groups=emoji_groups,
            others=others,
            asc=False,
            sort="fixed",
            metric="sender_name",
        )

        # chat_filter
        df, n = cat_filter(
            df,
            n=n_chats,
            cat="chat",
            include=include_chat,
            exclude=exclude_chat,
            groups=chat_groups,
            others=others,
            sort=sort,
            metric="sender_name",
        )

        df = (
            df.groupby([metric, "chat"])
            .agg({"sender_name": "count"})
            .reset_index()
            .pivot(index=metric, columns="chat", values="sender_name")
        )
        df = df.sort_index(key=df.sum(1).get, ascending=False).fillna(0)
        if return_data:
            if style:
                return df[:n].style.pipe(make_pretty)
            return df
        axes, fig = dynamic_sort(
            df[:n_emoji],
            n=n,
            label=label,
            legend=legend,
            width=3 * n_emoji,
            loc="upper center",
            ncol=len(df.columns),
            bbox_to_anchor=(0.5, 0.95),
            emoji=True,
            **kwargs,
        )
        return axes

    def single_emoji_per_chats(self, emoji, n=10, **kwargs):
        """
        Make a bar chart showing your emoji/react across chats.

        Parameters
        ----------
        emoji : str
            The emoji/react you want to inspect.
        n : int, default 10
            Number of chats to show.
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return self.emoji_Vs_chat(
            include_emoji=[emoji], n_chats=n, others=False, label=True, **kwargs
        )


@doc(
    chat_emoji,
    descr="""The class that enables you to deal with the emojis contained in your chat text messages.
It contains a variety of Methods and Attributes that aim
to help you gain insights about emojis you and your friend used,
whether in the level of a single emoji or a combinations of them.""",
    other_attrs="""emoji_per : float
    The emoji to text ratio of this chat.
emoji_combo : pd.Series
    All of your emoji combos and their count.""",
    other_methods="""most_used_emoji_combo(...)
    Make a bar chart of Top used emoji combinations.
single_emoji_var(emo,return_data=False,label=True,**kwargs)
    Make a bar chart of the counts of each distinct length an emoji.""",
    klass="emoji",
    other="",
)
class emojis(chat_emoji):
    def __init__(self, passed_data):
        self.data = passed_data.join(
            passed_data.emoji.dropna()
            .map(explode_emoji)
            .explode(),  # .replace(["",'️',' '],np.NaN).dropna()
            rsuffix="_single",
            how="inner",
        )[["timestamp_ms", "chat", "sender_name", "emoji", "emoji_single"]].dropna(
            subset="emoji_single"
        )
        super().__init__("emoji", self.data)
        self.emoji_per = (
            len(self.emoji_list)
            and round(
                len("".join(passed_data.emoji.dropna()))
                / len("".join(passed_data.content.dropna()))
                * 100,
                2,
            )
            or 0
        )

        # self.emoji_list=pd.Series([emo['emoji'] for emo in emoji.emoji_list(''.join(self.data.emoji))],dtype='object').value_counts()
        self.emoji_combo = (
            passed_data.groupby("emoji").size().sort_values(ascending=False)
        )

    def most_used_emoji_combo(
        self,
        n=10,
        include=[],
        exclude=[],
        keep_var=True,
        strip_others=False,
        label=True,
        return_data=False,
        **kwargs,
    ):
        """
        Make a bar chart of Top used emoji combinations(emojis used together).

        Parameters
        ----------
        n : int, default 10
            number of emoji combos to show
        include : str or list-like object, default []
            emojis to keep, single emoji & emoji combos are allowed,
            for example: '😂' / '😂❤️' / ['😂','❤️'] / ['😂','😂❤️']
            Note that '😂❤️' not the same as ['😂','❤️'],
            as the '😂❤️' is a pattern that will be mattched,
            while  ['😂','❤️'] isn't
        exclude : str or list-like object, default []
            emojis to exclude
        keep_var : bool, default True
            keep the other variation of the emoji,
            for example, if you included '😂',
            '😂😂' and '😂😂😂'will be keept,
             and for exclude, if you excluded '😂',
            '😂😂' and '😂😂😂' will be kept.
        strip_others : bool, default False
            When include, you can choose between strip any other emoji than
            the included ones from the emoji combo or not.
            for example, if the emoji combo is '😂❤️'
            and you included only ['😂'], '😂❤️' will be keept as it is,
            but if strip_others, '❤️' will be stripped.
        label : bool, default True
            whether to show the text(annotation) of the bar values (heights).
        return data : bool, default False
            return the filtered data with no plots
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.

        Returns
        -------
        matplotlib.axes.Axes
        """
        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "label_rot"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        s = self.emoji_combo.copy()
        if include or exclude:
            s = emoji_filter(
                s.reset_index(),
                include,
                exclude,
                keep_var=keep_var,
                strip_others=strip_others,
            )
        if len(s) < 1:
            raise ValueError("Filters you used resulted in empty data")
        if return_data:
            return s
        ax = emo_bar(s, n=n, label=label, **kwargs)

        return ax

    def single_emoji_var(self, emo, return_data=False, label=True, **kwargs):
        """
        Make a bar chart of the counts of each distinct length an emoji.

        distinct length an emoji means that a single emoji could be sent in different counts,
        for example '😂😂' and '😂😂😂' are varations of '😂' but with different lengths.
        
        So if you called this method on '😂', the number of times a message contained only '😂',
        the number of times a message contained only '😂😂'and so on until the max length of '😂' sent in 
        any message, will be plotted as bars, bar for each different length
        with a bar height represent the count of their occurrence.


        Parameters
        ----------
        emo : str
            The emoji to find its variations.
        return_data : bool, defualt Flase
            Returns the plotting data.
            Could be helpful if the data is too big to be fitted into a single chart.
        label : bool, default True
            Whether to show the text(annotation) of the bar values (heights) 
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.

        Returns
        -------
        matplotlib.axes.Axes
        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "label_rot"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        if not emoji.emoji_list(emo):
            raise ValueError(f"{emo} is not an emoji")
        if not emoji.is_emoji(emo):
            raise ValueError(
                "you may have entered more than one emoji or an emoji combined with text"
            )
        df = (
            self.data[~self.data.index.duplicated(keep="first")]
            .drop(columns=["emoji_single"])
            .copy()
        )
        single_emo = df.emoji.str.extract(f"({emo}+)").dropna()
        if not len(single_emo):
            raise ValueError(f"{emo} not found in your messages")
        single_emo["count"] = single_emo[0].apply(len)
        single_emo = single_emo.sort_values("count").groupby("count").count()
        if return_data:
            return single_emo
        ax = single_emo.plot.bar(
            figsize=kwargs.get(
                "figsize",
                (len(single_emo) * 0.6 > 15 and len(single_emo) * 0.6 or 15, 10),
            ),
            legend=False,
            width=0.9,
        )
        if label:
            tol = single_emo.max().values[0] * 0.02
            for idx, row in single_emo.reset_index().iterrows():
                ax.text(
                    idx,
                    row[0] + tol,
                    row[0],
                    ha="center",
                    rotation=kwargs.get("label_rot", 0),
                )
            ax.set_ylim(top=tol * 55)
        plt.title(
            f"The Count of different lengths of the emoji",
            fontsize=12,
            fontweight="bold",
        )
        return ax


@doc(
    chat_emoji,
    descr="",
    other_attrs="",
    other_methods="",
    klass="emoji",
    other="Other ",
)
class Multi_emojis(emojis, all_emoji):
    """
    The class that enables you to deal with the emojis in all of your messages.
    It contains a variety of Methods and Attributes that aim to help you gain insights
    about all of the emojis used in all chats,
    whether in the level of a single emoji or a combinations of them.

    Attributes
    ----------
    emoji_per : float
        The total emoji to text ratio of all of your chats.
    emoji_combo : pd.Series
        All of your emoji combos and their count.
    most_emoji_ratio : pd.Series
        Emoji to text ratio for each chat, sorted.

    Methods
    -------
    emoji_to_content_ratio(...)
        Make a bar chart of chats with the highest emoji to text ratio.
    most_used_emoji_combo(...)
        Make a bar chart of Top used emoji combinations.
    single_emoji_var(emo,return_data=False,label=True,**kwargs)
        Make a bar chart of the counts of each distinct length an emoji
    emoji_Vs_chat(...)
        Make a bar chart describing your Top emojis across Top chats.
    single_emoji_per_chats(emoji,n=10,**kwargs)
        Make a bar chart showing your emojis across chats.
    """

    def __init__(self, passed_data):
        super().__init__(passed_data)
        self.most_emoji_ratio = (
            passed_data.dropna(subset="content")
            .groupby("chat")[["content", "emoji"]]
            .apply(
                lambda ct: round(
                    len("".join(ct.emoji.dropna()))
                    / len("".join(ct.content.dropna()))
                    * 100,
                    2,
                )
            )
            .sort_values(ascending=False)
        )

    def emoji_to_content_ratio(
        self, n=10, include=[], exclude=[], label=True, **kwargs
    ):
        """
        Make a bar chart of chats with the highest emoji to content(messages text) ratio.

        Parameters
        ----------
        n : int, default 10
            Number of chats to show.
            Has no effect when include is used.
        include : list-like object, default []
            List of the only chats to include.
        exclude : list-like object, default []
            List of chats to exclude
        
        Returns
        -------
        matplotlib.axes.Axes
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        s = self.most_emoji_ratio
        if include:
            s = s[s.index.isin(include)]
        if exclude:
            s = s[~s.index.isin[exclude]]
        s = s[:n]
        s.index = s.index.map(lambda name: get_display(arabic_reshaper.reshape(name)))
        ax = s.plot.bar(
            xlabel="",
            figsize=kwargs.get("figsize", (n < 15 and 15 or n, 10)),
            rot=kwargs.get("rot", 45),
            width=0.9,
        )
        if label:
            tol = s.max() * 0.02
            for idx, item in s.reset_index().iterrows():
                ax.text(
                    idx,
                    item.values[1] + tol,
                    round(item.values[1], 2),
                    ha="center",
                    rotation=kwargs.get("label_rot", 0),
                )
            ax.set_ylim(top=tol * 55)
        plt.title(f"emoji-to-word ratio per chat", fontsize=12, fontweight="bold")
        return ax
