import pandas as pd
import numpy as np
import os
import emoji
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from imojify import imojify

cwd = os.path.dirname(__file__)
import json


def get_meta_data(filename="meta_data.json"):
    try:
        with open(os.path.join(cwd, filename), "r") as openfile:

            # Reading from json file
            meta_data = json.load(openfile)
    except:
        meta_data = {}
    return meta_data


def offset_image(coords, name, ax, zoom=0.04):
    img = plt.imread(imojify.get_img_path(name))
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    ab = AnnotationBbox(
        im,
        (coords[0], coords[1]),
        xybox=(0.0, -16.0),
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0,
    )

    ax.add_artist(ab)


stck_dflts = {
    "count": True,
    "nunique": True,
    "sum": True,
    "mean": False,
    "max": False,
    "min": False,
    "first": False,
    "last": False,
    "var": False,
    "std": False,
}
sort_ops = {True: "sum", False: "mean"}
freq_mapper = {
    "day": "D",
    "week": "W",
    "month": "MS",
    "quarter": "Q",
    "year": "YS",
    "hour": "H",
    "minute": "T",
}
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
nxtlvlofdtils = {"year": "month", "quarter": "month", "month": "day", "week": "day"}
valid_dt = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "microseconds",
    "nanoseconds",
]
metric_categories = {
    "photos": ["chat"],
    "gifs": ["chat"],
    "files": ["chat", "ext"],
    "sticker": ["chat"],
    "audio_files": ["chat"],
    "videos": ["chat"],
    "media_type": ["chat", "media"],
    "content": ["chat"],
    "call_duration": ["chat"],
    "call_type": ["chat"],
    "share": ["chat", "domain"],
    "chat": ["sender_name"],
    "index": ["chat"],
}
title_freq = {
    "year": "Yearly",
    "quarter": "Quarterly",
    "month": "Monthly",
    "week": "Weekly",
    "day": "Daily",
    "hour": "Hourly",
    "minute": "Minutely",
}
metric_display = {
    "media_type": "All Media",
    "index": "Messages",
    "audio_files_uri": "Audio Files",
    "audio_files_length": "Audio duration",
    "call_type": "Calls",
    "call_duration": "Call duration",
    "share": "Shared links",
    "content": "Text messages",
    "chat": "Chat messages",
    "video_uri": "Videos",
    "video_length": "Video duration",
    "files": "Files",
    "sticker": "Stickers",
    "gifs": "Gifs",
    "photos": "Photos",
}
cat_display = {
    "chat": "Chats",
    "media": "Meida types",
    "ext": "File extensions",
    "domain": "Domains",
    "sender_name": "Senders",
    "call_type": "Call type ",
    "call_status": "Call status ",
}
valid_cum_aggr = ["count", "nunique", "sum", "mean", "max", "min", "var", "std"]


def make_pretty(styler):
    styler.background_gradient(axis=None, vmin=1, cmap="Blues")
    styler.set_properties(**{"width": "10em", "text-align": "center"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )
    return styler


def metric_correct(metric, aggr):
    if callable(aggr):
        if metric in ["audio_files", "video"]:
            return metric + "_length"
        return metric
    if aggr not in stck_dflts.keys():
        raise ValueError(
            f"{aggr} is not a valid aggregation function; supported aggr are {list(stck_dflts.keys())}"
        )
    if metric in ["audio_files", "video"]:
        if aggr in ["count", "nunique"]:
            metric = metric + "_uri"
        else:
            metric = metric + "_length"
    elif metric == "call_duration":
        if aggr in ["count", "nunique"]:
            metric = "call_type"
    if aggr not in ["count", "nunique"] and metric not in [
        "call_duration",
        "audio_files_length",
        "video_length",
    ]:
        raise ValueError(
            f"{aggr} is not a supported value for {metric}; supported aggregation are 'count' and 'nunique' "
        )
    return metric


# edit for diffrent agg than count
def cat_filter(
    df,
    n=5,
    cat="chat",
    include=[],
    exclude=[],
    others=False,
    groups={},
    asc=False,
    sort="fixed",
    metric="",
    aggr="count",
):
    df = df.copy()
    # print(f'include :{include},exclude : {exclude},others : {others},cat :{cat}')
    if include:

        if include == "Top" and cat == "chat":
            meta_data = get_meta_data()
            include = meta_data["Top"]
        elif np.ndim(include) == 0:
            raise TypeError(f"{include} is not list-like object")
        if others:
            df[cat] = df[cat].apply(lambda name: name if name in include else "Others")
            n = df[cat].nunique() + 1

        else:
            df = df[df[cat].isin(include)]
            n = df[cat].nunique()
        if n == 0:
            raise ValueError(f"{cat} do not contain any of {include} ")

    else:

        if n in ["All", 0, None]:
            n = df[cat].nunique()
        elif not isinstance(n, int):
            raise TypeError(f"expected an integer, got{type(n)}")
        elif n and sort == "fixed":
            if n < 0:
                n = df[cat].nunique() + n
            top_n = (
                df.groupby(cat)[metric].agg(aggr).sort_values(ascending=asc).index[:n]
            )
            if n > len(top_n):
                n = len(top_n)
            if others:
                df[cat] = df[cat].apply(
                    lambda name: name if name in top_n else "Others"
                )
                n = n + 1
            else:
                df = df[df[cat].isin(top_n)]

    if exclude:
        if np.ndim(exclude) == 0:
            raise TypeError(f"{exclude} is not list-like object")

        df = df[~df[cat].isin(exclude)]
        n = df[cat].nunique()
        if n == 0:
            raise ValueError(f"{cat} do not contain any of {include} ")

    if groups:
        if not isinstance(groups, dict):
            raise TypeError(f"{groups} is not a dict ")
        df[cat] = df[cat].map(lambda name: groups.get(name, "Others"))
        n = df[cat].nunique()
    return df, n


def dt_filter(df, dt_range):

    before = df.timestamp_ms.min()
    after = df.timestamp_ms.max()
    if isinstance(dt_range, str):
        tokens = dt_range.split()
        if len(tokens) < 2:
            raise Exception(f"{dt_range} is not a valid dt_range")
        if tokens[0].lower() in ["first", "last"]:
            if len(tokens) == 2:
                tokens[1] = tokens[1].endswith("s") and tokens[1] or tokens[1] + "s"
                if tokens[1] not in valid_dt:
                    raise Exception(
                        f"{tokens[1]} is not a valid freq, valid freqs : {valid_dt}"
                    )
                DateOffset = pd.DateOffset(**{tokens[1]: 1})
            elif len(tokens) == 3:
                if not tokens[1].isnumeric():
                    raise ValueError(f"{tokens[1]} is not a number")
                d_val = tokens[1]
                tokens[2] = tokens[2].endswith("s") and tokens[2] or tokens[2] + "s"
                if tokens[2] not in valid_dt:
                    raise Exception(
                        f"{tokens[2]} is not a valid freq, valid freqs : {valid_dt}"
                    )
                DateOffset = pd.DateOffset(**{tokens[2]: int(d_val)})
            else:
                raise Exception(f"{dt_range} is not a valid dt_range")
            if tokens[0].lower() == "first":
                first_d = before
                dt_range = [first_d, first_d + DateOffset]
            else:
                last_d = after
                dt_range = [last_d - DateOffset, last_d]
        else:
            raise Exception(
                "A valid string should starts with either 'first' or 'last' "
            )

    if dt_range[0] and dt_range[0] != "Start":
        try:
            before = pd.to_datetime(dt_range[0])
        except:
            raise ValueError(f"{dt_range[0]} is not a valid datetime format")
    if dt_range[1] and dt_range[1] != "End":
        try:
            after = pd.to_datetime(dt_range[1])
        except:
            raise ValueError(f"{dt_range[1]} is not a valid datetime format")
    df = df[(df.timestamp_ms >= before) & (df.timestamp_ms <= after)]
    if not len(df):
        raise ValueError("not data to plot for the date range you specified")
    return df


def label_counts(ax, df, stacked, show_sender, **kwargs):

    if stacked:
        for idx, row in df.reset_index().iterrows():
            if row[df.columns[0]]:
                ax.text(
                    idx,
                    row[df.columns[0]] / 2,
                    row[df.columns[0]],
                    rotation=kwargs.get("label_rot", 90),
                    fontsize=10,
                    ha="center",
                )
            try:
                if row[df.columns[1]]:
                    ax.text(
                        idx,
                        row[df.columns[0]] + (row[df.columns[1]]) / 2,
                        row[df.columns[1]],
                        rotation=kwargs.get("label_rot", 90),
                        fontsize=10,
                        ha="center",
                    )
            except:
                pass
        ax.set_ylim(top=df.sum(axis=1).max() * 1.1)
    else:
        tol = df.max().max() * 0.02
        if not show_sender:

            for idx, row in df.reset_index().iterrows():
                if row[1]:
                    ax.text(
                        idx,
                        row[1] + tol,
                        row[1],
                        rotation=kwargs.get("label_rot", 0),
                        fontsize=10,
                        ha="center",
                    )
        else:

            for idx, row in df.reset_index().iterrows():
                if row[df.columns[0]]:
                    ax.text(
                        "emoji" in kwargs.keys() and idx - 0.2125 or idx - 0.125,
                        row[df.columns[0]] + tol,
                        row[df.columns[0]],
                        rotation=kwargs.get("label_rot", 90),
                        fontsize=10,
                        ha="center",
                    )
                try:
                    if row[df.columns[1]]:
                        ax.text(
                            "emoji" in kwargs.keys() and idx + 0.2125 or idx + 0.125,
                            row[df.columns[1]] + tol,
                            row[df.columns[1]],
                            rotation=kwargs.get("label_rot", 90),
                            fontsize=10,
                            ha="center",
                        )
                except:
                    pass
        ax.set_ylim(top=tol * 55)
    # plt.xticks(rotation=45);
    return ax


def top_timely(
    data,
    metric,
    aggr,
    cat,
    n=20,
    dt_hrchy="month",
    dt_range="last year",
    show_name=False,
    legend=True,
    rank=False,
    **kwargs,
):

    df = data.copy()
    if dt_range:
        df = dt_filter(data, dt_range)
    df = (
        df.groupby([pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]), cat])
        .agg({metric: aggr})
        .reset_index()
    )
    df = (
        df.groupby("timestamp_ms")
        .agg({metric: "max"})
        .reset_index()
        .merge(on=["timestamp_ms", metric], right=df)
    )
    if dt_hrchy in ["hour", "minute"]:
        df.timestamp_ms = df.timestamp_ms.dt.floor(freq_mapper[dt_hrchy])
    else:
        df.timestamp_ms = df.timestamp_ms.dt.date
    if rank:
        df.sort_values(metric, inplace=True, ascending=True)
    if n == None or n == "All":
        n = len(df)
    df = df[-n:].pivot(index="timestamp_ms", columns=cat, values=metric)
    df.columns = df.columns.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    if not rank:
        df = df[-n:].dropna(thresh=1, axis=1)
        ax = df.plot.bar(
            figsize=kwargs.get("figsize", (0.6 * n >= 15 and 0.6 * n or 15, 10)),
            width=0.95,
            stacked=True,
            xlabel="",
            legend=legend,
        )
        plt.xticks(rotation=kwargs.get("rot", 45))
        tol = df.sum(1).max() * 0.01

        for row in df.reset_index().iterrows():
            name = row[1].dropna().keys().values[1:]
            val = row[1].dropna()[name[:1]]
            if show_name:
                ax.text(
                    row[0] + 0.2,
                    tol,
                    ", ".join(name),
                    color="black",
                    fontsize=10,
                    rotation=kwargs.get("label_rot", 90),
                    ha="center",
                    va="bottom",
                )
                ax.text(
                    row[0] - 0.2,
                    tol,
                    int(val),
                    color="black",
                    fontsize=10,
                    rotation=kwargs.get("label_rot", 90),
                    ha="center",
                    va="bottom",
                )
            else:
                ax.text(
                    row[0],
                    val + tol,
                    int(val),
                    color="blue",
                    fontsize=10,
                    ha="center",
                    rotation=kwargs.get("label_rot", 0),
                )
        ax.set_ylim(top=df.sum(1).max() * 1.1)
        plt.ylabel(f"COUNT OF {metric_display[metric].upper()}", fontdict={"size": 12})

        plt.title(
            f"Highest {title_freq[dt_hrchy]} {aggr.title()} of {metric_display[metric]} By {cat_display[cat][:-1]} -Timeline",
            fontdict={"weight": "bold", "size": 12},
        )
        if legend:
            plt.legend(ncol=len(df.columns), loc="upper center")
    else:
        df = df.sort_index(key=df.sum(1).get)
        ax = df.plot.barh(
            stacked=True,
            legend=legend,
            figsize=kwargs.get("figsize", (15, 0.6 * n >= 10 and 0.6 * n or 10)),
            ylabel="",
            xlabel="",
            width=0.9,
            rot=kwargs.get("rot"),
        )
        tol = df.sum(1).max() * 0.005
        for row in df.reset_index().iterrows():
            name = row[1].dropna().keys().values[1:]
            val = row[1].dropna()[name[:1]]
            ax.text(
                val + tol,
                row[0] + 0.25,
                ", ".join(name),
                color="blue",
                fontsize=10,
                ha="left",
                va="center",
                rotation=kwargs.get("label_rot", 0),
            )
            ax.text(
                val + tol,
                row[0] - 0.25,
                int(val),
                color="blue",
                fontsize=10,
                ha="left",
                va="center",
                rotation=kwargs.get("label_rot", 0),
            )
        ax.set_xlim(right=df.sum(1).max() * 1.15)
        # plt.legend(prop={'weight':'bold'})
        plt.xlabel(f"COUNT OF {metric_display[metric].upper()}", fontdict={"size": 12})
        plt.title(
            f"Highest {title_freq[dt_hrchy]} {aggr.title()} of {metric_display[metric]} By {cat_display[cat]}(Ranked)",
            fontdict={"weight": "bold", "size": 12},
        )

        # plt.title(f'Ranked Most {title_freq[dt_hrchy] } {metric_display[metric]} By {cat.title()}',fontdict={'weight':'bold','size':12})
        ax.yaxis.set_ticks_position("none")
    # plt.yticks(weight = 'bold')
    # plt.xticks(weight = 'bold')

    return ax


def sum_of_most(data, metric, aggr, cat, n=10, dt_hrchy="day", **kwargs):
    df = data.copy()
    df = (
        df.groupby([pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]), cat])
        .agg({metric: aggr})
        .reset_index()
    )
    df = (
        df.groupby("timestamp_ms")
        .agg({metric: "max"})
        .reset_index()
        .merge(on=["timestamp_ms", metric], right=df)
    )
    df.sort_values(metric, inplace=True, ascending=True)
    df = pd.DataFrame(df[[cat]].value_counts()[:n]).reset_index()
    df[cat] = df[cat].map(lambda name: get_display(arabic_reshaper.reshape(name)))
    ax = df.plot.bar(
        x=cat,
        y=0,
        color=plt.cm.tab20(range(n)),
        figsize=kwargs.get("figsize", (n > 15 and n or 15, 10)),
        legend=False,
        xlabel="",
        rot=kwargs.get("rot", 45),
        width=0.9,
    )
    tol = df[0].max() * 0.02
    for row in df.iterrows():

        val = row[1][0]
        if val:
            ax.text(
                row[0],
                val + tol,
                int(val),
                color="blue",
                fontsize=10,
                ha="center",
                va="center",
                rotation=kwargs.get("label_rot", 0),
            )

    plt.title(
        f"The Total Number of Times a {cat_display[cat][:-1]} Had The Highest {title_freq[dt_hrchy]} {aggr.title()} of {metric_display[metric]}",
        fontdict={"weight": "bold", "size": 12},
    )
    # The total number of times a chat had the highest daily count of messages
    return ax


def most_per_chat(data, metric, aggr, n=10, dt_hrchy="day", **kwargs):
    df = data.copy()
    df = (
        df.groupby([pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]), "chat"])
        .agg({metric: aggr})
        .reset_index()
    )
    df = (
        df.groupby("chat")[[metric]]
        .max()
        .sort_values(metric, ascending=False)[:n]
        .reset_index()
    )
    df.chat = df.chat.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    # return df
    ax = df.plot.bar(
        x="chat",
        y=metric,
        color=plt.cm.tab20(range(n)),
        figsize=kwargs.get("figsize", (n > 15 and n or 15, 10)),
        legend=False,
        xlabel="",
        rot=kwargs.get("rot", 45),
        width=0.9,
    )
    tol = df[metric].max() * 0.02
    for row in df.iterrows():
        val = row[1][metric]
        if val:
            ax.text(
                row[0],
                val + tol,
                int(val),
                color="blue",
                fontsize=10,
                ha="center",
                va="center",
                rotation=kwargs.get("label_rot", 0),
            )
    plt.title(
        f"Most #Of {metric_display[metric]} Sent In One {dt_hrchy.title()} Per Chat",
        fontdict={"weight": "bold", "size": 12},
    )
    plt.ylabel(f"COUNT OF {metric_display[metric].upper()}", fontdict={"size": 12})
    return ax


def dt_Vs_counts(
    df,
    metric,
    aggr="count",
    cumulative=False,
    cum_aggr="sum",
    window="default",
    dt_disc=False,
    dt_hrchy="month",
    dt_groups={},
    show_sender=True,
):
    if dt_hrchy not in freq_mapper.keys():
        raise ValueError(
            f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
        )

    meta_data = get_meta_data()

    groupping_params = {
        True: ["timestamp_ms", "sender_name"],
        False: [
            pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]),
            "sender_name",
        ],
    }
    if df.sender_name.nunique() > 2:
        df["sender_name"] = df["sender_name"].apply(
            lambda name: name if name in [meta_data["user"]] else "Others"
        )
    if dt_disc:

        df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
        if dt_groups:
            df.timestamp_ms = df.timestamp_ms.map(
                lambda dt: dt_groups.get(dt, "Others")
            )

    if show_sender:
        df = (
            df.groupby(groupping_params[dt_disc])
            .agg({metric: aggr})
            .reset_index()
            .pivot(index="timestamp_ms", columns="sender_name", values=metric)
        )
    else:
        df = df.groupby(groupping_params[dt_disc][0]).agg({metric: aggr})

    if not dt_disc:
        df = df.reindex(
            pd.date_range(df.index[0], df.index[-1], freq=freq_mapper[dt_hrchy])
        ).fillna(0)
        df.index = df.index.date
    df = df.astype("int")
    if cumulative:
        if cum_aggr not in valid_cum_aggr:
            raise ValueError(
                f"{cum_aggr} is not a valid aggregation function for cumulation; valid aggr are {valid_cum_aggr}"
            )
        if window == "default":
            window = len(df)

        df = df.rolling(window=window, min_periods=1).agg(cum_aggr)
        df = df.apply(round, args=[2])
    return df


def dt_Vs_counts_bar(
    df,
    metric,
    aggr="count",
    cumulative=False,
    cum_aggr="sum",
    window="default",
    dt_disc=False,
    dt_hrchy="month",
    dt_groups={},
    show_sender=True,
    label=True,
    stacked=False,
    return_data=False,
    style=False,
    legend=True,
    **kwargs,
):
    df = dt_Vs_counts(
        df,
        metric,
        aggr,
        cumulative,
        cum_aggr,
        window,
        dt_disc,
        dt_hrchy,
        dt_groups,
        show_sender,
    )

    if not show_sender:
        stacked = False
        legend = False
    if return_data:
        if style:
            return df.style.pipe(make_pretty)
        return df

    df.columns = df.columns.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    # df.index=df.index.map(lambda name:get_display(arabic_reshaper.reshape(name)))
    ax = df.plot.bar(
        figsize=kwargs.get("figsize", (len(df) // 10 and len(df) or 10, 10)),
        stacked=stacked,
        legend=legend,
        xlabel="",
    )
    if label:
        ax = label_counts(ax, df, stacked, show_sender, **kwargs)
    plt.xticks(rotation=kwargs.get("rot", 45))
    if window == "default":
        window = len(df)
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
    plt.title(
        f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} {cum_subtitle_end}",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel(f"{aggr} of {metric_display[metric]}".upper(), fontdict={"size": 12})
    return ax


def cat_Vs_counts(
    df,
    metric,
    show_sender=False,
    cat="chat",
    n=5,
    stacked=False,
    label=True,
    legend=True,
    asc=False,
    aggr="count",
    return_data=False,
    style=True,
    *args,
    **kwargs,
):

    if df.sender_name.nunique() > 2:
        meta_data = get_meta_data()
        df["sender_name"] = df["sender_name"].apply(
            lambda name: name if name in [meta_data["user"]] else "Others"
        )

    if show_sender:
        df = (
            df.groupby([cat, "sender_name"])
            .agg({metric: aggr})
            .reset_index()
            .pivot(index=cat, columns="sender_name", values=metric)
        )
        df = df.fillna(0).sort_index(key=df.agg("sum", axis=1).get, ascending=asc)
        df = df.astype("int")
    else:

        df = df[cat].value_counts(ascending=asc).to_frame()
        stacked = False
        legend = False
    if "emoji" in kwargs.keys() and cat == "reactions" and show_sender:
        try:
            df.columns = [df.columns[1], df.columns[0]]
            df = df[[df.columns[1], df.columns[0]]]
        except:
            stacked = True
    if return_data:
        if style:
            return df.style.pipe(make_pretty)
        return df
    if "emoji" not in kwargs.keys():
        df.index = df.index.map(lambda name: get_display(arabic_reshaper.reshape(name)))
        df.columns = df.columns.map(
            lambda name: get_display(arabic_reshaper.reshape(name))
        )
    # ax=df.plot.bar(figsize=(len(df.columns)>15 and len(df) or 15,10),stacked=stacked,xlabel="",ylabel=f"{aggr} of {metric}".upper(),legend=legend)
    ax = df.plot.bar(
        figsize=kwargs.get(
            "figsize", "emoji" in kwargs.keys() and (len(df) > 15 and len(df) or 15, 10)
        )
        or (len(df.columns) // 10 and len(df) or 10, 10),
        stacked=stacked,
        xlabel="",
        legend=legend,
        width="emoji" in kwargs.keys() and 0.85 or 0.5,
    )
    if "emoji" in kwargs.keys():
        for idx, emo in enumerate(df.index):
            offset_image([idx, 0], emo, ax)
        ax.xaxis.set_ticklabels([])
        sub_title = cat == "reactions" and "react" or "absolute emoji"
        plt.title(f"Top {n} used {sub_title}", fontsize=12, fontweight="bold")
    else:
        plt.title(
            f"The {n} {cat_display[cat]} with the highest {aggr} of {metric_display[metric]}",
            fontsize=12,
            fontweight="bold",
        )
    if label:
        ax = label_counts(ax, df, stacked, show_sender, **kwargs)
    plt.xticks(rotation=kwargs.get("rot", 45))
    plt.ylabel(f"{aggr} of {metric_display[metric]}".upper(), fontdict={"size": 12})
    return ax


def dynamic_sort(df, n=5, label=True, legend=True, width=15, *args, **kwargs):
    df.columns = df.columns.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    if len(df) < 2:
        df = df.T.sort_index(key=df.T.sum(1).get, ascending=False)[:n]
        ax = df.plot.bar(
            figsize=kwargs.get("figsize", (n < 15 and 15 or n, 10)),
            legend=False,
            xlabel="",
            rot=kwargs.get("rot", 45),
            width=0.9,
        )
        if "emoji" in kwargs.keys():
            plt.title(
                f"The use of Emoji across chats", y=1, fontsize=12, fontweight="bold"
            )
        if label:
            tol = df.max().values[0] * 0.02
            for idx, row in df.reset_index().iterrows():
                if row[1]:
                    ax.text(
                        idx,
                        row[1] + tol,
                        row[1],
                        ha="center",
                        rotation=kwargs.get("label_rot", 0),
                    )
            ax.set_ylim(top=tol * 55)
        return ax, ax.get_figure()
    chat_colors = {}
    handles = []
    fig, axes = plt.subplots(
        nrows=1, ncols=len(df), sharey=True, figsize=kwargs.get("figsize", (width, 10))
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    tol = df.max().max() * 0.02

    for idx, row in df.reset_index().iterrows():

        row.name = row[0]
        row = row[1:].sort_values(ascending=False)[:n]

        for name in row.index:
            if name not in chat_colors.keys():
                chat_colors[name] = colors[len(chat_colors) % 10]
                handles.append(mpatches.Patch(color=chat_colors[name], label=name))

        row.plot.bar(
            width=1,
            color=[
                chat_colors[name] for name in row.index
            ],  # colordf.set_index("name").loc[row.index].color.values
            xlabel="",
            ax=axes[idx],
        )
        axes[idx].tick_params(axis="both", length=0)
        ax2 = axes[idx].twiny()
        ax2.set_xticks([0.5])
        ax2.set_xticklabels([row.name])
        if "emoji" in kwargs.keys() and emoji.emoji_count(row.name):
            offset_image([0.5, tol * 55], row.name, ax2)
            ax2.set_xticklabels([])

        ax2.tick_params(which="major", length=0)
        if label:
            for i, val in enumerate(row):
                if val:
                    axes[idx].text(
                        i,
                        val + tol,
                        int(val),
                        rotation=kwargs.get("label_rot", 90),
                        ha="center",
                    )

    if "emoji" in kwargs.keys():
        fig.suptitle(
            f"The use of Top {len(df)} used emojis across chats",
            y=legend and 0.975 or 0.92,
            fontsize=12,
            fontweight="bold",
        )
    if label:
        axes[idx].set_ylim(top=df.max().max() * 1.1)

    if legend:

        fig.legend(
            handles,
            chat_colors.keys(),
            loc=kwargs.get("loc"),
            ncol=kwargs.get("ncol", 1),
            bbox_to_anchor=kwargs.get("bbox_to_anchor"),
        )
    return axes, fig


def dt_Vs_cat(
    df,
    cat,
    metric,
    stacked,
    n=5,
    m=5,
    sort="fixed",
    aggr="count",
    cumulative=False,
    cum_aggr="sum",
    window="default",
    dt_disc=False,
    dt_hrchy="month",
    dt_groups={},
    subplots=False,
    return_data=False,
    style=False,
    label=True,
    legend=True,
    asc=False,
    **kwargs,
):

    if dt_hrchy not in freq_mapper.keys():
        raise ValueError(
            f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
        )
    groupping_params = {
        True: ["timestamp_ms", cat],
        False: [pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]), cat],
    }

    # order=df.groupby(cat).agg({metric:aggr}).sort_values(metric,ascending=asc).index.values
    if dt_disc:

        df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
        if dt_groups:
            df.timestamp_ms = df.timestamp_ms.map(
                lambda dt: dt_groups.get(dt, "Others")
            )

    df = (
        df.groupby(groupping_params[dt_disc])
        .agg({metric: aggr})
        .reset_index()
        .pivot(index="timestamp_ms", columns=cat, values=metric)
    )
    if not dt_disc:
        df = df.reindex(
            pd.date_range(df.index[0], df.index[-1], freq=freq_mapper[dt_hrchy])
        )
        df.index = df.index.date
        df.index.name = "timestamp_ms"
    df = df.fillna(0).astype("int")

    if cumulative:
        if cum_aggr not in valid_cum_aggr:
            raise ValueError(
                f"{cum_aggr} is not a valid aggregation function for cumulation; valid aggr are {valid_cum_aggr}"
            )
        if window == "default":
            window = len(df)
        df = df.rolling(window=window, min_periods=1).agg(cum_aggr)
        df = df.apply(round, args=[2])

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
    # df=df[order]
    if return_data:
        if style:
            return df.style.pipe(make_pretty)
        return df

    if sort == "fixed":
        df.columns = df.columns.map(
            lambda name: get_display(arabic_reshaper.reshape(name))
        )

    if len(df) == 1 or len(df.columns) == 1:
        ax = df.plot.bar(
            figsize=kwargs.get("figsize", (len(df) // 5 and 2 * len(df) or 10, 10)),
            legend=False,
            rot=kwargs.get("rot", 45),
            width=0.9,
            xlabel="",
        )
        if label:
            tol = df.max().values[0] * 0.02
            for idx, row in df.reset_index().iterrows():
                if row[1]:
                    ax.text(
                        idx,
                        row[1] + tol,
                        row[1],
                        ha="center",
                        rotation=kwargs.get("label_rot", 0),
                    )
            ax.set_ylim(top=tol * 55)
        # return df[:n].plot.bar(figsize=(len(df)//5 and 2*len(df) or 10,10),width=0.9,legend=legend)
        plt.title(
            f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
            fontsize=12,
            fontweight="bold",
        )
        plt.ylabel(f"{aggr} of {metric_display[metric]}".upper(), fontdict={"size": 12})
        return ax
    if subplots:
        tol = df.max().max() * 0.02
        order = df.sum(0).sort_values(ascending=False).index
        df = df[order]
        fig, axes = plt.subplots(
            nrows=len(df.columns[:n]), ncols=1, sharex=True, sharey=True
        )
        plt.subplots_adjust(wspace=0, hspace=0.2)
        figsize = kwargs.get("figsize", (len(df) // 2, len(df.columns[:n]) * 2))
        for idx, sub in enumerate(df.columns[:n]):
            subset = df[sub]
            # subset.index=subset.index.date
            subset.plot.bar(
                xlabel="",
                ax=axes[idx],
                figsize=figsize,
                label=sub,
                color=colors[idx % 10],
            )
            axes[idx].legend(loc=(1.02, 0.5))
            if label:
                for i, row in subset.reset_index().iterrows():
                    if row[1]:
                        axes[idx].text(
                            i,
                            row[1] + tol,
                            row[1],
                            ha="center",
                            rotation=kwargs.get("label_rot", 0),
                        )
        if label:
            axes[idx].set_ylim(top=df.max().max() * 1.2)
        plt.xticks(rotation=kwargs.get("rot", 45))
        plt.tight_layout(pad=0, h_pad=0.2)
        fig.suptitle(
            f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        fig.text(
            -0.02,
            0.5,
            f"{aggr} of {metric_display[metric]}".upper(),
            rotation="vertical",
            fontsize=12,
        )
        return axes

    if stacked:
        df = df.iloc[:, :n]
        if not legend:
            df = df.sum(1)
        ax = df.plot.bar(
            figsize=kwargs.get("figsize", (len(df) // 5 and 2 * len(df) or 10, 10)),
            width=0.9,
            stacked=True,
            legend=legend,
            rot=kwargs.get("rot"),
            xlabel="",
        )
        plt.title(
            f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
            fontsize=12,
            fontweight="bold",
        )
        if label:
            # tol=df.iloc[0].sum()*0.02
            # tol=df.sum(1).max()
            tol = ax.get_ylim()[1] * 0.02
            for idx, row in df.reset_index().iterrows():
                val = row[1:].sum()
                if val:
                    ax.text(
                        idx,
                        val + tol,
                        val,
                        ha="center",
                        rotation=kwargs.get("label_rot", 0),
                    )
                ax.set_ylim(top=tol * 55)
        plt.ylabel(f"{aggr} of {metric_display[metric]}".upper(), fontdict={"size": 12})
        return ax
    if sort == "dynamic":
        figsize = kwargs.get("figsize", (2 * len(df), 10))
        kwargs["figsize"] = figsize
        axes, fig = dynamic_sort(
            df,
            n=n,
            label=label,
            legend=legend,
            width=2 * len(df),
            loc="upper center",
            ncol=len(df.columns),
            bbox_to_anchor=(0.5, 0.95),
            **kwargs,
        )
        fig.suptitle(
            f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
            fontsize=12,
            fontweight="bold",
            y=legend and 0.975 or 0.92,
        )
        # fig.text(0.08*(figsize[0]/15), 0.5, f"{aggr} of {metric}".upper(), va='center',ha='left', rotation='vertical')
        return axes

    figsize = kwargs.get("figsize", (len(df) // 5 and 2 * len(df) or 10, 10))
    fig, axes = plt.subplots(nrows=1, ncols=len(df), sharey=True, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    tol = df.max().max() * 0.02
    for idx, row in df.reset_index().iterrows():
        row[1 : n + 1].plot.bar(
            width=1, color=colors[: len(df.columns[:n])], xlabel="", ax=axes[idx]
        )

        ax2 = axes[idx].twiny()
        ax2.set_xticks([0.5])
        ax2.set_xticklabels([row.timestamp_ms])
        ax2.tick_params(which="major", length=0)
        if label:
            for i, val in enumerate(row[1 : n + 1]):
                if val:
                    axes[idx].text(
                        i,
                        val + tol,
                        round(val, 2),
                        rotation=kwargs.get("label_rot", 90),
                        ha="center",
                    )
    if label:
        axes[idx].set_ylim(top=df.max().max() * 1.1)

    if legend:
        handles = []
        for idx, col in enumerate(df.columns[:n]):
            handles.append(mpatches.Patch(color=colors[idx], label=col))
        fig.legend(
            handles,
            df.columns,
            loc="upper center",
            ncol=len(df.columns),
            bbox_to_anchor=(0.5, 0.95),
        )
    # plt.xticks(rotation=kwargs.get('rot',90))

    fig.suptitle(
        f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
        fontsize=12,
        fontweight="bold",
        y=legend and 0.975 or 0.93,
    )
    # fig.text(0.08*(figsize[0]/15), 0.5, f"{aggr} of {metric}".upper(),  rotation='vertical')
    return axes
