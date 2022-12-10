import emoji
import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from .special_plots import BubbleChart
from .plot_helper import get_meta_data, cat_filter
from .SW import get_sw

stops = get_sw()


def message_filter(data, n_words=2, emoji_strip=False, subset=None):
    data = data.copy()
    if subset:
        meta_data = get_meta_data()
        if subset == "me":
            data = data[data.sender_name == meta_data["user"]]
        elif subset == "others":
            data = data[~(data.sender_name == meta_data["user"])]
    if emoji_strip:
        if len(data.shape) < 2:
            data = data.dropna()
            # data=data.map(lambda text: emoji.replace_emoji(text, "")).str.strip().replace('',np.nan).dropna()
        else:

            # data['content']=data['content_striped']
            # data=data.dropna(subset='content')
            data = data.dropna(subset="content_striped")
            data["content"] = data["content_striped"]
            # data.content=data.content.map(lambda text: emoji.replace_emoji(text, "")).str.strip().replace('',np.nan)

    if n_words:
        if len(data.shape) < 2:
            data = data[data.str.split().apply(len) <= n_words]
        else:
            data = data.content[data.content.str.split().apply(len) <= n_words]
    return data


def trim_counter_sw(counter):
    for stop in stops:
        if stop in counter.keys():
            del counter[stop]
    return counter


def word(data, width=800, height=400):
    data.index = data.index.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    frequencies = dict(data)
    return WordCloud(
        font_path="arial", width=width, height=height
    ).generate_from_frequencies(frequencies)


def word_cloud_single(data, title, width=1600, height=800):
    if not len(data):
        raise ValueError("No data to plot")
    wordcloud = word(data, width=width, height=height)
    plt.figure(figsize=(15, 10), facecolor="k")
    plt.title(title, fontsize=30, y=1, color="white")
    fig = plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    return fig


def word_cloud_sender(data_lst, titles, suptitle, width=800, height=280):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    fig.suptitle(suptitle, y=1, fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0, h_pad=0)
    for idx, data in enumerate(data_lst):
        if not len(data):
            continue
        ax = axes[idx]
        wordcloud = word(data, width=width, height=height)
        ax.imshow(wordcloud)
        ax.set_title(
            titles[idx], x=-0.02, y=0.2, rotation=90, fontdict={"fontsize": 20}
        )
        ax.axis("off")
    return axes


def word_cloud_time(data_lst, titles, suptitle, width=800, height=300):

    rows = len(data_lst) % 3 and 1 + len(data_lst) // 3 or int(len(data_lst) / 3)
    fig, axes = plt.subplots(ncols=3, nrows=rows, figsize=(30, rows * 4))
    fig.suptitle(suptitle, y=1, fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    try:
        titles = list(
            map(lambda title: get_display(arabic_reshaper.reshape(title)), titles)
        )
    except:
        pass
    axes = axes.flatten()
    for idx, data in enumerate(data_lst):
        if not len(data):
            continue
        ax = axes[idx]
        wordcloud = word(data, width, height)
        ax.set_title(
            titles[idx],
            x=-0.02,
            y="chat" in suptitle and 0.05 or 0.4,
            rotation=90,
            fontdict={"fontsize": "chat" in suptitle and 15 or 20},
        )
        ax.imshow(wordcloud)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout(
        pad=("chat" in suptitle and rows < 3) and 1 or 2,
        h_pad="chat" in suptitle and 1 or 1 / rows,
    )

    return axes


def bubble_data(data):
    data = data.reset_index()
    data.columns = ["word", "count"]
    data = data.sample(len(data), replace=False).reset_index()
    data.word = data.word.map(lambda name: get_display(arabic_reshaper.reshape(name)))
    return data


def bubble_single(data, title):
    data = bubble_data(data)
    if len(data) < 2:
        raise ValueError(
            f"Bubble Chart need at least 2 data points, got only {len(data)}"
        )
    bubble_chart = BubbleChart(area=data["count"], bubble_spacing=2)
    bubble_chart.collapse()
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    ax.set_title(title, fontsize=30)
    fig.set_size_inches(20, 20, forward=False)
    bubble_chart.plot(
        ax,
        data["word"],
        [plt.colormaps["tab20"](i) for i in np.random.choice(range(20), len(data))],
        fontsize=14,
        show_counts=True,
    )
    ax.axis("off")
    ax.relim()
    ax.autoscale_view()

    return ax


def bubble_sender(data_lst, titles, suptitle):

    print(data_lst)
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 12.5))
    fig.suptitle(suptitle, fontsize=30, y=1.04)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    for idx, data in enumerate(data_lst):
        if len(data) < 2:
            continue
        data = bubble_data(data)
        bubble_chart = BubbleChart(area=data["count"], bubble_spacing=1)
        bubble_chart.collapse()
        ax = bubble_chart.plot(
            axes[idx],
            data["word"],
            [plt.colormaps["tab20"](i) for i in np.random.choice(range(20), len(data))],
            fontsize=14,
            show_counts=True,
        )
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()
        ax.set_title(titles[idx].upper(), fontdict={"fontsize": 20})

    return axes


def bubble_time(data_lst, titles, suptitle):

    rows = len(data_lst) % 3 and 1 + len(data_lst) // 3 or int(len(data_lst) / 3)
    fig, axes = plt.subplots(ncols=3, nrows=rows, figsize=(30, rows * 10))
    axes = axes.flatten()
    fig.suptitle(suptitle, y=1 + 0.1 / rows, fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    try:
        titles = list(
            map(lambda title: get_display(arabic_reshaper.reshape(title)), titles)
        )
    except:
        pass
    for idx, data in enumerate(data_lst):
        if len(data) < 2:
            continue
        data = bubble_data(data)

        bubble_chart = BubbleChart(area=data["count"], bubble_spacing=0.1)
        bubble_chart.collapse()
        ax = bubble_chart.plot(
            axes[idx],
            data["word"],
            [plt.colormaps["tab20"](i) for i in np.random.choice(range(20), len(data))],
            fontsize=16,
            show_counts=True,
        )
        ax.set_title(titles[idx], fontdict={"fontsize": 20})
        # ax.relim()
        ax.autoscale_view()
    for ax in axes:
        ax.axis("off")

    return axes


charts = {
    "single": {"bubble": bubble_single, "word_cloud": word_cloud_single},
    "sender": {"bubble": bubble_sender, "word_cloud": word_cloud_sender},
    "time": {"bubble": bubble_time, "word_cloud": word_cloud_time},
}
