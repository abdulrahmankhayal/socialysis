from .media import files_plot, media_plot, chat_media_plot, chat_files_plot
from .share import share_plot, chat_share_plot
from .emoji import chat_emoji, all_emoji, emojis, Multi_emojis
from .words import Words, Multi_Words

from .utils import image_formatter, get_thumbnail
from .plot_helper import get_meta_data, metric_correct, cat_filter
from .common_plots import plot, chat_plot
import pandas as pd
from .get_df import df_from_jsons, set_dur_unit
import tldextract
import os
import shutil
import platform

from tqdm import tqdm
import json
import emoji
import re
import numpy as np
from pandas.util._decorators import doc

cwd = os.path.dirname(__file__)
metric_to_cat = {"share": "domain", "files": "ext", "media_type": "media"}


def set_meta_data(meta_data, cwd, filename="meta_data.json"):
    json_object = json.dumps(meta_data, indent=4)
    with open(os.path.join(cwd, filename), "w") as outfile:
        outfile.write(json_object)


@doc()
def describe(self, n=5, others=False):
    """
    Similar to to pandas.DataFrame.describe

    Generate descriptive statistics of the data for every chat included 
    and for every sender in that chat

    Parameters
    ----------
    n : int , default 5
        number of chats to included,
        it will filter the data by default for only the Top n chats
    others: bool, default False
        whether to keep the chats other than the Top n or not
        When True, the other chats are grouped together and labeled
        as Others

    Returns
    -------
    Pandas.DataFrame

    """
    meta_data = get_meta_data()
    df = self.data.copy()
    do_sum = self.plot._metric in ["call_duration", "audio_files", "video"]
    metric = metric_correct(self.plot._metric, do_sum and "sum" or "count")
    if metric in metric_to_cat.keys():
        metric = metric_to_cat[metric]
    if df.chat.nunique() > 1:
        df, _ = cat_filter(df, n, metric=metric, others=others)

    if others:

        df.loc[
            (df.chat == "Others") & (df.sender_name != meta_data["user"]), "sender_name"
        ] = "Others"
    order = {key: value for (value, key) in enumerate(df.chat.value_counts().index)}
    data = df.groupby(["chat", "sender_name"])[metric].describe()
    if do_sum:
        data["sum"] = df.groupby(["chat", "sender_name"])[metric].sum()
    for chat in data.index.levels[0]:
        chat_data = df[df.chat == chat]
        overall_row = tuple(chat_data[metric].describe())
        if do_sum:
            overall_row = overall_row + (chat_data[metric].sum(),)
        data.loc[(chat, "#Overall"), :] = overall_row
    data = data.sort_index(level=0, key=lambda x: x.map(order))

    if df.chat.nunique() > 1:

        overall_row = tuple(df[metric].describe())
        if do_sum:
            overall_row = overall_row + (df[metric].sum(),)
        data.loc[("#Overall", "#Overall"), :] = overall_row

        df["sender_name"] = df["sender_name"].map(
            lambda name: name if name in [meta_data["user"]] else "Others"
        )
        for sender, row in df.groupby("sender_name")[metric].describe().iterrows():
            overall_row = tuple(row)
            if do_sum:
                overall_row = overall_row + (df[metric].sum(),)
            data.loc[("#Overall", sender), :] = overall_row

    if not do_sum:
        if metric not in ["media", "sticker", "ext", "domain"]:
            data.drop(columns=["unique", "top", "freq"], inplace=True)
        elif metric == "sticker":
            base = get_meta_data()["data_dir"]
            try:
                get_ipython().__class__.__name__
                from IPython.display import HTML
            except:
                return data
            try:
                data["top"] = data["top"].map(lambda uri: os.path.join(base, uri))
                data["top"] = data.top.map(get_thumbnail)
            except:
                return data

            return HTML(data.to_html(formatters={"top": image_formatter}, escape=False))
    else:
        data = data.apply(round, args=[2])
    return data


def detect_languages(data):
    if not data.content.count():
        raise ValueError("No text data exist")
    if platform.system() == "Linux":
        try:
            from polyglot.detect import Detector
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """polyglot is required for language detection,install it using `pip install polyglot`.
                please note that polyglot requires  PyICU and pycld2 in order to operate correctly, consider installing them
                using `pip install PyICU`, `pip install pycld2`"""
            )

        from operator import attrgetter

        get_name = attrgetter("name")
        get_per = attrgetter("confidence")
        languages = [
            {get_name(lan): f"{get_per(lan)}%"}
            for lan in Detector("".join(data.content_striped.dropna())).languages
            if get_per(lan)
        ]
    else:

        try:
            import cld2
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """cld2 is required for language detection,install it using `pip install cld2-cffi`."""
            )

        languages = [
            {lan[0]: f"{lan[2]}%"}
            for lan in cld2.detect("".join(data.content_striped.dropna()))[2]
            if lan[2]
        ]
    return languages


plot_type = {"single": chat_plot, "multi": plot}
files_pt = {"single": chat_files_plot, "multi": files_plot}
media_pt = {"single": chat_media_plot, "multi": media_plot}


@doc(klass="Photos", other_attrs="", other_methods="")
class Photos:
    """
    The Class that enables you to deal with {klass} data

    Attributes
    ----------
    data : pandas.DataFrame
        The {klass} row data
    count : int
        The total number of {klass} you have sent or recieved
    plot : obj
        The obj that you use to plot your data
    {other_attrs}

    Methods
    -------
    describe(n=5,others=False)
        Generate descriptive statistics
    {other_methods}
    """

    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["photos"] = self.data["photos"].apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.count = self.data.photos.count()
        self.plot = plot_type[p_type]("photos", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):

        return describe(self, n, others)


@doc(Photos, klass="Gifs", other_attrs="", other_methods="")
class Gifs:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["gifs"] = self.data["gifs"].apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.count = self.data.gifs.count()
        self.plot = plot_type[p_type]("gifs", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


@doc(
    Photos,
    klass="Files",
    other_attrs="exts : pandas.Series \n    the file extensions you used and how often",
    other_methods="",
)
class Files:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["files"] = self.data["files"].apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.data["ext"] = self.data["files"].apply(
            lambda file: os.path.splitext(file)[1]
        )
        self.count = self.data.files.count()
        self.exts = self.data.ext.value_counts()
        self.plot = files_pt[p_type]("files", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


@doc(Photos, klass="Stickers", other_attrs="", other_methods="")
class Stickers:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["sticker"] = self.data["sticker"].apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.count = self.data.sticker.count()
        self.plot = plot_type[p_type]("sticker", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


@doc(
    Photos,
    klass="Audio Files",
    other_attrs="""total_audio_files_dur : int
    The total length of all of your audio files
longest_audio_file : pandas.Series
    Information about the audio file with the longest duration
curr_time_unit : str
    the current time unit of your audio files length""",
    other_methods=""""change_dur_unit(unit)
    Change the time unit of audio files length""",
)
class Audios:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["audio_files_uri"] = self.data.audio_files.apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.data["audio_files_length"] = self.data.audio_files.apply(
            lambda item: item["length"] if item == item else np.nan
        )
        self.data.drop(columns=["audio_files"], inplace=True)
        self.count = self.data.audio_files_uri.count()
        self.plot = plot_type[p_type]("audio_files", self.data)
        self.__duration_unit = get_meta_data()["time_unit"]

    @property
    def total_audio_files_dur(self):
        return self.data.audio_files_length.sum()

    @property
    def longest_audio_file(self):
        return self.data.loc[self.data.audio_files_length.idxmax()]

    @property
    def curr_time_unit(self):
        return self.__duration_unit

    @doc(metric="audio")
    def change_dur_unit(self, unit):
        """
        Used to change the time unit of your {metric} data duration

        Parameters
        ----------
        unit : ('sec','minute','hour','day')
            the time unit to convert the data into
        """
        if unit == self.__duration_unit:
            return
        self.data.audio_files_length = set_dur_unit(
            self.data.audio_files_length, self.__duration_unit, unit
        )
        self.__duration_unit = unit

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


@doc(
    Photos,
    klass="Videos",
    other_attrs="""total_video_dur : int
    The total length of all of your videos
longest_video : pandas.Series
    Information about the video with the longest duration
curr_time_unit : str
    the current time unit of your videos length """,
    other_methods="change_dur_unit(unit)\n    Change the time unit of videos length",
)
class Videos:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.data["video_uri"] = self.data.videos.apply(
            lambda item: item["uri"] if item == item else np.nan
        )
        self.data["video_length"] = self.data.videos.apply(
            lambda item: item["duration"] if item == item else np.nan
        )
        # self.data["video_frame_count"]=self.data.videos.apply(lambda item: item["frame_count"] if item==item else np.nan)
        self.data.drop(columns=["videos"], inplace=True)
        self.count = self.data.video_uri.count()
        self.plot = plot_type[p_type]("video", self.data)
        self.__duration_unit = get_meta_data()["time_unit"]

    @property
    def curr_time_unit(self):
        return self.__duration_unit

    @property
    def total_video_dur(self):
        return self.data.video_length.sum()

    @property
    def longest_video(self):
        return self.data.loc[self.data.video_length.idxmax()]

    @doc(Audios.change_dur_unit, metric="video")
    def change_dur_unit(self, unit):
        if unit == self.__duration_unit:
            return
        self.data.video_length = set_dur_unit(
            self.data.video_length, self.__duration_unit, unit
        )
        self.__duration_unit = unit

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


@doc(
    Photos,
    klass="The Whole Media",
    other_attrs="used_media : pandas.Series \n    the types of media you used and how often",
    other_methods="",
)
class Media:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        media_files = {}
        for media in ["audio_files", "videos", "photos", "gifs", "files"]:
            media_files[media] = (
                self.data[["timestamp_ms", "chat", "sender_name", media]]
                .dropna()
                .explode(column=media)
            )

        media_files["sticker"] = self.data[
            ["timestamp_ms", "chat", "sender_name", "sticker"]
        ].dropna()

        if len(media_files["audio_files"]):
            self.Audios = Audios(media_files["audio_files"].copy(), p_type)
        if len(media_files["videos"]):
            self.Videos = Videos(media_files["videos"].copy(), p_type)
        if len(media_files["photos"]):
            self.Photos = Photos(media_files["photos"].copy(), p_type)
        if len(media_files["gifs"]):
            self.Gifs = Gifs(media_files["gifs"].copy(), p_type)
        if len(media_files["files"]):
            self.Files = Files(media_files["files"].copy(), p_type)
        if len(media_files["sticker"]):
            self.Stickers = Stickers(media_files["sticker"].copy(), p_type)

        for media in media_files.keys():
            media_files[media][media] = media

        self.data = (
            pd.concat(media_files.values())
            .melt(
                id_vars=["timestamp_ms", "chat", "sender_name"],
                value_vars=media_files.keys(),
                var_name="media_type",
            )
            .dropna()
        )
        self.data["value"] = self.data.media_type.replace(
            {
                "audio_files": "Audio",
                "videos": "Video",
                "photos": "Photo",
                "gifs": "GIF",
                "files": "File",
                "sticker": "Sticker",
            }
        )
        self.count = self.data.media_type.count()
        self.used_media = self.data.media_type.value_counts()
        self.data.rename(columns={"value": "media"}, inplace=True)
        if p_type == "single":
            self.data.drop(columns=["chat"])
        self.plot = media_pt[p_type]("media_type", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


emoji_pt = {"single": emojis, "multi": Multi_emojis}
words_pt = {"single": Words, "multi": Multi_Words}
content_pt = {"single": chat_plot, "multi": plot}


@doc(
    Photos,
    klass="Messages content",
    other_attrs="""Emoji : obj
    The obj that enables you to deal with the emojis in your messages
Words : obj
    The obj that enables you to deal with the text of your messages""",
    other_methods="",
)
class Content:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.count = self.data.content.count()
        if self.data.emoji.count():
            self.Emoji = emoji_pt[p_type](passed_data)
        self.Words = words_pt[p_type](
            passed_data[
                ["timestamp_ms", "chat", "sender_name", "content", "content_striped"]
            ]
        )
        self.plot = content_pt[p_type](
            "content", passed_data[["timestamp_ms", "chat", "sender_name", "content"]]
        )

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


calls_pt = {"single": chat_plot, "multi": plot}


@doc(
    Photos,
    klass="Calls",
    other_attrs="""first_call_date : timestamp 
    The date in which the first call were made
last_call_date : timestamp
    The date in which the first call were made
total_calls_dur : float
    The total duration of all calls
total_video_dur : float
    The total duration of all video calls
total_voice_dur : float
    The total duration of all voice calls
longest_call : pandas.Series
    Information about the call with the longest duration
curr_time_unit : str
    the current time unit of your call duration """,
    other_methods="""change_dur_unit(unit)
    Change the time unit of call duration""",
)
class Calls:
    def __init__(self, passed_data, p_type):
        self.data = passed_data
        self.count = self.data.call_duration.count()
        self.first_call_date = self.data.timestamp_ms.min()
        self.last_call_date = self.data.timestamp_ms.max()

        if p_type == "multi":
            self.most_called = (
                self.data.groupby("chat")["call_duration"].count().idxmax()
            )
        # or self.data.chat.value_counts()[0].index
        self.plot = calls_pt[p_type](
            "call_duration",
            passed_data[
                [
                    "timestamp_ms",
                    "chat",
                    "sender_name",
                    "call_duration",
                    "call_status",
                    "call_type",
                ]
            ],
        )
        self.__duration_unit = get_meta_data()["time_unit"]

    @property
    def curr_time_unit(self):
        return self.__duration_unit

    @property
    def total_calls_dur(self):
        return self.data.call_duration.sum()

    @property
    def total_video_dur(self):
        return self.data[self.data.call_type == "video"].call_duration.sum()

    @property
    def total_voice_dur(self):
        return self.data[self.data.call_type == "voice"].call_duration.sum()

    @property
    def longest_call(self):
        return self.data.loc[self.data.call_duration.idxmax()]

    @doc(Audios.change_dur_unit, metric="call")
    def change_dur_unit(self, unit):
        if unit == self.__duration_unit:
            return
        self.data.call_duration = set_dur_unit(
            self.data.call_duration, self.__duration_unit, unit
        )
        self.__duration_unit = unit

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


links_pt = {"single": chat_share_plot, "multi": share_plot}


@doc(
    Photos,
    klass="Share Links",
    other_attrs="all_domains : pandas.Series \n    the domains of links you shared and their counts",
    other_methods="",
)
class links:
    def __init__(self, passed_data, p_type):
        self.data = passed_data.copy()
        self.data["share"] = self.data["share"].apply(
            lambda item: item["link"] if item == item else np.nan
        )
        self.data["domain"] = (
            self.data["share"].apply(tldextract.extract).apply(lambda row: row.domain)
        )
        self.count = self.data.share.count()
        self.all_domains = self.data.domain.value_counts()
        self.plot = links_pt[p_type]("share", self.data)

    @doc(describe)
    def describe(self, n=5, others=False):
        return describe(self, n, others)


class Chat:
    """
    The class for single chat
    Use [chat_name] to access a chat instance. 

    Attributes
    ----------
    data : pandas.DataFrame
        A subset of all data for a single chat
    name : str
        Chat name as it appears in messenger
    n_of_messages : int
        The count of all messages in this chat
    languages : dict
        The languages used in this chat and their percentages
    chat_starter : str
        The name of the person who started the conversation
    first_msg_date : timestamp
        The date of the first message
    n_of_distinct_days : int
        The total number of different days you chatted on
    longest_streak : int
        Longest number of days of continuous chatting/chatting streak
    streak_periods : dict
        The period in which you achieved the longest streak
    last_hear : timestamp
        The time of the last message between you
    longest_msgs : pandas.DataFrame
        Information about the longest 10 messages in terms of word count
    Media : Media obj
        to deal with chat Media, only exist if the chat have at least one media file
    Calls : Calls obj
        to deal with chat Calls, only exist if you did at least one call 
    Content : Content obj
        to deal with chat messages content, only exist if you have at least one message between you
    Reacts : Reacts obj
        to deal with chat Reacts, only exist if at least one of you reacted with a message
    Links : Links obj
        to deal with Links you shared in the chat, only exist if you have sent at least one link between you
    plot : Plot obj
        used to plot your data

    """

    def __init__(self, passed_name, passed_data, SubModules):
        if hasattr(self, "data"):
            self.__reload(SubModules)
        else:
            self.data = passed_data
            self.name = passed_name
            self.n_of_messages = len(self.data)
            if self.data.content.count():
                self.longest_msgs = (
                    self.data.join(
                        self.data.content.dropna().apply(len), rsuffix="_len"
                    )
                    .sort_values("content_len", ascending=False)[
                        ["timestamp_ms", "sender_name", "content", "content_len"]
                    ][:10]
                    .reset_index(drop=True)
                )
            # self.reactions=self.data['reactions'].dropna().str.split().explode().value_counts()
            self.chat_starter = (
                self.data.sort_values("timestamp_ms").iloc[0].sender_name
            )
            self.first_msg_date = str(self.data.timestamp_ms.min())
            self.n_of_distinct_days = self.data.timestamp_ms.dt.date.nunique()
            unique_days = pd.Series(
                sorted(self.data.timestamp_ms.dt.date.unique()), name="dates"
            )
            if len(unique_days) <= 1:
                self.longest_streak = 1
                self.streak_periods = {
                    "Start": unique_days.min().strftime("%Y-%m-%d"),
                    "End": unique_days.min().strftime("%Y-%m-%d"),
                }
            else:
                consec_dates = unique_days.to_frame("dates").join(
                    (
                        unique_days.diff()
                        .dt.days.ne(1)
                        .cumsum()
                        .to_frame()
                        .groupby("dates")
                        .cumcount()
                        + 1
                    ).rename("n_consec")
                )
                self.longest_streak = consec_dates["n_consec"].max()
                periods = []
                for date in consec_dates[
                    consec_dates.n_consec == self.longest_streak
                ].dates:
                    periods.append(
                        {
                            "Start": (
                                date - pd.Timedelta(days=self.longest_streak - 1)
                            ).strftime("%Y-%m-%d"),
                            "End": date.strftime("%Y-%m-%d"),
                        }
                    )
                self.streak_periods = periods
            # self.longest_streak=len(unique_days)>1 and (pd.Series(sorted(unique_days)).diff().dt.days.ne(1).cumsum().to_frame().groupby(0).cumcount()+1).max() or 1
            self.last_hear = str(self.data.timestamp_ms.max())
            self.plot = chat_plot("chat", self.data)
            self.__reload(SubModules)

    @property
    def languages(self):
        return detect_languages(self.data)

    def __reload(self, SubModules):

        if (
            self.data[["photos", "sticker", "files", "videos", "audio_files", "gifs"]]
            .count()
            .max()
            > 0
            and (SubModules == "All" or "Media" in SubModules)
            and not hasattr(self, "Media")
        ):
            self.Media = Media(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "photos",
                        "sticker",
                        "files",
                        "videos",
                        "audio_files",
                        "gifs",
                    ]
                ],
                "single",
            )
        # self.total_voice_dur=hasattr(self,'Media') and hasattr(self.Media,'Audios') and self.Media.Audios.data.audio_files_length.sum()//60 or 0
        if (
            self.data.call_duration.count()
            and (SubModules == "All" or "Calls" in SubModules)
            and not hasattr(self, "Calls")
        ):
            self.Calls = Calls(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "call_duration",
                        "call_status",
                        "call_type",
                    ]
                ].dropna(),
                "single",
            )
        if (
            self.data.content.count()
            and (SubModules == "All" or "Content" in SubModules)
            and not hasattr(self, "Content")
        ):
            self.Content = Content(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "content",
                        "content_striped",
                        "emoji",
                    ]
                ].dropna(subset="content"),
                "single",
            )
        if (
            self.data.reactions.count()
            and (SubModules == "All" or "Reacts" in SubModules)
            and not hasattr(self, "Reacts")
        ):
            self.Reacts = chat_emoji(
                "reactions",
                self.data[
                    ["timestamp_ms", "chat", "sender_name", "reactions"]
                ].dropna(),
            )
            # self.Reacts=Reacts(,'single')
        if (
            self.data.share.count()
            and (SubModules == "All" or "Links" in SubModules)
            and not hasattr(self, "Links")
        ):
            self.Links = links(
                self.data[["timestamp_ms", "chat", "sender_name", "share"]].dropna(),
                "single",
            )


class Stats:
    """
    The class for single chat

    Attributes
    ----------
    data : pandas.DataFrame
        the full data
    n_of_messages : int
        The count of all messages 
    top_chats : list
        The top 5 chats in terms of number of messages
    languages : dict
        The all languages used in your chat and their percentages -- limted to only 3 languages
    first_msg_date : timestamp
        The date of the first message
    first_chat : str 
        The first ever person you chatted with
    top_streaks : pandas.Series
        Top 10 Longest chatting streak, one for each chat
    longest_msgs : pandas.DataFrame
        Information about the longest 10 messages in terms of word count
    Media : Media obj
        to deal with Media, only exist if the data contains at least one media file
    Calls : Calls obj
        to deal with Calls, only exist if you did/recieved at least one call 
    Content : Content obj
        to deal with messages content, only exist if your data have at least one message
    Reacts : Reacts obj
        to deal with Reacts, only exist if at least one react is made
    Links : Links obj
        to deal with shared Links, only exist if at least one link is shared
    plot : Plot obj
        used to plot your data

    Methods
    -------
    __init__(base='./',restore=False,**kwargs)
        Creates the data
    load(SubModules=False,chat_instances=True)
        Build the main classes and create chat instances
    freeze()
        to save your data to be restored in a later time
    update(base='./',after=True,**kwargs)
        to extend your data


    """

    def __init__(self, base="./", restore=False, **kwargs):
        """
        Generates a DataFrame from your row data or restore the df you saved

        Parameters
        ----------
        base : str, default cwd
            The the directory where the data is located.
            Only passed when generating the data for the first time,
            has no effect when `restore` is True.
        restore : bool, default False
            if you saved the data using `freeze()` method, you can restore it without the need to regenerating it.
        **kwargs : parameters to control data generation
            parallel : bool , default False
                When True , data is processed in parallel whenever it possible.
                It helps speed up the process of generating the data, but it consumes your computer resources.
            max_workers : int, default 2
                The number of threads to use when processing the data in parallel.
            process_audio : bool, default True
                whether to process the audio files to get their durtion or not.
                Processing audio files is a computationally intensive process, 
                especially if your data contains a lot of audio files,
                you can speed up this process by passing {parallel = True} to process audio data in parallel.
            dur_unit : {'sec','minute','hour','day'}, default 'sec'
                The time unit of the length of calls, videos, and audio files
        """
        self.__Chats = {}
        # self.Chats=locals()
        if restore:
            if restore == "sample":
                meta_data = get_meta_data("sample_meta_data.json")
                meta_data["data_dir"] = cwd
                set_meta_data(meta_data, cwd, "sample_meta_data.json")
            data_name = restore == "sample" and "SAMPLE_DATA.pkl" or "MY_DATA.pkl"
            if os.path.isfile(os.path.join(cwd, data_name)):
                self.data = pd.read_pickle(os.path.join(cwd, data_name))
                shutil.copy(
                    os.path.join(
                        cwd,
                        restore == "sample"
                        and "sample_meta_data.json"
                        or "gen_meta_data.json",
                    ),
                    os.path.join(cwd, "meta_data.json"),
                )
                meta_data = get_meta_data()
                self.user = meta_data["user"]
                # os.path.splitext(file)[1]
            else:
                raise FileNotFoundError(f"No data to restore")

        else:
            if not os.path.exists(base):
                raise FileNotFoundError("directory not found")
            dur_unit = kwargs.get("dur_unit", "sec")
            self.data = df_from_jsons(base, **kwargs)

            info = json.load(
                open(os.path.join(base, "messages\\autofill_information.json"), "r")
            )
            self.user = info[list(info)[-1]]["FULL_NAME"][0]
            meta_data = {
                "user": self.user,
                "Top": [],
                "data_dir": base,
                "time_unit": dur_unit,
            }
            set_meta_data(meta_data, cwd, "gen_meta_data.json")
            set_meta_data(meta_data, cwd, "meta_data.json")
        self.__loaded = False

    @property
    def first_msg_date(self):
        return str(self.data.timestamp_ms.min())

    @property
    def last_msg_date(self):
        return str(self.data.timestamp_ms.max())

    def __getitem__(self, name):
        if not len(self.__Chats):
            raise Exception("Please load chat_instances first")
        if name not in self.__Chats.keys():
            raise KeyError(f"chat : {name}, not found in your data")
        return self.__Chats[name]
        # return self.Chats[arg]

    # def __getattr__(self,attr):
    # return self.Chats[attr]

    def load(self, SubModules="All", chat_instances=True):
        """
        Build the main classes and create the chat instances.
        Without load, you wont be able to use or access any of Stats attributes/Modules except `df` attribue.

        Parameters
        ----------
        SubModules : {'Media','Calls','Content','Reacts','Links'} or a list of more than one them, default 'All'
            When 'All', All Modules are loaded.
            When [] : No Module will be loaded
            if specified, only the Modules you specified are loaded.
            Can be used to reduce the load time if you don't need to use all Modules.
        chat_instances : bool, default True
            Whether to create chat instances or not
            Can be used to reduce the load time if you are not willing to inspect the data of individual chats.
        """
        if not len(self.data):
            raise Exception("data not exist")
        if SubModules and SubModules != "All":
            if not isinstance(SubModules, list):
                SubModules = [SubModules]
            for SubModule in SubModules:
                if SubModule not in ["Media", "Calls", "Content", "Reacts", "Links"]:
                    raise ValueError(f"No Module called {SubModule},")
        if SubModules:
            print("Building Main Modules ...")
        if self.__loaded:
            self.__reload(SubModules, chat_instances)
        else:
            self.n_of_messages = len(self.data)
            self.top_chats = self.data.chat.value_counts()[:5].index.tolist()
            meta_data = get_meta_data()
            meta_data["Top"] = self.top_chats
            set_meta_data(meta_data, cwd)

            if self.data.content.count():

                self.data.content = (
                    self.data[
                        ~(
                            (
                                self.data.content.dropna().str.contains(
                                    "Reacted .* to your message|تم التفاعل باستخدام .* مع رسالتك"
                                )
                            )
                            | ~self.data.call_duration.isna()
                            | self.data.content.dropna().str.contains(
                                "لقد أرسلتَ مرفقًا.|You sent an attachment."
                            )
                            | self.data.content.dropna().str.contains(
                                "تم إرسال مرفق بواسطة|sent an attachment."
                            )
                        )
                    ]
                    .content.reindex(range(len(self.data)))
                    .map(lambda txt: re.sub(r"http\S+", "", txt), na_action="ignore")
                    .replace("", np.NaN)
                )

                self.data["content_striped"] = (
                    self.data.content.map(
                        lambda text: emoji.replace_emoji(text, ""), na_action="ignore"
                    )
                    .str.strip()
                    .replace("", np.nan)
                )

                self.longest_msgs = (
                    self.data.join(
                        self.data.content.dropna().apply(len), rsuffix="_len"
                    )
                    .sort_values("content_len", ascending=False)[
                        [
                            "timestamp_ms",
                            "chat",
                            "sender_name",
                            "content",
                            "content_len",
                        ]
                    ][:10]
                    .reset_index(drop=True)
                )

            self.first_chat = self.data.sort_values("timestamp_ms").iloc[0].chat
            self.top_streaks = (
                self.data.groupby(["chat", pd.Grouper(key="timestamp_ms", freq="D")])
                .size()
                .reset_index()
                .groupby("chat")["timestamp_ms"]
                .unique()
                .map(
                    lambda dates: (
                        pd.Series(sorted(dates), name="dates")
                        .diff()
                        .dt.days.ne(1)
                        .cumsum()
                        .to_frame()
                        .groupby("dates")
                        .cumcount()
                        + 1
                    ).max()
                )
                .sort_values(ascending=False)[:10]
            )
            # self.reactions=self.data['reactions'].dropna().str.split().explode().value_counts()

            self.__reload(SubModules, chat_instances)
            self.plot = plot("index", self.data.reset_index())
            self.__loaded = True

    @property
    def languages(self):
        return detect_languages(self.data)

    def freeze(self):
        """
        Saves your data to be restored in a later time
        """
        self.data.to_pickle(os.path.join(cwd, "MY_DATA.pkl"))

    def __reload(self, SubModules=False, chat_instances=True):

        if (
            self.data[["photos", "sticker", "files", "videos", "audio_files", "gifs"]]
            .count()
            .max()
            > 0
            and (SubModules == "All" or "Media" in SubModules)
            and not hasattr(self, "Media")
        ):
            print("Creating Media ...")
            self.Media = Media(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "photos",
                        "sticker",
                        "files",
                        "videos",
                        "audio_files",
                        "gifs",
                    ]
                ],
                "multi",
            )
        if (
            self.data.call_duration.count()
            and (SubModules == "All" or "Calls" in SubModules)
            and not hasattr(self, "Calls")
        ):
            print("Creating Calls ...")
            self.Calls = Calls(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "call_duration",
                        "call_status",
                        "call_type",
                    ]
                ].dropna(),
                "multi",
            )
        if (
            self.data.content.count()
            and (SubModules == "All" or "Content" in SubModules)
            and not hasattr(self, "Content")
        ):
            print("Creating Content ...")
            self.Content = Content(
                self.data[
                    [
                        "timestamp_ms",
                        "chat",
                        "sender_name",
                        "content",
                        "content_striped",
                        "emoji",
                    ]
                ].dropna(subset="content"),
                "multi",
            )
        if (
            self.data.reactions.count()
            and (SubModules == "All" or "Reacts" in SubModules)
            and not hasattr(self, "Reacts")
        ):
            print("Creating Reacts ...")
            self.Reacts = all_emoji(
                "reactions",
                self.data[
                    ["timestamp_ms", "chat", "sender_name", "reactions"]
                ].dropna(),
            )
        if (
            self.data.share.count()
            and (SubModules == "All" or "Links" in SubModules)
            and not hasattr(self, "Links")
        ):
            print("Creating Links ...")
            self.Links = links(
                self.data[["timestamp_ms", "chat", "sender_name", "share"]].dropna(),
                "multi",
            )

        if chat_instances:
            print("Create Chat instances ...")
            subset = False
            if not isinstance(chat_instances, bool):
                if isinstance(chat_instances, str):
                    chat_instances = [chat_instances]
                not_existed_chats = set(chat_instances).difference(
                    set(self.data["chat"].unique())
                )
                if not_existed_chats:
                    raise ValueError(f"{not_existed_chats.pop()} not exist")
                subset = True
            if not len(self.__Chats):
                for name in tqdm(
                    subset and chat_instances or self.data["chat"].unique()
                ):
                    self.__Chats[name] = Chat(
                        name, self.data[self.data["chat"] == name], SubModules
                    )
            else:
                if chat_instances == True:
                    chat_instances = self.data["chat"].unique()
                new_chats = set(chat_instances).difference(set(self.__Chats.keys()))
                if new_chats:
                    print("Adding new chats")
                    for name in tqdm(new_chats):
                        self.__Chats[name] = Chat(
                            name, self.data[self.data["chat"] == name], SubModules
                        )
                    print("Update existing chats")
                for name in tqdm(self.__Chats.keys()):
                    self.__Chats[name]._Chat__reload(SubModules)
                # self.Chats[name]=Chat(name,self.data[self.data["chat"]==name],SubModules)

    def update(self, base, after=True, freeze=True, **kwargs):
        """
        Extend your data.
        Useful for continuous data update, as instead of downloading the entire data each time,
        you can just add the new data to the existing one.

        Parameters
        ----------
        base : str, required
            The directory of the data to be added.
        after : bool, default True
            Whether the new data comes chronologically after the existing data.
            This allows you to add data older than the existing one too .
        freeze : bool, default True
            Whether to freeze the newly created combined data or not.
            Please note that it will override yor existing data.
        **kwargs : parameters to control data generation
            parallel : bool , default False
                When True , data is processed in parallel whenever it possible.
                It helps speed up the process of generating the data, but it consumes your computer resources.
            max_workers : int, default 2
                The number of threads to use when process the data in parallel.
            process_audio : bool, default True
                whether to process the audio files to get their durtion or not.


        Notes
        -----
        For consistency, dur_unit is automatically set to the time unit of the existing data.

        The new data is added after/before the date of the first/last record in the existing data,
        no matter the first/last date of the data to be added.
        `first_msg_date`,`last_msg_date` will help you avoid downloading unnecessary data,
        just download the the data starting from the date of `last_msg_date` or ending at the date of `first_msg_date`.

        if you used `load`, reloading process would be done.
        """

        if not os.path.exists(base):
            raise FileNotFoundError("directory not found")
        if kwargs.get("time_unit"):
            raise ValueError(
                "time_unit will be automatically selected to match the existing data time unit"
            )
        try:
            meta_data = get_meta_data("gen_meta_data.json")
        except:
            raise FileNotFoundError(
                "There is no existing data to be updated, sample data cannot be used, use Stats() to create the data for first time"
            )
        df = df_from_jsons(base, dur_unit=meta_data["time_unit"], **kwargs)
        if after:
            cut_date = self.data.timestamp_ms.max().normalize()
            df1 = self.data[self.data.set_index("timestamp_ms").index < cut_date]
            df2 = df[df.set_index("timestamp_ms").index >= cut_date]
        else:
            cut_date = self.data.timestamp_ms.min().normalize()
            df1 = self.data[self.data.set_index("timestamp_ms").index >= cut_date]
            df2 = df[df.set_index("timestamp_ms").index < cut_date]
        self.data = (
            pd.concat([df1, df2], ignore_index=True)
            .sort_values(["chat", "timestamp_ms"], ascending=[True, False])
            .reset_index(drop=True)
        )
        self.data = self.data.drop(["index"], axis=1, errors="ignore").reset_index()
        if freeze:
            self.freeze()

        if self.__loaded:
            attrs = [i for i in self.__dict__.keys() if i[:1] != "_"]
            for name in attrs:
                delattr(self, name)
            self.__init__(restore=True)
            print("Reloading ...")
            self.load()
