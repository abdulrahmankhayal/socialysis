from .words_helper import *
from .plot_helper import get_meta_data, freq_mapper

# import nltk
import re
from collections import Counter
import pandas as pd
import numpy as np
import emoji
from pandas.util._decorators import doc

# tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
stops_set = set(stops)


@doc(occur="in your chat and by whom it sent and how often")
class Words:
    """
    Enables you to perform textual analysis on the text content of your messages.

    Attributes
    ----------
    data : pandas.DataFrame
        Messages row data.
    urTopWords : list
        Your Top 100 used words.
    friendsTopWords : list
        Your friends Top 100 used words.
    top_words : list
        Top 100 used words overall.
    top_messages : pd.Series
        Most 50 repeated messages.

    Methods
    -------
    plot(...)
        Plot top words in many levels using different types of chart. 
    phrase_occurance(phrase):
        Search for specific words occurance {occur}
    """

    def __init__(self, passed_data):
        self.data = passed_data.copy()
        my_words = " ".join(
            message_filter(self.data, n_words=None, subset="me").content.dropna()
        )
        tokens = re.split(r"[\b\W\b]+", my_words)
        # tokens = tokenizer.tokenize(my_words)
        tokens_ns = [word for word in tokens if word not in stops_set]
        my_count = Counter(tokens_ns)
        self.__urWords = my_count
        self.urTopWords = my_count.most_common(100)
        friends_words = " ".join(
            message_filter(self.data, n_words=None, subset="others").content.dropna()
        )
        tokens = re.split(r"[\b\W\b]+", friends_words)
        # tokens = tokenizer.tokenize(friends_words)
        tokens_ns = [word for word in tokens if word not in stops_set]
        friends_count = Counter(tokens_ns)
        self.__friendsWords = friends_count
        self.friendsTopWords = my_count.most_common(100)
        all_counts = my_count + friends_count
        self.__wordsANDcounts = all_counts
        self.top_words = all_counts.most_common(100)

        self.data.content = self.data.content.map(
            lambda txt: txt if txt not in stops_set else np.NaN
        )
        self.data.dropna(subset="content", inplace=True)
        # self.data['content_striped']=self.data.content.map(lambda text: emoji.replace_emoji(text, "")).str.strip().replace('',np.nan)
        self.top_messages = self.data.dropna(
            subset="content_striped"
        ).content.value_counts()[:50]

    @doc(chat="")
    def plot(
        self,
        kind="word_cloud",
        level="word",
        by_sender=False,
        over_time=False,
        dt_hrchy="year",
        n_words="default",
        max_word_length=2,
        subset=None,
        strip_emoji=True,
        return_data=False,
    ):
        """
        Create WordCloud/s and Packed-bubble chart/s.

        Allows you to see the textual content of your data in many text levels,
        and making subsets of it in many different ways.

        Parameters
        ----------
        kinds : {{'word_cloud','bubble'}}, default 'word_cloud'
            Choose between making WordCloud or Packed-bubble chart.
        level : {{'word','phrase','message'}}, default 'word'
            The level at which the textual analysis is done.
            When 'word' , your text data handled word for word.
            When 'phrase' , only messages with no more words than
            the number you specify are considered for analysis.
            When 'message', the text content of messages is treated as a single unit.
        n_words : int
            The number of observations to display for each chart.
            By default, the best number is chosen based on the number of charts generated.
        max_word_length : int, default 2
            When level is 'phrase', the maximum number of words a message could contain,
            messages with higher number of words are ignored.
        {chat}
        by_sender : bool, default False
            Divide your data into 2 subsets, one for your messages ,
            and one for the messages you received, hence 2 charts will be shown.
        over_time : bool, default False
            Divide your data into n subsets, one for each time interval,
            hence n charts will be shown, one for each subset.
            Use `dt_hrchy` to specify the time frequency.
            Time is considered discrete by default.
        dt_hrchy : {{'minute','hour','day','week','month','quarter','year'}}, default 'year'
            When over_time, used to choose a time hierarchy.
        subset : {{'me','others'}}, default None
            Used to filter the data to include only your messages or messages you received.
            Ignored when by_sender.
        strip_emoji : bool, default True
            Emojis in messages don't fit well when plotted, so there are ignored(stripped),
            you can choose to include them.
        return_data : bool, default False
            Returns the plotting data and no charts are returned. 

        Returns
        -------
        matplotlib.axes.Axes/np.ndarray of them or matplotlib.image.AxesImage.
        """
        if kind not in ["word_cloud", "bubble"]:
            raise ValueError(
                f"{kind} is not a valid chart type, Choose between 'word_cloud' and 'bubble' "
            )
        if dt_hrchy not in freq_mapper.keys():
            raise ValueError(
                f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
            )
        if not isinstance(n_words, int) and n_words != "default":
            raise TypeError(
                f"expected 'int' or 'default' got '{type(n_words)}' instead"
            )
        if not isinstance(max_word_length, int):
            raise TypeError(f"expected 'int'  got '{type(n_words)}' instead")
        if subset and subset not in ["me", "others"]:
            raise ValueError(
                f"{subset} is not a valid subset; valid values are 'me', 'others'"
            )
        title_part = (
            level == "word"
            and "words"
            or level == "phrase"
            and f"phrases with {max_word_length} or less words"
            or "message"
        )
        df = self.data.copy()
        if level == "word":
            if not (by_sender or over_time):
                n_words = n_words == "default" and 200 or n_words
                if subset:
                    if subset == "me":
                        data = pd.Series(dict(self.__urWords.most_common(n_words)))
                    else:
                        data = pd.Series(dict(self.__friendsWords.most_common(n_words)))
                else:
                    data = pd.Series(dict(self.__wordsANDcounts.most_common(n_words)))
                title = f"Top {n_words} {title_part}"
                if return_data:
                    return data
                return charts["single"][kind](data, title)

            if by_sender:
                n_words = n_words == "default" and 100 or n_words
                suptitle = f"Top {n_words} {title_part} by sender"
                data_lst = [
                    pd.Series(dict(data.most_common(n_words)))
                    for data in [self.__urWords, self.__friendsWords]
                ]
                if return_data:
                    return data_lst
                return charts["sender"][kind](
                    data_lst,
                    self.data.sender_name.nunique() == 2
                    and [get_meta_data()["user"]]
                    + np.setdiff1d(
                        self.data.sender_name.unique(), get_meta_data()["user"]
                    ).tolist()
                    or ["YOU", "FRIENDS"],
                    suptitle,
                )
            elif over_time:
                n_words = n_words == "default" and 50 or n_words
                suptitle = f"Top {n_words} {title_part} for each {dt_hrchy}"
                if subset:
                    df = message_filter(df, n_words=None, subset=subset)
                df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
                data_s = (
                    df[["timestamp_ms", "content"]]
                    .dropna()
                    .groupby("timestamp_ms")
                    .agg(lambda rows: Counter(re.split(r"[\b\W\b]+", " ".join(rows))))
                    .content.map(trim_counter_sw)
                )
                data_lst = [
                    pd.Series(dict(data.most_common(n_words)))
                    for idx, data in enumerate(data_s)
                ]
                if return_data:
                    return data_lst
                return charts["time"][kind](data_lst, data_s.index, suptitle)

        elif level == "phrase":

            if subset and not by_sender:
                df = message_filter(df, n_words=None, subset=subset)
            if not (by_sender or over_time):
                n_words = n_words == "default" and 200 or n_words
                title = f"Top {n_words} {title_part}"
                data = message_filter(df, max_word_length, strip_emoji).value_counts()[
                    :n_words
                ]
                if return_data:
                    return data
                return charts["single"][kind](data, title)
            elif by_sender:
                n_words = n_words == "default" and 100 or n_words
                suptitle = f"Top {n_words} {title_part} by sender"
                data_lst = [
                    message_filter(
                        df, max_word_length, strip_emoji, sub
                    ).value_counts()[:n_words]
                    for sub in ["me", "others"]
                ]
                if return_data:
                    return data_lst
                return charts["sender"][kind](
                    data_lst,
                    self.data.sender_name.nunique() == 2
                    and [get_meta_data()["user"]]
                    + np.setdiff1d(
                        self.data.sender_name.unique(), get_meta_data()["user"]
                    ).tolist()
                    or ["YOU", "FRIENDS"],
                    suptitle,
                )
            elif over_time:
                n_words = n_words == "default" and 50 or n_words
                suptitle = f"Top {n_words} {title_part} for each {dt_hrchy}"
                if strip_emoji:
                    df.content = df.content_striped
                df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
                df.set_index("timestamp_ms", inplace=True)
                dts = sorted(df.index.unique())
                data_lst = [
                    message_filter(
                        df.loc[dt].content, max_word_length, strip_emoji
                    ).value_counts()[:n_words]
                    for dt in dts
                ]
                if return_data:
                    return data_lst
                return charts["time"][kind](data_lst, dts, suptitle)
        elif level == "message":
            if subset and not by_sender:
                df = message_filter(df, n_words=None, subset=subset)
            if not (by_sender or over_time):
                n_words = n_words == "default" and 200 or n_words
                title = f"Top {n_words} {title_part}"
                data = df.dropna(subset="content_striped")[
                    strip_emoji and "content_striped" or "content"
                ].value_counts()[:n_words]
                if return_data:
                    return data
                return charts["single"][kind](data, title)
            elif by_sender:
                n_words = n_words == "default" and 100 or n_words
                suptitle = f"Top {n_words} {title_part} by sender"
                data_lst = [
                    message_filter(df, None, False, sub)
                    .dropna(subset="content_striped")[
                        strip_emoji and "content_striped" or "content"
                    ]
                    .value_counts()[:n_words]
                    for sub in ["me", "others"]
                ]
                if return_data:
                    return data_lst
                return charts["sender"][kind](
                    data_lst,
                    self.data.sender_name.nunique() == 2
                    and [get_meta_data()["user"]]
                    + np.setdiff1d(
                        self.data.sender_name.unique(), get_meta_data()["user"]
                    ).tolist()
                    or ["YOU", "FRIENDS"],
                    suptitle,
                )
            elif over_time:
                n_words = n_words == "default" and 50 or n_words
                suptitle = f"Top {n_words} {title_part} for each {dt_hrchy}"
                df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)
                df.set_index("timestamp_ms", inplace=True)
                dts = sorted(df.index.unique())
                data_lst = [
                    strip_emoji
                    and df.loc[dt]
                    .dropna(subset="content_striped")[
                        strip_emoji and "content_striped" or "content"
                    ]
                    .value_counts()[:n_words]
                    for dt in dts
                ]
                return charts["time"][kind](data_lst, dts, suptitle)
        else:
            raise ValueError(
                f"{level} is not a valid level, valid levels are 'word', 'phrase' and 'message' "
            )

    @doc(occur="in your chat and by whom it sent and how often")
    def phrase_occurance(self, phrase):
        """
        Searches for specific words occurance {occur}

        Parameters
        ----------
        phrase : str
            The text to search for, typically text of any length of words.

        Returns
        -------
        pd.Series
        """
        return (
            self.data.groupby("sender_name")
            .content.agg(lambda rows: " ".join(rows))
            .str.count(f"(?<!\S){phrase}(?!\S)")
            .sort_values(ascending=False)
        )


@doc(
    Words, occur="in every chat and and the number of times it is repeated.",
)
class Multi_Words(Words):
    @doc(
        Words.plot,
        chat="""per_chat : bool, default False
    Divide your data into n subsets, one for each included chat,
    hence n charts will be shown, one for each subset.
    By default, your Top chats will be included,
    unless `chats_to_include` used.
chats_to_include : list-like objects, default []
    When per_chat, chats to include.
    Any chat other than them will be filtered unless 'others' is True.
others : bool, default False
    When per_chat, group any chat other the included ones and label them as 'Others'.""",
    )
    def plot(
        self,
        kind="word_cloud",
        level="word",
        per_chat=False,
        chats_to_include=[],
        others=False,
        by_sender=False,
        over_time=False,
        dt_hrchy="year",
        n_words="default",
        max_word_length=2,
        subset=None,
        strip_emoji=True,
        return_data=False,
    ):

        if per_chat:
            if kind not in ["word_cloud", "bubble"]:
                raise ValueError(
                    f"{kind} is not a valid chart type, Choose between 'word_cloud' and 'bubble' "
                )
            if dt_hrchy not in freq_mapper.keys():
                raise ValueError(
                    f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
                )
            if not isinstance(n_words, int) and n_words != "default":
                raise TypeError(
                    f"expected 'int' or 'default' got '{type(n_words)}' instead"
                )
            if not isinstance(max_word_length, int):
                raise TypeError(f"expected 'int'  got '{type(n_words)}' instead")
            if subset and subset not in ["me", "others"]:
                raise ValueError(
                    f"{subset} is not a valid subset; valid values are 'me', 'others'"
                )

            title_part = (
                level == "word"
                and "words"
                or level == "phrase"
                and f"phrases with {max_word_length} or less words"
                or "message"
            )
            n_words = n_words == "default" and 50 or n_words
            suptitle = f"Top {n_words} {title_part} for each chat"
            df = self.data.copy()

            if level == "word":
                if subset:
                    df = message_filter(df, n_words=None, subset=subset)
                df, n = cat_filter(
                    df, 0, "chat", include=chats_to_include or "Top", others=others
                )
                data_s = (
                    df[["chat", "content"]]
                    .dropna()
                    .groupby("chat")
                    .agg(lambda rows: Counter(re.split(r"[\b\W\b]+", " ".join(rows))))
                    .content.map(trim_counter_sw)
                )
                # return data_s
                # .squeeze()
                # data_s=isinstance(data_s,Counter) and [(data_s)] or data_s.map(trim_counter_sw)
                data_lst = [
                    pd.Series(dict(data.most_common(n_words)))
                    for idx, data in enumerate(data_s)
                ]
                if return_data:
                    return data_lst
                return charts["time"][kind](data_lst, data_s.index, suptitle)
            elif level == "phrase":
                if strip_emoji:
                    df.content = df.content_striped
                if subset:
                    df = message_filter(df, n_words=None, subset=subset)
                df, n = cat_filter(
                    df, 0, "chat", include=chats_to_include or "Top", others=others
                )
                chats = df.chat.unique()
                data_lst = [
                    message_filter(
                        df[df.chat == chat].content, max_word_length, strip_emoji
                    ).value_counts()[:n_words]
                    for chat in chats
                ]
                if return_data:
                    return data_lst
                return charts["time"][kind](data_lst, chats, suptitle)

            elif level == "message":
                if subset:
                    df = message_filter(df, n_words=None, subset=subset)
                df, n = cat_filter(
                    df, 0, "chat", include=chats_to_include or "Top", others=others
                )
                chats = df.chat.unique()
                data_lst = [
                    df[df.chat == chat]
                    .dropna(subset="content_striped")[
                        strip_emoji and "content_striped" or "content"
                    ]
                    .value_counts()[:n_words]
                    for chat in chats
                ]
                if return_data:
                    return data_lst
                return charts["time"][kind](data_lst, chats, suptitle)
            else:
                raise ValueError(
                    f"{level} is not a valid level, valid levels are 'word', 'phrase' and 'message' "
                )

        return super(Multi_Words, self).plot(
            kind,
            level,
            by_sender,
            over_time,
            dt_hrchy,
            n_words,
            max_word_length,
            subset,
            strip_emoji,
            return_data,
        )

    @doc(
        Words.phrase_occurance,
        occur="""in every chat and and the number of times it is repeated,
and returns the Top 20 results""",
    )
    def phrase_occurance(self, phrase):
        return (
            self.data.groupby("chat")
            .content.agg(lambda rows: " ".join(rows))
            .str.count(f"(?<!\S){phrase}(?!\S)")
            .sort_values(ascending=False)[:20]
        )
