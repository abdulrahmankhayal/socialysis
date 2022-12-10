from .plot_helper import (
    metric_correct,
    metric_categories,
    dt_filter,
    title_freq,
    stck_dflts,
    cat_filter,
    cat_Vs_counts,
    dt_Vs_cat,
    freq_mapper,
    make_pretty,
    colors,
    dt_Vs_counts,
    top_timely,
    sum_of_most,
    most_per_chat,
    dt_Vs_counts_bar,
    dynamic_sort,
    nxtlvlofdtils,
    valid_cum_aggr,
    cat_display,
    metric_display,
)
from .special_plots import dot_plot, sunburst
import pandas as pd
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from pandas.util._decorators import doc


class plot:
    def __init__(self, metric, passed_data):
        self._metric = metric
        self.data = passed_data

    def bar(
        self,
        per_chat=False,
        over_time=False,
        n=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include=[],
        exclude=[],
        chat_groups={},
        others=False,
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        dt_groups={},
        subplots=False,
        stacked="default",
        return_data=False,
        style=False,
        label=True,
        legend=True,
        asc=False,
        show_sender=True,
        sort="fixed",
        **kwargs,
    ):
        """
        Make vertical bar plot/s.
        
        You can create many variations of bar plots such as chat bars vs counts,
        bars vs datetime, chats vs datetime and chat subplots.

        Chat bars can answer questions like "Who are the top people I chatted with?" and 
        "How many messages/ photos/ voices/ videos/.... did we send to each other's?" 
        How are these people compared to one another? , and for data with numbers (audio, video, and calls),
        it can answer further questions such as: what the total length of audio/ video/ calls you 
        sent to each other is, and how this differs from one chat to another.
        
        Bars vs. datetime can answer questions like "How many messages/photos/voices/videos/... you
        sent/recieved over time?" and for data with numbers it can answer fruther questions like
        "How the length of audio/video/calls you sent/recieved vary from one time to another?"
        
        Chats vs. datetime bars combine the uses of chat bars and bars vs. datetime, it can answer question
        like "How the messages/photos/voices/videos/... i sent and received varies over time for every chat?" or
        "How has the people I chat with changed over time, and how has the number of
        messages/photos/voices/videos/... between us changed from time to time?".
        Generally, it allows you to see how things have changed from one chat to another while taking time into account.

        Parameters
        ----------
        per_chat : bool, default False
            Whether to categorize bars by chats or not.
        over_time : bool, default False
            Whether to plot data over time or not.
        n : int or 'All', default 5
            If per_chat, number of chats to include -- sorted.
            It has no effect when `include` parameter is used or ``sort=='dynamic'``.
            Used along with `include`,`exclude`,`others`,`asc` to filter the data.
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
            Custom aggregation functions are also allowed.
            Numeric data is limited to audio, video, and call data
        cumulative : bool, default False
            Whether to apply a cumulative aggregation or not.
            Used along with `window` paramter to select a moving or a running aggreagtion.
        cum_aggr : {"count","nunique","sum","mean","max","min","var","std"}, default 'sum'
            The aggregation to be applied for the cumulation.
        window : int or 'default', default 'default'
            Used to define the window within it the aggreagtion is done.
            By default it is the length of the data which calculate running aggregates
            Choose a suitable window length to calculate moving aggregates.
        include : list or 'Top', default []
            if per_chat, select which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        exclude : list, default []
            if per_chat, used to exclude specific chats
        chat_groups : dict, default {}
            if per_chat, used to group chats, for example:
            {'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        others : bool, default False
            if per_chat and when True, group any chat other than the selected n 
            or the included ones and label them as 'Others'
        dt_disc : bool, default False
            when True, datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range`,`dt_groups` to manipulate datetime
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
        dt_groups : dict, default {}
            if dt_disc, used to group datetime values 
            used in the same way as chat_groups
            for example, it can be used to group hours of the day as follows:
                {**dict.fromkeys([1,2,3], 'Too late'), 
                **dict.fromkeys([4,5,6], 'Too early'),
                **dict.fromkeys([7,8,9,10,11], 'Early'),
                **dict.fromkeys([12,13,14,15,16,17], 'Mid day'),
                **dict.fromkeys([18,19,20,21], 'First of night'),
                **dict.fromkeys([22,23,0], 'Late')}
        subplots : bool, default False
            When per_chat and over_time both are True,
            It is used to create subplots, one per chat for every selected chat.
        stacked : bool, default 'default'
            whether to stack category bars together or not,
            the category when per_chat or over_time is 
            the sender_name(the name of the person who sent the message in a chat),
            and the chat itself when per_chat and over_time used togeher.
            by default, the best choice is selected based on the aggr you passed.
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table 
        label : bool, default True
            whether to show the text(annotation) of the bar values (heights)
        legend : bool, default True
            whether to show the plot legend or not
            has no effect when per_chat and `show_sender=False`
        asc : bool, default False
            if `n` is used , Sort ascending or descending.
        show_sender : bool, default True
            whether to show sender_name or not
        sort : {'fixed', 'dynamic'}, default 'fixed'
            Used when per_chat and over_time.
            Used to select a 'fixed' `n` {alias} for all the time intervals,
            Who/which are in total the Top over the selected period,
            or select them 'dynamic'ally , each interval with its own Top `n` {alias}s.
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
        matplotlib.axes.Axes or np.ndarray of them

        Notes
        -----
        When return_data, no Axes are returned, just the data.
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot", "cat"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        # print(cat)
        if "cat" not in kwargs.keys():
            kwargs["cat"] = "chat"

        metric = metric_correct(self._metric, aggr)

        df = self.data.copy()

        if dt_range:
            df = dt_filter(df, dt_range)

        if stacked == "default":
            stacked = stck_dflts.get(aggr, False)

        if per_chat:

            df, n = cat_filter(
                df,
                n,
                kwargs.get("cat"),
                include,
                exclude,
                others,
                chat_groups,
                asc,
                sort=over_time and sort or "fixed",
                metric=metric,
                aggr=aggr,
            )
            if not over_time:

                return cat_Vs_counts(
                    df,
                    metric=metric,
                    aggr=aggr,
                    asc=asc,
                    n=n,
                    stacked=stacked,
                    show_sender=show_sender,
                    legend=legend,
                    label=label,
                    return_data=return_data,
                    style=style,
                    **kwargs,
                )
            else:
                return dt_Vs_cat(
                    df,
                    metric=metric,
                    stacked=stacked,
                    n=n,
                    m=0,
                    sort=sort,
                    aggr=aggr,
                    cumulative=cumulative,
                    cum_aggr=cum_aggr,
                    window=window,
                    dt_disc=dt_disc,
                    dt_hrchy=dt_hrchy,
                    dt_groups=dt_groups,
                    subplots=subplots,
                    return_data=return_data,
                    style=style,
                    label=label,
                    legend=legend,
                    asc=asc,
                    **kwargs,
                )

        else:
            if over_time:

                return dt_Vs_counts_bar(
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
                    label,
                    stacked,
                    return_data,
                    style,
                    legend,
                    **kwargs,
                )

            raise Exception(
                "You have not specified any variable to plot the data by; Set per_chat, over_time, or both to True"
            )

    def line(
        self,
        per_chat,
        n=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include=[],
        exclude=[],
        chat_groups={},
        asc=False,
        show_sender=False,
        others=False,
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        subplots=False,
        return_data=False,
        style=False,
        legend=True,
        area=False,
        **kwargs,
    ):
        """
        Make line plot/s.
        
        Line plots are useful to display timeline of your data to help you identify trends
        It can help you answer questions like "How messages/photos/voices/videos/... you
        sent/recieved changes over time?" in a continous way, you can also include chats to plot
        a line for each chat to see how this vary from one chat to another.
        
        They are most useful when it comes to dealing with wide date ranges.
        
        Parameters
        ----------
        per_chat : bool, required
            whether to make different lines for each chat or combine them together in an single one
        n : int or 'All', default 5
            If per_chat, number of chats to include -- sorted.
            It has no effect when `include` parameter is used.
            Used along with `include`,`exclude`,`others`,`asc` to filter the data.
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'}.
            Custom aggregation functions are also allowed.
            Numeric data is limited to audio, video, and call data.
        cumulative : bool, default False
            Whether to apply a cumulative aggregation or not.
            Used along with `window` paramter to select a moving or a running aggreagtion.
        cum_aggr : {"count","nunique","sum","mean","max","min","var","std"}, default 'sum'
            The aggregation to be applied for the cumulation.
        window : int or 'default', default 'default'
            Used to define the window within it the aggreagtion is done.
            By default it is the length of the data which calculate running aggregates
            Choose a suitable window length to calculate moving aggregates.
        include : list or 'Top', default []
            if per_chat, select which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        exclude : list, default []
            if per_chat, used to exclude specific chats
        chat_groups : dict, default {}
            if per_chat, used to group chats, for example:
            {'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        asc : bool, default False
            if `n` is used , Sort ascending or descending.
        show_sender : bool, default False
            when per_chat is False, you can plot 2 lines one for messages you sent,
            and another for messages you received.
        others : bool, default False
            if per_chat and when True, group any chat other than the selected n 
            or the included ones and label them as 'Others'
        dt_disc : bool, default False
            when True, datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range`,`dt_groups` to manipulate datetime
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
            When ``dt_range=None`` the full time range is selected.
        subplots : bool, default False
            When per_chat, used to create subplots, one line for every selected chat.
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table 
        legend : bool, default True
            whether to show the plot legend or not
            has no effect when not per_chat and not show_sender
        area : bool, default False
            if per_chat and when True plot area chart instead of lines
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and heigh
            rot : int , typically angle between 0,360
                change the rotation x and y axis tick labels

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot", "cat"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        if dt_hrchy not in freq_mapper.keys():
            raise ValueError(
                f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
            )

        cat = kwargs.get("cat", "chat")
        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        if dt_range:
            df = dt_filter(df, dt_range)
        if per_chat:
            df, n = cat_filter(
                df,
                n,
                include=include,
                exclude=exclude,
                others=others,
                groups=chat_groups,
                metric=metric,
                aggr=aggr,
                cat=cat,
            )
            plot_area = {True: "area", False: "line"}
            groupping_params = {
                True: ["timestamp_ms", cat],
                False: [
                    pd.Grouper(key="timestamp_ms", freq=freq_mapper[dt_hrchy]),
                    cat,
                ],
            }

            if dt_disc:
                if dt_hrchy in ["hour", "min"]:
                    df.timestamp_ms = df.timestamp_ms.dt.floor(freq_mapper[dt_hrchy])
                else:
                    df.timestamp_ms = getattr(df.timestamp_ms.dt, dt_hrchy)

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
            df = df[df.agg(sum, axis=0).sort_values(ascending=asc).index[:n]]

            if return_data:
                if style:
                    return df.style.pipe(make_pretty)
                return df
            df.columns = df.columns.map(
                lambda name: get_display(arabic_reshaper.reshape(name))
            )
            if len(df.columns) < 2:
                subplots = False
            if subplots:
                fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, sharey=True)
                for idx, col in enumerate(df.columns):
                    subset = df[col]
                    subset.plot(
                        xlabel="",
                        ax=axes[idx],
                        figsize=kwargs.get("figsize", (10, 10)),
                        label=col,
                        color=colors[idx],
                        rot=kwargs.get("rot"),
                    )
                    if legend:
                        axes[idx].legend(loc=(1.02, 0.5))
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
                    f"{aggr} of {metric}".upper(),
                    rotation="vertical",
                    fontsize=12,
                )

                return axes
            ax = df.iloc[:, :n].plot(
                kind=plot_area[area],
                figsize=kwargs.get("figsize", (20, 10)),
                legend=legend,
                rot=kwargs.get("rot"),
                xlabel="",
            )
            plt.title(
                f"{cum_subtitle_start} {title_freq[dt_hrchy]} Timeline of the {aggr} of {metric_display[metric]} across {cat_display[cat]} {cum_subtitle_end}",
                fontsize=12,
                fontweight="bold",
            )
            plt.ylabel(
                f"{aggr} of {metric_display[metric]}".upper(), fontdict={"size": 12}
            )
            return ax

        df = dt_Vs_counts(
            df,
            metric,
            aggr=aggr,
            cumulative=cumulative,
            cum_aggr=cum_aggr,
            window=window,
            dt_disc=dt_disc,
            dt_hrchy=dt_hrchy,
            dt_groups=0,
            show_sender=show_sender,
        )
        if return_data:
            if style:
                return df.style.pipe(make_pretty)
            return df
        if len(df.columns) < 2:
            legend = False
        ax = df.plot.line(
            figsize=kwargs.get("figsize", (15, 10)),
            legend=legend,
            rot=kwargs.get("rot", 45),
            xlabel="",
        )
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

    def pie(
        self,
        n=5,
        include=[],
        exclude=[],
        chat_groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs,
    ):
        """
        Generate a pie plot.

        Pie plots are a great way to illustrate the proportion of your data.
        It can be used to answer questions like "How the total messages/photos/voices/videos/...
        are divided between chats?".

        Parameters
        ----------
        n : int or 'All', default 5
            If per_chat, number of chats to include -- sorted.
            It has no effect when `include` parameter is used.
        include : list or 'Top', default []
            Used instead of `n` if you want to specify which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        exclude : list, default []
            Used to exclude specific chats.
        chat_groups : dict, default {}
            Used to group chats, for example:
            {'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        others : bool, default True
            Group any chat other than the selected n or the included ones,
            and label them as 'Others' to see the proportion of your selected
            chats against the total of all chats not the total of the only selected ones.
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
                custom aggregation function is also allowed.
            Numeric data is limited to audio, video, and call data
        label : bool, default True
            Whether to show annotations of the proportions values and their percentages or not
            
        **kwargs : 
            figsize : tuple (w, h) 
                control the plot width and height
                
        Returns
        -------
        matplotlib.axes.Axes
        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "cat"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        cat = kwargs.get("cat", "chat")
        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        df, _ = cat_filter(
            df,
            cat=cat,
            n=n,
            include=include,
            exclude=exclude,
            groups=chat_groups,
            others=others,
            metric=metric,
            aggr=aggr,
        )

        df = df.groupby(cat).agg({metric: aggr})
        df.index = df.index.map(lambda name: get_display(arabic_reshaper.reshape(name)))
        df = df.sort_values(metric, ascending=False)
        explode = [False] * len(df)
        if len(df) == 2:
            explode = [0.0, 0.1]
        if return_data:
            return df
        ax = df.plot.pie(
            figsize=kwargs.get("figsize", (10, 10)),
            y=metric,
            legend=False,
            ylabel="",
            autopct=label
            and (
                lambda pct: "{:.1f}%\n{:d} ".format(
                    pct, int(np.round(pct / 100.0 * np.sum(df)))
                )
            )
            or None,
            textprops={"fontsize": 14},
            startangle=-270,
            explode=explode,
            counterclock=False,
            wedgeprops={"edgecolor": "k", "linewidth": 2, "antialiased": True},
            pctdistance=0.85,
        )

        plt.title(
            f"{cat_display[cat]} proportion of {metric_display[metric]}",
            fontsize=12,
            fontweight="bold",
        )
        return ax

    def sunburst(
        self,
        hrchy_levels=["sender_name"],
        filters={},
        subset=None,
        label_value=True,
        label_name=True,
        **kwargs,
    ):
        """
        Sunburst chart is ideal for displaying more than one category in a hierarchical view,
        it can be considered as a multi-level pie chart.
        
        Parameters
        ----------
        hrchy_levels : list, default ['sender_name']
            Levels of hierarchy, the categories to be included --ordered.
            Chat is included by default.
            sender_name is the only avaliable extra category for all classes except:
            ['media'] for Media, ['ext'] for Media.Files, ['domain'] for Links,
            ['call_type','call_status'] for Calls
        filters : dict, default {}
            The filters to apply for each category included.
            It excpect an input structured like :
            {'cat_1':{filters},'cat_2':{filters},....}
            for example : {'chat':{'n':8,'others':True,'exclude':['Ahmed Mohamed']},
            'domain':{'include':['facebook','twitter']}}.
            filters are : n,others,include,exclude,groups
            by default chats are limted to 5 chats, while any other category
            limited to 2 with others set to True.
        subset : str or list, default None
            To use only a subset of a category.
            for example you can show only messages you send by set subset='You'
            or subset=['You',['facebook','twitter']] to show only links you shared that
            is a facebook link or twitter link.  
        label_value : bool or list, default True
            Whether to show each slice value or not.
            list can be passed for each level include chat.
            for example, if hrchy_levels=['sender_name'],
            you can pass [True,False] to label_value,
            in order to show chat values and hide sender_name values.
        label_name : bool or list, default True
            Whether to show each slice name or not.
            list can be passed for each level include chat.
        **kwargs : 
            figsize : tuple (w, h) 
                control the plot width and height
        
        Returns
        -------
        matplotlib.axes.Axes
        """

        metric = metric_correct(self._metric, "count")
        return sunburst(
            self.data,
            metric,
            hrchy_levels,
            filters,
            subset,
            label_value,
            label_name,
            **kwargs,
        )

    def dot(
        self,
        x="month",
        y="day",
        disc=(0, 1),
        aggr="count",
        showcmap=True,
        size=200,
        clip=None,
        include=[],
        n=5,
        exclude=[],
        chat_groups={},
        legend=False,
        others=False,
        return_data=False,
        style=False,
        dt_range="last year",
        **kwargs,
    ):
        """
        Make dot plot.

        Parameters
        ----------
        x : {'chat','minute','hour','day','week','month','quarter','year'}, default 'month'
            The variable to plot on the x axis.
            You can pass a datetime interval to create a date vs. date plot,
            or 'chat' to create a category vs date plot.
        y : {'minute','hour','day','week','month','quarter','year'}, default 'day'
            The variable to plot on the y axis.
            Only datetime intervals are allowed.
        disc : bool or tuple of bool : default (0,1)
            whether to treat datetime a discrete(categorical) variable or not
            if both x and y are datetime intervals pass a tuple for each axis,
            when x='chat' you can pass a bool for short to be applied on y datetime.
        showcmap : bool, default True
            Whether to show the color map scale bar or not.
        size : int or 'nunique', default 200
            The size of a single dot.
            'nunique' adds another level of details to your plot,
            for example if y is set to 'month', the size of each dot will be equal
            to the unique count of days you chatted with each person for each month.
            nunique only has effect when x='chat' and disc=False
        clip : dict, default None
            Controls the range of cmap.
            Used to adjust the color density of your dots.
            It is very helpful when your data contain outliers
            (one or more chat is extremely higher than the others).
            you can use 'vmin' to specify the start point of cmap,
            'vmax' for the end point, and 'vcenter' for the middle point, or 
            you can use one of implemted methods.
            there are 2 avaliable methods, 'unbalanced' and 'cut_outliers',
            'unbalanced' set the middle point to the median of the data, 
            'cut_outliers' outliers identified based on IQR method and then cmap is clipped to remove them.
            you should either specify 'v' values or a 'method' such as :
            {'vmin':'10'}, {'vmin':10,'vmax':300,'vcenter':'30'},{'method':'unbalanced'}
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
                custom aggregation function is also allowed.
            Numeric data is limited to audio, video, and call data.
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'
            use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected.
        include : list or 'Top', default []
            when x ='chat', select which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        exclude : list, default []
            when x ='chat', used to exclude specific chats
        chat_groups : dict, default {}
            when x ='chat', used to group chats for example:
            {'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        others : bool, default False
            when x ='chat' and when True, group any chat other than the selected n 
            or the included ones and label them as 'Others'
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table   
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and heigh
            rot : int , typically angle between 0,360
                change the rotation x and y axis tick labels
       
        Returns
        -------
        matplotlib.axes.Axes

        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "cat"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        legend_title = "Sizes"
        cat = kwargs.get("cat", "chat")
        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        if dt_range:
            df = dt_filter(df, dt_range)
        df.rename(columns={"timestamp_ms": "y"}, inplace=True)
        if y not in freq_mapper.keys():
            raise ValueError(
                f"{y} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};"
            )
        groupping_params = {}
        if x != cat:
            if type(disc) != tuple:
                raise TypeError(f"{disc} is not a tuple")
            if len(disc) != 2:
                raise ValueError(
                    f"expected a tuple with 2 values; when both x and y are datetime you must pass a tuple contain 2 values for x and y"
                )
            if x not in freq_mapper.keys():
                raise ValueError(
                    f"{x} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};/n if you are trying to pass the categorical variable, allowed categories are {metric_categories[self._metric]}"
                )
            if y == cat:
                raise ValueError(f"the categorical variable can only be passed to x")

            df["x"] = df["y"]
            groupping_params = {
                (0, 0): [
                    pd.Grouper(key="x", freq=freq_mapper[x]),
                    pd.Grouper(key="y", freq=freq_mapper[y]),
                ],
                (0, 1): [pd.Grouper(key="x", freq=freq_mapper[x]), "y"],
                (1, 0): ["x", pd.Grouper(key="y", freq=freq_mapper[y])],
                (1, 1): ["x", "y"],
            }
            for val, axis in zip(disc, ["x", "y"]):
                if val:
                    df[axis] = getattr(df[axis].dt, locals()[axis])
            if disc[0] and x in ["month", "week", "day", "hour"]:
                df["x"] = df["x"].astype("str").str.zfill(2)
            if disc[1] and y in ["month", "week", "day", "hour"]:
                df["y"] = df["y"].astype("str").str.zfill(2)
            hide_axis = False

        else:
            df, _ = cat_filter(
                df,
                n,
                cat=cat,
                include=include,
                exclude=exclude,
                others=others,
                groups=chat_groups,
                metric=metric,
                aggr=aggr,
            )
            df.rename(columns={cat: "x"}, inplace=True)
            groupping_params = {
                0: ["x", pd.Grouper(key="y", freq=freq_mapper[y])],
                (0, 0): ["x", pd.Grouper(key="y", freq=freq_mapper[y])],
                1: ["x", "y"],
                (0, 1): ["x", "y"],
            }
            if disc not in [0, 1, (0, 0), (0, 1)]:
                raise ValueError(
                    "allowed values for disc when x is categorical are 0, 1, (0,0), (0,1)"
                )
            hide_axis = True
            # df.x=df["x"].map(lambda name:get_display(arabic_reshaper.reshape(name)))
            if disc and type(disc) != tuple or type(disc) == tuple and disc[1]:
                df["y"] = getattr(df.y.dt, y)
            if size == "nunique" and not disc:
                size = df.groupby(
                    [pd.Grouper(key="y", freq=freq_mapper[nxtlvlofdtils[y]]), "x"]
                ).agg({metric: "size"})
                size = (
                    size.reset_index()
                    .groupby(["x", pd.Grouper(key="y", freq=freq_mapper[y])])
                    .count()
                    .reset_index()
                )
                size = size.pivot(index="x", columns="y", values=metric).fillna(0)
                size = size.T.values.flatten() ** 2
                # legend_title=f"Count of distinct active {nxtlvlofdtils[y]}s per {y}"
                legend_title = f"Unique active {nxtlvlofdtils[y]}s"

            elif size == "nxtlvl" and disc:
                size = 200
        df = (
            df.groupby(groupping_params[disc])
            .agg({metric: aggr})
            .reset_index()
            .pivot(index="x", columns="y", values=metric)
        )
        if return_data:
            if style:
                return df[:n].style.pipe(make_pretty)
            return df
        if x == "chat":
            df.index = df.index.map(
                lambda name: get_display(arabic_reshaper.reshape(name))
            )

        ax = dot_plot(
            df,
            showcmap,
            legend,
            clip,
            size=size,
            legend_title=legend_title,
            hide_axis=hide_axis,
            **kwargs,
        )
        plt.title(f"{y.title()} vs. {x.title()}", fontsize=12, fontweight="bold")
        return ax

    def gantt_chart(
        self,
        msg_freq="day",
        ylabel_freq="month",
        dt_range="last year",
        n=5,
        include=[],
        exclude=[],
        chat_groups={},
        others=False,
        return_data=False,
        style=False,
        **kwargs,
    ):
        """
        Make Gantt chart.

        It can help you see chatting streaks, how frequently you chat with your friends,
        days you chat with them compared to the days you don't,
        and your pattern of chatting with different people.

        Parameters
        ----------
        msg_freq : {'minute','hour','day','week','month','quarter','year'}, default 'day'
            the level at which a bar chart if created if a message exist
        ylabel_freq : {'minute','hour','day','week','month','quarter','year'}, default 'month'
            the interval between y axis tick labels
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'.
            Use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        n : int or 'All', default 5
            If per_chat, number of chats to include -- sorted.
            It has no effect when `include` parameter is used.
        include : list or 'Top', default []
            select which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        exclude : list, default []
            exclude specific chats
        chat_groups : dict, default {}
            used to group chats for example:
            {'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        others : bool, default False
            when True, group any chat other than the selected n 
            or the included ones and label them as 'Others'
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table   

        Returns
        -------
        matplotlib.axes.Axes

        """
        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "cat"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        if msg_freq not in freq_mapper.keys():
            raise ValueError(
                f"{msg_freq} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};"
            )
        if ylabel_freq not in freq_mapper.keys():
            raise ValueError(
                f"{ylabel_freq} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};"
            )
        cat = kwargs.get("cat", "chat")
        df = self.data.copy()
        metric = metric_correct(self._metric, "count")
        df, _ = cat_filter(
            df,
            n,
            include=include,
            exclude=exclude,
            others=others,
            groups=chat_groups,
            metric=metric,
            aggr="count",
            cat=cat,
        )
        if dt_range:
            df = dt_filter(df, dt_range)
        x = df[cat].unique()
        y = pd.date_range(
            df.timestamp_ms.min().to_period(freq_mapper[ylabel_freq][0]).to_timestamp(),
            df.timestamp_ms.min(),
            freq=freq_mapper[msg_freq],
        )[:-1]
        X, Y = np.meshgrid(x, y)
        xy = pd.DataFrame({cat: X.flatten(), "timestamp_ms": Y.flatten()})

        try:
            df.timestamp_ms = df.timestamp_ms.astype(
                f"datetime64[{freq_mapper[msg_freq][0]}]"
            )
        except:
            df.timestamp_ms = df.timestamp_ms.astype(
                f"datetime64[{freq_mapper[msg_freq][0].lower()}]"
            )
        df = df[["timestamp_ms", cat]].drop_duplicates()
        df = pd.concat([xy, df[["timestamp_ms", cat]]]).drop_duplicates()

        df["n"] = [1] * len(df)
        df = df.pivot(index="timestamp_ms", columns=cat, values="n")
        df.columns = df.columns.map(
            lambda name: get_display(arabic_reshaper.reshape(name))
        )
        order = {
            key: value
            for (value, key) in enumerate(df.sum().sort_values(ascending=False).index)
        }

        df = (
            df.reindex(
                pd.date_range(df.index[0], df.index[-1], freq=freq_mapper[msg_freq])
            )
            .fillna(0)
            .reset_index()
            .melt(id_vars="index", value_vars=df.columns, value_name="exist")
        )

        df["dt_index"] = (
            df["index"].dt.to_period(freq_mapper[msg_freq][0])
            - df["index"][0].to_period(freq_mapper[msg_freq][0])
        ).apply(attrgetter("n"))

        df.sort_values(
            ["index", cat], key=lambda x: x.map(order).fillna(x), inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        if return_data:
            # df.replace(df[cat].unique,x,inplace=True)
            if style:
                return df[len(xy) :].style.pipe(make_pretty)
            return df[len(xy) :]
        fig, ax = plt.subplots(1, figsize=kwargs.get("figsize", (10, 15)))
        ax.bar(
            df[cat][len(xy) :],
            df.exist[len(xy) :],
            bottom=df.dt_index[len(xy) :],
            width=0.5,
            label="",
        )
        xticks = list(
            map(
                attrgetter("n"),
                pd.date_range(
                    start=df["index"].min(),
                    end=df["index"].max(),
                    freq=freq_mapper[ylabel_freq],
                ).to_period(freq_mapper[msg_freq][0])
                - pd.to_datetime(df["index"].min()).to_period(freq_mapper[msg_freq][0]),
            )
        )
        ax.set_yticks(xticks)
        plt.xticks(rotation=kwargs.get("rot", 0))
        plt.title(
            f"{title_freq[msg_freq]} active {metric_display[metric]} sending across {cat_display[cat]}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_yticklabels(
            pd.date_range(
                start=df["index"].min(),
                end=df["index"].max(),
                freq=freq_mapper[ylabel_freq],
            ).date,
            ha="right",
        )
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_alpha(0.2)
        ax.tick_params(axis="both", length=0)
        return ax

    def ranking(
        self,
        key="time",
        n=15,
        dt_hrchy="day",
        cat="chat",
        count_of_maxs=False,
        aggr="count",
        dt_range=None,
        show_name=False,
        legend=True,
        rank=False,
        **kwargs,
    ):
        """
        Apply multi-level(chained) aggregation on your data
        
        It can be used to answer questions such as :
        Q1-What is the Top 15 highest number of messages/photos/voices/videos/... sent by one person on a single day?
        Q2=How the highest number of messages/photos/voices/videos/... sent by one person varies from one month to another?
        Q3-What is the highest number of messages/photos/voices/videos/... sent in one day for each chat?
        Q4-How often each chat was the chat with the most messages/photos/voices/videos/... per day?

        Parameters
        ----------
        key : {'time','cat'}, default 'time'
            Used to specify the variable of the bars axis.
            for example, Question 1 and 2 depends on time, while Q3,Q4 is all about chats
        n : int, default 15
            The number of observations to display.
        dt_hrchy : {'minute','hour','day','week','month','quarter','year'}, default 'day'
            used to select the level/frequancy at which the aggregation is done
        count_of_maxs : bool, default False
            when key is 'chat', used to apply additional count aggregation on the aggregated data 
            to count number of times a chat was the Top, needed to answer Q4.
        aggr : str or function, default 'count'
            The first level of aggregation to apply on data.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'}.
            Custom aggregation function is also allowed.
            Numeric data are only limited to audio, video, and call data
        dt_range : list or a str , default 'last year'
            used to select a date range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'.
            Use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        show_name : bool, default False
            whether to annotate chat names on bars or not
            useful when legend is messy
        legend : bool, default True
            whether to show the plot legend or not
        rank : bool, default False,
            when key is 'time', instead of show sequential timeline of bars, sort them (Q1).
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.
        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them

        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot", "cat"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        # cat=kwargs.get('cat',"chat")
        if dt_hrchy not in freq_mapper.keys():
            raise ValueError(
                f"{dt_hrchy} not a valid datetime frequency; supported frequencies are {list(freq_mapper.keys())}"
            )
        metric = metric_correct(self._metric, aggr)
        if cat not in sum(metric_categories.values(), []):
            raise ValueError(
                f"{cat} is not a valid category; valid categories for {self._metric} are {metric_categories[self._metric]}"
            )
        if key == "time":
            return top_timely(
                self.data,
                metric,
                aggr,
                cat,
                n,
                dt_hrchy,
                dt_range,
                show_name,
                legend,
                rank,
                **kwargs,
            )
        elif key == "cat":
            if count_of_maxs:
                return sum_of_most(self.data, metric, aggr, cat, n, dt_hrchy, **kwargs)
            else:
                return most_per_chat(self.data, metric, aggr, n, dt_hrchy, **kwargs)
        else:
            raise ValueError(
                f"{key} is not a valid value for key; supported values are 'time', 'cat' "
            )


class multicat_plot(plot):
    def bar(
        self,
        by_cat=False,
        n_cat=5,
        include_cat=[],
        exclude_cat=[],
        cat_groups={},
        cat_others=False,
        per_chat=False,
        over_time=False,
        n=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include=[],
        exclude=[],
        chat_groups={},
        others=False,
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        dt_groups={},
        subplots=False,
        stacked="default",
        return_data=False,
        style=False,
        label=True,
        legend=True,
        asc=False,
        show_sender=True,
        sort="fixed",
        cat="chat",
        **kwargs,
    ):

        if by_cat and per_chat and over_time:
            raise ValueError(
                f"You can only select 2 out of the 3 parameters together, by_{cat}, per_chat, and over_time were selected"
            )
        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        metric = metric_correct(self._metric, aggr)
        if by_cat and per_chat:

            df = self.data.copy()
            # media filter
            df, n_cat = cat_filter(
                df,
                n=n_cat,
                cat=cat,
                include=include_cat,
                exclude=exclude_cat,
                groups=cat_groups,
                others=cat_others,
                asc=asc,
                sort="fixed",
                metric=metric,
            )

            if dt_range:
                df = dt_filter(df, dt_range)

            if stacked == "default":
                stacked = stck_dflts.get(aggr, False)

            # chat_filter
            df, n = cat_filter(
                df,
                n=n,
                cat="chat",
                include=include,
                exclude=exclude,
                groups=chat_groups,
                others=others,
                sort=not return_data and "dynamic" or "fixed",
                metric=metric,
            )

            df = (
                df.groupby([cat, "chat"])
                .agg({metric: aggr})
                .reset_index()
                .pivot(index=cat, columns="chat", values=metric)
            )
            df = df.sort_index(key=df.sum(1).get, ascending=False).fillna(0)
            if return_data:
                if style:
                    return df[:n].style.pipe(make_pretty)
                return df
            axes, fig = dynamic_sort(
                df[:n_cat],
                label=label,
                legend=legend,
                width=3 * n_cat,
                loc="upper right",
                bbox_to_anchor=(0.9, 0.88),
                **kwargs,
            )
            fig.suptitle(f"{cat_display[cat]} Vs Chats", fontsize=12, fontweight="bold")
            return axes
        cat = by_cat and cat or "chat"
        if by_cat:
            include = include_cat
            exclude = exclude_cat
            n = n_cat
            chat_groups = cat_groups
            per_chat = by_cat
            others = cat_others
        return super(multicat_plot, self).bar(
            per_chat,
            over_time,
            n,
            aggr,
            cumulative,
            cum_aggr,
            window,
            include,
            exclude,
            chat_groups,
            others,
            dt_disc,
            dt_hrchy,
            dt_range,
            dt_groups,
            subplots,
            stacked,
            return_data,
            style,
            label,
            legend,
            asc,
            show_sender,
            sort,
            cat=cat,
            **kwargs,
        )

    def line(
        self,
        by_cat=False,
        per_chat=False,
        n=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include=[],
        exclude=[],
        groups={},
        asc=False,
        show_sender=False,
        others=False,
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        subplots=False,
        return_data=False,
        style=False,
        legend=True,
        area=False,
        cat="chat",
        **kwargs,
    ):
        cat = by_cat and cat or "chat"
        if by_cat:
            per_chat = True
        return super(multicat_plot, self).line(
            per_chat,
            n,
            aggr,
            cumulative,
            cum_aggr,
            window,
            include,
            exclude,
            groups,
            asc,
            show_sender,
            others,
            dt_disc,
            dt_hrchy,
            dt_range,
            subplots,
            return_data,
            style,
            legend,
            area,
            cat=cat,
            **kwargs,
        )

    def pie(
        self,
        by_cat=False,
        n=5,
        include=[],
        exclude=[],
        groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        cat="chat",
        **kwargs,
    ):
        cat = by_cat and cat or "chat"
        return super(multicat_plot, self).pie(
            n,
            include,
            exclude,
            groups,
            others,
            return_data,
            aggr,
            label,
            cat=cat,
            **kwargs,
        )

    def dot(
        self,
        n_cat=5,
        include_cat=[],
        exclude_cat=[],
        cat_groups={},
        cat_others=False,
        x="month",
        y="day",
        disc=(0, 1),
        aggr="count",
        showcmap=True,
        size=200,
        clip=None,
        include=[],
        n=5,
        exclude=[],
        chat_groups={},
        legend=False,
        others=False,
        return_data=False,
        style=False,
        dt_range="last year",
        cat="chat",
        **kwargs,
    ):

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "rot"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        metric = metric_correct(self._metric, aggr)
        if x + y in [f"{cat}chat", f"chat{cat}"]:
            df = self.data.copy()
            df, _ = cat_filter(
                df,
                n=n,
                cat="chat",
                include=include,
                exclude=exclude,
                groups=chat_groups,
                others=others,
                metric=metric,
            )
            df, _ = cat_filter(
                df,
                n=n_cat,
                cat=cat,
                include=include_cat,
                exclude=exclude_cat,
                groups=cat_groups,
                others=cat_others,
                metric=metric,
            )
            df.rename(columns={y: "y"}, inplace=True)
            df.rename(columns={x: "x"}, inplace=True)
            if x == "chat":
                df["x"] = df["x"].map(
                    lambda name: get_display(arabic_reshaper.reshape(name))
                )
            else:
                df["y"] = df["y"].map(
                    lambda name: get_display(arabic_reshaper.reshape(name))
                )
            df = (
                df.groupby(["x", "y"])
                .agg({metric: aggr})
                .reset_index()
                .pivot(index="x", columns="y", values=metric)
            )
            if return_data:
                return df

            return dot_plot(df, showcmap, legend, clip, size=size, **kwargs)
        cat = x == cat and cat or "chat"
        if x == cat:
            include = include_cat
            exclude = exclude_cat
            n = n_cat
            chat_groups = cat_groups
        return super(multicat_plot, self).dot(
            x,
            y,
            disc,
            aggr,
            showcmap,
            size,
            clip,
            include,
            n,
            exclude,
            chat_groups,
            legend,
            others,
            return_data,
            style,
            dt_range,
            cat=cat,
            **kwargs,
        )

    def gantt_chart(
        self,
        by_cat=False,
        msg_freq="day",
        ylabel_freq="month",
        dt_range="last year",
        n=5,
        include=[],
        exclude=[],
        groups={},
        others=False,
        return_data=False,
        style=False,
        cat="chat",
        **kwargs,
    ):
        cat = by_cat and cat or "chat"
        return super(multicat_plot, self).gantt_chart(
            msg_freq,
            ylabel_freq,
            dt_range,
            n,
            include,
            exclude,
            groups,
            others,
            return_data,
            style,
            cat=cat,
            **kwargs,
        )


class chat_plot:
    def __init__(self, metric, passed_data):
        self._metric = metric
        self.data = passed_data

    def pie(self, aggr="count", label=True, **kwargs):
        """
        Generate a 2-slice pie plot, one for you, the other for your friend.

        Pie plots are a great way to illustrate the proportion of your data.
        It can be used to answer questions like "Out of total messages/photos/voices/videos/...,
        how many messages/photos/voices/videos/... you have sent vs how many your friend have sent.

        Parameters
        ----------
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
                custom aggregation function is also allowed.
            Numeric data is limited to audio, video, and call data
        label : bool, default True
            Whether to show annotations of the proportions values and their percentages or not
            
        **kwargs : 
            figsize : tuple (w, h) 
                control the plot width and height
        
        Returns
        -------
        matplotlib.axes.Axes
        """
        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "cat"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )
        cat = kwargs.get("cat", "sender_name")
        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        df = df.groupby(cat).agg({metric: aggr})
        ax = df.plot.pie(
            figsize=kwargs.get("figsize", (10, 10)),
            y=metric,
            legend=False,
            ylabel="",
            explode=[0.0, 0.1],
            autopct=label
            and (
                lambda pct: "{:.1f}%\n{:d} ".format(
                    pct, int(np.round(pct / 100.0 * np.sum(df)))
                )
            )
            or None,
            textprops={"fontsize": 14},
            startangle=-270,
            counterclock=False,
            wedgeprops={"edgecolor": "k", "linewidth": 2, "antialiased": True},
        )
        plt.title(
            f"{cat_display[cat]} proportion of {metric_display[metric]}",
            fontsize=12,
            fontweight="bold",
        )
        return ax

    def dot(
        self,
        x="month",
        y="day",
        disc=(0, 1),
        aggr="count",
        showcmap=True,
        size=200,
        clip=None,
        legend=False,
        return_data=False,
        style=False,
        dt_range="last year",
        **kwargs,
    ):
        """
        Make dot plot.

        Parameters
        ----------
        x : {minute','hour','day','week','month','quarter','year'}, default 'month'
            The variable to plot on the x axis.
        y : {'minute','hour','day','week','month','quarter','year'}, default 'day'
            The variable to plot on the y axis.
        disc : tuple of bool : default (0,1)
            whether to treat datetime a discrete(categorical) variable or not
            pass a tuple for each axis,
        showcmap : bool, default True
            Whether to show the color map scale bar or not.
        size : int or '', default 200
            The size of a single dot.
            '' adds another level of details to your plot,
            the size of each dot equals the 
            only has effect when x='chat'
        clip : dict, default None
            Controls the range of cmap.
            Used to adjust the color density of your dots.
            It is very helpful when your data contain outliers
            (one or more chat is extremely higher than the others).
            you can use 'vmin' to specify the start point of cmap,
            'vmax' for the end point, and 'vcenter' for the middle point, or 
            you can use one of implemted methods.
            there are 2 avaliable methods, 'unbalanced' and 'cut_outliers',
            'unbalanced' set the middle point to the median of the data, 
            'cut_outliers' outliers identified based on IQR method and then cmap is clipped to remove them.
            you should either specify 'v' values or a 'method' such as :
            {'vmin':'10'}, {'vmin':10,'vmax':300,'vcenter':'30'},{'method':'unbalanced'}
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
                custom aggregation function is also allowed.
            Numeric data is limited to audio, video, and call data.
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'
            use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {first/last} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected.
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table   

        Returns
        -------
        matplotlib.axes.Axes

        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "rot"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        if x not in freq_mapper.keys():
            raise ValueError(
                f"{x} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};"
            )

        if y not in freq_mapper.keys():
            raise ValueError(
                f"{y} not a valid datetime frequency;/n supported frequencies are {list(freq_mapper.keys())};"
            )
        if type(disc) != tuple:
            raise TypeError(f"{disc} is not a tuple")
        if len(disc) != 2:
            raise ValueError(
                f"expected a tuple with 2 values; when both x and y are datetime you must pass a tuple contain 2 values for x and y"
            )

        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        if dt_range:
            df = dt_filter(df, dt_range)
        df.rename(columns={"timestamp_ms": "y"}, inplace=True)
        groupping_params = {}
        df["x"] = df["y"]
        groupping_params = {
            (0, 0): [
                pd.Grouper(key="x", freq=freq_mapper[x]),
                pd.Grouper(key="y", freq=freq_mapper[y]),
            ],
            (0, 1): [pd.Grouper(key="x", freq=freq_mapper[x]), "y"],
            (1, 0): ["x", pd.Grouper(key="y", freq=freq_mapper[y])],
            (1, 1): ["x", "y"],
        }
        for val, axis in zip(disc, ["x", "y"]):
            if val:
                df[axis] = getattr(df[axis].dt, locals()[axis])
        if disc[0] and x in ["month", "week", "day", "hour"]:
            df["x"] = df["x"].astype("str").str.zfill(2)
        if disc[1] and y in ["month", "week", "day", "hour"]:
            df["y"] = df["y"].astype("str").str.zfill(2)

        df = (
            df.groupby(groupping_params[disc])
            .agg({metric: aggr})
            .reset_index()
            .pivot(index="x", columns="y", values=metric)
        )

        if return_data:
            if style:
                df.style.pipe(make_pretty)
            return df
        ax = dot_plot(df, showcmap, legend, clip, size=size, **kwargs)
        plt.title(f"{y.title()} vs. {x.title()}", fontsize=12, fontweight="bold")
        return ax

    def bar(
        self,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        dt_groups={},
        stacked="default",
        return_data=False,
        style=False,
        label=True,
        legend=True,
        show_sender=True,
        **kwargs,
    ):
        """
        Make vertical bar plot/s.
        
        It used to create bars vs datetime for a single chat.

        Bars vs. datetime can answer questions like "How many messages/photos/voices/videos/... you
        sent/recieved over time?" and for data with numbers(audio/video/calls) it can answer fruther questions like
        "How the length of audio/video/calls you sent/recieved vary from one time to another?"
        
        Parameters
        ----------
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'},
            Custom aggregation functions are also allowed.
            Numeric data is limited to audio, video, and call data.
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
            datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range`,`dt_groups` to manipulate datetime
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
        dt_groups : dict, default {}
            if dt_disc, used to group datetime values 
                for example, it can be used to group hours of the day as follows:
                {**dict.fromkeys([1,2,3], 'Too late'), 
                **dict.fromkeys([4,5,6], 'Too early'),
                **dict.fromkeys([7,8,9,10,11], 'Early'),
                **dict.fromkeys([12,13,14,15,16,17], 'Mid day'),
                **dict.fromkeys([18,19,20,21], 'First of night'),
                **dict.fromkeys([22,23,0], 'Late')}.
        stacked : bool, default 'default'
            whether to stack sender_name bars together or not,
            by default, the best choice is selected based on the aggr you passed.
            has no effect when show_sender=False.
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table 
        label : bool, default True
            whether to show the text(annotation) of the bar values (heights)
        legend : bool, default True
            whether to show the plot legend or not
            has no effect when `show_sender=False`
        show_sender : bool, default True
            whether to show sender_name or not
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.
        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them

        Notes
        -----
        When return_data, no Axes are returned, just the data.
        """

        unexpected_kwargs = set(kwargs.keys()).difference(
            set(["figsize", "rot", "label_rot"])
        )
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        if dt_range:
            df = dt_filter(df, dt_range)

        sender = {False: "Others", True: df.chat.iloc[0]}
        if dt_range:
            df = dt_filter(df, dt_range)

        if stacked == "default":
            stacked = stck_dflts.get(aggr, False)

        return dt_Vs_counts_bar(
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
            label,
            stacked,
            return_data,
            style,
            legend,
            **kwargs,
        )

    def line(
        self,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        show_sender=False,
        dt_disc=False,
        dt_hrchy="month",
        dt_range="last year",
        return_data=False,
        style=False,
        legend=True,
        **kwargs,
    ):
        """
        Make line plot/s for chat data.
        
        Line plots are useful to display timeline of your data to help you identify trends
        It can help you answer questions like "How messages/photos/voices/videos/...
        you sent/recieved changes over time?" in a continous way.
        
        They are most useful when it comes to dealing with wide date ranges.
        
        Parameters
        ----------
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {'count','nunique'} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {'count','nunique','sum','mean','max','min','std','var','first','last'}.
            Custom aggregation functions are also allowed.
            Numeric data is limited to audio, video, and call data.
        cumulative : bool, default False
            Whether to apply a cumulative aggregation or not.
            Used along with `window` paramter to select a moving or a running aggreagtion.
        cum_aggr : {"count","nunique","sum","mean","max","min","var","std"}, default 'sum'
            The aggregation to be applied for the cumulation.
        window : int or 'default', default 'default'
            Used to define the window within it the aggreagtion is done.
            By default it is the length of the data which calculate running aggregates
            Choose a suitable window length to calculate moving aggregates.
        show_sender : bool, default False
            Make 2 subsets of your data, one for your messages,
            the other for your friend messages, and make 2 line charts, one for each subset .
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
        return_data : bool, default False
            When True, return a CrossTab of the plotting data
        style : bool, default False
            Used with return_data to sytle the returned CrossTab,
            typically, it creates highlighted table 
        legend : bool, default True
            whether to show the plot legend or not
            has no effect when not show_sender
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and heigh
            rot : int , typically angle between 0,360
                change the rotation x and y axis tick labels

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them

        Notes
        -----
        When return_data, no Axes are returned, just the data.
        """

        unexpected_kwargs = set(kwargs.keys()).difference(set(["figsize", "rot"]))
        if unexpected_kwargs:
            raise TypeError(
                f"got an unexpected keyword argument {unexpected_kwargs.pop()}"
            )

        metric = metric_correct(self._metric, aggr)
        df = self.data.copy()
        if dt_range:
            df = dt_filter(df, dt_range)

        df = dt_Vs_counts(
            df,
            metric,
            aggr=aggr,
            cumulative=cumulative,
            cum_aggr=cum_aggr,
            window=window,
            dt_disc=dt_disc,
            dt_hrchy=dt_hrchy,
            dt_groups={},
            show_sender=show_sender,
        )
        if return_data:
            if style:
                return df.style.pipe(make_pretty)
            return df
        if cumulative and window == "default":
            window == len(df)
        ax = df.plot.line(
            figsize=kwargs.get("figsize", (15, 10)),
            legend=show_sender and legend or False,
            rot=kwargs.get("rot", 45),
        )
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

    def ranking(
        self,
        n=15,
        dt_hrchy="day",
        rank=False,
        aggr="count",
        dt_range=None,
        legend=True,
        **kwargs,
    ):
        """
        Apply multi-level(chained) aggregation on your chat data
        
        It can be used to answer questions such as :
        -What is the Top 15 highest number of messages/photos/voices/videos/... sent by you/your friend on a single day?
        -For every month, who sent more  messages/photos/voices/videos/..., you or your friend?

        Parameters
        ----------
        n : int, default 15
            The number of observations to display.
        dt_hrchy : {'minute','hour','day','week','month','quarter','year'}, default 'day'
            used to select the level/frequancy at which the aggregation is done
        rank : bool, default False,
            when key is 'time', instead of show sequential timeline of bars, sort them 
        aggr : str or function, default 'count'
            The first level of aggregation to apply on data.
            Allowed aggr for categorical data : 
                {{'count','nunique'}} (nunique is the equivalent of count distinct).
            Allowed aggr for numeric data : 
                {{'count','nunique','sum','mean','max','min','std','var','first','last'}}.
            Custom aggregation function is also allowed.
            Numeric data are only limited to audio, video, and call data
        dt_range : list or a str , default 'last year'
            used to select a date range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'.
            Use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {{first/last}} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        legend : bool, default True
            whether to show the plot legend or not
        **kwargs : control the plot appearance
            figsize : tuple (w, h) 
                control the plot width and height
            rot : int , typically an angle between 0 and 360
                change the rotation x axis and y axis tick labels.
            label_rot : int , typically an angle between 0 and 360
                change the rotation of the annotation text.
        
        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them

        """

        metric = metric_correct(self._metric, aggr)
        return top_timely(
            self.data,
            metric,
            aggr,
            "sender_name",
            n,
            dt_hrchy,
            dt_range,
            False,
            legend,
            rank,
            **kwargs,
        )
