from .common_plots import plot, multicat_plot, chat_plot
from .plot_helper import metric_correct, top_timely
from pandas.util._decorators import doc


class media_plot(multicat_plot):
    @doc(cat="media", alias="media type", metric="media files")
    def bar(
        self,
        by_media=False,
        n_media=6,
        include_media=[],
        exclude_media=[],
        media_groups={},
        media_others=False,
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
        **kwargs
    ):
        """
        Make vertical bar plot/s.
        
        You can create many variations of bar plots such as chat/{cat} bars vs counts,
        bars vs datetime, chats/{alias}s vs datetime, chat/{cat} subplots and {cat} vs. chats.

        Chat bars can answer questions like "Who are the top people we shared {metric} with?" and 
        "How many {metric} did we send to each other's?", How are these people compared to one another?,
        While {cat} bars can answer questions like "Which {alias} used the most on chats?",
        "What {alias} you prefer vs what type your freinds prefer?" 
        
        Bars vs. datetime can answer questions like "How many {metric} you
        sent/recieved over time?".
        
        Chats/{alias} types vs. datetime bars combine the uses of chat/{cat} bars and bars vs. datetime.
        Generally, it allows you to see how number of {metric} varies for each chat/{alias} while taking time into account.

        {cat} vs. chats allows you to see top chats for each {alias}.

        Parameters
        ----------
        by_{cat} :  bool, default False, per_chat : bool, default False
            Choose between categorize bars by chat or by {cat}, 
            or use them together to make chat vs {cat} bars.
        over_time : bool, default False
            Whether to plot data over time or not.
        n : int or 'All', default 5
            when per_chat, number of chats to include -- sorted.
            It has no effect when `include` parameter is used or ``sort=='dynamic'``.
            Used along with `include`,`exclude`,`others`,`asc` to filter the data.
        n_{cat} : int or 'All', default 6
            same as n but for {cat}
        aggr : str or function, default 'count'
            The aggregation function to apply on data, any valid pandas aggregation function.
            Allowed aggr for categorical data : 
                {{'count','nunique'}} (nunique is the equivalent of count distinct).
            Custom aggregation functions are also allowed.
        cumulative : bool, default False
            Whether to apply a cumulative aggregation or not.
            Used along with `window` paramter to select a moving or a running aggreagtion.
        cum_aggr : {{"count","nunique","sum","mean","max","min","var","std"}}, default 'sum'
            The aggregation to be applied for the cumulation.
        window : int or 'default', default 'default'
            Used to define the window within it the aggreagtion is done.
            By default it is the length of the data which calculate running aggregates
            Choose a suitable window length to calculate moving aggregates.
        include : list or 'Top', default []
            if per_chat, select which chats to include,
            Use 'Top' to Include the 5 people you chat with the most in total,
            regardless of how many messages they have sent, photos sent, etc.
        include_{cat} : list, default []
            same as include but for {cat}
        exclude : list, default []
            if per_chat, used to exclude specific chats
        exclude_{cat} : list, default []
            same as exclude but for {cat}
        chat_groups : dict, default {{}}
            if per_chat, used to group chats, for example:
            {{'Sienna Meyer':'Family','Sebastien Meyer':'Family'
            ,'Shawn Harris':'Friends','Kelly Richardson':'Friends'}}
            this will group Sienna Meyer,Sebastien Meyer as 
            a single chat labeled 'Family' and the same with 'Friends'
            When used, you must specify the group to which every chat 
            appears in the plot belong, otherwise it will be labeled as 'Others'.
        {cat}_groups : dict, default {{}}
            same as chat_groups but for {cat}
        others : bool, default False
            if per_chat and when True, group any chat other than the selected n 
            or the included ones and label them as 'Others'
        {cat}_others : bool, default False
            same as others but for {cat}
        dt_disc : bool, default False
            when True, datetime is treated as a discrete(categorical) variable
            used along with `dt_hrchy`,`dt_range`,`dt_groups` to manipulate datetime
        dt_hrchy : {{'minute','hour','day','week','month','quarter','year'}}, default 'month'
            used to select the hierarchy(frequancy) of the datetime variable
        dt_range : list or a str , default 'last year'
            used to select a range to plot the data within
            pass a list with 2 elements first for the start date, the other for the end
            valid input looks like : ['2020','2022-01-01'], ['2019-01-01','2021-01'],
            ['2022-01-01 01:02:22','2022-02-05 11'], 'last year', 'last 2 weeks', 'first 100 months'
            use None or 'Start'/'End' to select the data from the beginning/ending
            such as ['Start','2022'] ,  ['2010','End'] , [None,'2020-02'], ['Start','End']
            When passing a string, format it like {{first/last}} then a value and end with a time period
            When ``dt_range=None`` the full time range is selected
        dt_groups : dict, default {{}}
            if dt_disc, used to group datetime values 
            for example, it can be used to group hours of the day as follows:
                {{**dict.fromkeys([1,2,3], 'Too late'), 
                **dict.fromkeys([4,5,6], 'Too early'),
                **dict.fromkeys([7,8,9,10,11], 'Early'),
                **dict.fromkeys([12,13,14,15,16,17], 'Mid day'),
                **dict.fromkeys([18,19,20,21], 'First of night'),
                **dict.fromkeys([22,23,0], 'Late')}}
        subplots : bool, default False
            When per_chat/by_{cat} and over_time both are True,
            It is used to create subplots, one per chat/{alias} for every selected chat/{alias}.
        stacked : bool, default 'default'
            whether to stack bars together or not,
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
            has no effect when per_chat/by_{cat} and `show_sender=False`
        asc : bool, default False
            if `n/n_{cat}` is used , Sort ascending or descending.
        show_sender : bool, default True
            whether to show sender_name or not
        sort : {{'fixed', 'dynamic'}}, default 'fixed'
            Used when per_chat/by_{cat} and over_time.
            Used to select a 'fixed' `n` chats/{alias} for all the time intervals,
            Who/which are in total the Top over the selected period,
            or select them 'dynamic'ally , each interval with its own Top `n` chats/{alias}.
            Has no effect when include/{cat}_include is used.
            When per_chat and by_{cat}, dynamic sorting is applied regardless of your choice.
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

        cat = "media"

        return super(media_plot, self).bar(
            by_media,
            n_media,
            include_media,
            exclude_media,
            media_groups,
            media_others,
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
            cat,
            **kwargs
        )

    @doc(cat="media", alias="media type", metric="media files")
    def line(
        self,
        by_media=False,
        per_chat=False,
        n=6,
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
        **kwargs
    ):
        """
        Make line plot/s.
        
        Line plots are useful to display timeline of your data to help you identify trends.
        It can help you answer questions like "How {metric} you
        sent/recieved changes over time?" in a continous way, you can also include chats/{alias}s to plot
        a line for each chat/{alias} to see how this vary from one chat/{alias} to another.
        
        They are most useful when it comes to dealing with wide date ranges.

        Parameters
        ----------
        by_{cat} :  bool, default False, per_chat : bool, default False
            Choose between categorize lines by chat or by {cat}.
            only one category should be selected.
        Other Parameters
        ----------------
        See stats.plot.line

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them.
        """

        cat = "media"
        return super(media_plot, self).line(
            by_media,
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
            cat,
            **kwargs
        )

    @doc(cat="media", alias="media type")
    def pie(
        self,
        by_media=False,
        n=6,
        include=[],
        exclude=[],
        groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs
    ):
        """
        Generate a pie plot.
        
        Parameters
        ----------
        by_{cat} :  bool, default False
            make pie chart out of {alias}, default is per chat.
        Other Parameters
        ----------------
        See stats.plot.pie

        Returns
        -------
        matplotlib.axes.Axes
        """
        cat = "media"
        return super(media_plot, self).pie(
            by_media,
            n,
            include,
            exclude,
            groups,
            others,
            return_data,
            aggr,
            label,
            cat,
            **kwargs
        )

    def dot(
        self,
        n_media=6,
        include_media=[],
        exclude_media=[],
        media_groups={},
        media_others=False,
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
        **kwargs
    ):
        cat = "media"
        return super(media_plot, self).dot(
            n_media,
            include_media,
            exclude_media,
            media_groups,
            media_others,
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
            cat,
            **kwargs
        )

    def gantt_chart(
        self,
        by_media=False,
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
        **kwargs
    ):
        """
        Make Gantt chart.
        
        Parameters
        ----------
        by_{cat} :  bool, default False
            make gantt_chart out of {alias}, default is per chat.
        Other Parameters
        ----------------
        See stats.plot.gantt_chart

        Returns
        -------
        matplotlib.axes.Axes 
        """
        return super(media_plot, self).gantt_chart(
            by_media,
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
            cat="media",
            **kwargs
        )


class files_plot(multicat_plot):
    @doc(media_plot.bar, cat="ext", alias="file extension", metric="files")
    def bar(
        self,
        by_ext=False,
        n_ext=5,
        include_ext=[],
        exclude_ext=[],
        ext_groups={},
        ext_others=False,
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
        **kwargs
    ):
        cat = "ext"

        return super(files_plot, self).bar(
            by_ext,
            n_ext,
            include_ext,
            exclude_ext,
            ext_groups,
            ext_others,
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
            cat,
            **kwargs
        )

    @doc(media_plot.line, cat="ext", alias="file extension", metric="files")
    def line(
        self,
        by_ext=False,
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
        **kwargs
    ):
        cat = "ext"
        return super(files_plot, self).line(
            by_ext,
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
            cat,
            **kwargs
        )

    @doc(media_plot.pie, cat="ext", alias="file extension")
    def pie(
        self,
        by_ext=False,
        n=5,
        include=[],
        exclude=[],
        groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs
    ):
        cat = "ext"
        return super(files_plot, self).pie(
            by_ext,
            n,
            include,
            exclude,
            groups,
            others,
            return_data,
            aggr,
            label,
            cat,
            **kwargs
        )

    def dot(
        self,
        n_ext=5,
        include_ext=[],
        exclude_ext=[],
        ext_groups={},
        ext_others=False,
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
        **kwargs
    ):

        cat = "ext"
        return super(files_plot, self).dot(
            n_ext,
            include_ext,
            exclude_ext,
            ext_groups,
            ext_others,
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
            cat,
            **kwargs
        )

    @doc(media_plot.gantt_chart, cat="ext", alias="file extension")
    def gantt_chart(
        self,
        by_ext=False,
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
        **kwargs
    ):
        return super(files_plot, self).gantt_chart(
            by_ext,
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
            cat="ext",
            **kwargs
        )


class chat_media_plot(plot):
    def bar(
        self,
        by_media=False,
        over_time=False,
        n_media=6,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_media=[],
        exclude_media=[],
        media_groups={},
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
        **kwargs
    ):
        """
        Make vertical bar plot/s of chat media data using media type as its categorical variable.

        See stats.Media.plot.bar
        """
        if not by_media and not over_time:
            raise Exception(
                "You have not specified any variable to plot the data by; Set by_media, over_time, or both to True"
            )

        cat = by_media and "media" or "sender_name"
        per_chat = by_media or False
        include = by_media and include_media or []
        exclude = by_media and exclude_media or []
        n = by_media and n_media or 0
        chat_groups = by_media and media_groups or {}

        return super(chat_media_plot, self).bar(
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
            **kwargs
        )

    def line(
        self,
        by_media=False,
        n_media=6,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_media=[],
        exclude_media=[],
        media_groups={},
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
        **kwargs
    ):
        """
        Make line chart/s of chat media data using media type as its categorical variable.

        See stats.Media.plot.line
        """
        cat = by_media and "media" or "sender_name"
        per_chat = by_media or False
        include = by_media and include_media or []
        exclude = by_media and exclude_media or []
        n = by_media and n_media or 0
        chat_groups = by_media and media_groups or {}
        return super(chat_media_plot, self).line(
            per_chat,
            n,
            aggr,
            cumulative,
            cum_aggr,
            window,
            include,
            exclude,
            chat_groups,
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
            **kwargs
        )

    def pie(
        self,
        by_media=False,
        n_media=6,
        include_media=[],
        exclude_media=[],
        media_groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs
    ):
        """
        Make pie chart of chat media data using media type as its categorical variable.

        See stats.Media.plot.pie
        """
        cat = by_media and "media" or "sender_name"
        per_chat = by_media or False
        include = by_media and include_media or []
        exclude = by_media and exclude_media or []
        n = by_media and n_media or 0
        chat_groups = by_media and media_groups or {}
        return super(chat_media_plot, self).pie(
            n,
            include,
            exclude,
            chat_groups,
            others,
            return_data,
            aggr,
            label,
            cat=cat,
            **kwargs
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
        include_media=[],
        n_media=6,
        exclude_media=[],
        media_groups={},
        legend=False,
        others=False,
        return_data=False,
        style=False,
        dt_range="last year",
        **kwargs
    ):
        """
        Make dot plot of chat media data using media type as its categorical variable.

        See stats.Media.plot.dot
        """
        by_media = x == "media"
        cat = by_media and "media" or "sender_name"
        include = by_media and include_media or []
        exclude = by_media and exclude_media or []
        n = by_media and n_media or 0
        chat_groups = by_media and media_groups or {}
        return super(chat_media_plot, self).dot(
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
            **kwargs
        )

    def gantt_chart(
        self,
        by_media=False,
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
        **kwargs
    ):
        """
        Make gantt chart of chat media data using media type as its categorical variable.

        See stats.Media.plot.gantt_chart
        """
        return super(chat_media_plot, self).gantt_chart(
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
            cat="media",
            **kwargs
        )

    @doc(chat_plot.ranking)
    def ranking(
        self,
        n=15,
        dt_hrchy="day",
        cat="sender_name",
        aggr="count",
        dt_range=None,
        show_name=False,
        legend=True,
        rank=False,
        **kwargs
    ):
        metric = metric_correct(self._metric, aggr)
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
            **kwargs
        )


class chat_files_plot(plot):
    def bar(
        self,
        by_ext=False,
        over_time=False,
        n_ext=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_ext=[],
        exclude_ext=[],
        ext_groups={},
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
        **kwargs
    ):
        """
        Make vertical bar plot/s of chat files data using file extension as its categorical variable.

        See stats.Media.Files.plot.bar
        """
        if not by_ext and not over_time:
            raise Exception(
                "You have not specified any variable to plot the data by; Set by_ext, over_time, or both to True"
            )
        cat = by_ext and "ext" or "sender_name"
        per_chat = by_ext or False
        include = by_ext and include_ext or []
        exclude = by_ext and exclude_ext or []
        n = by_ext and n_ext or 0
        chat_groups = by_ext and ext_groups or {}

        return super(chat_files_plot, self).bar(
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
            **kwargs
        )

    def line(
        self,
        by_ext=False,
        n_ext=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_ext=[],
        exclude_ext=[],
        ext_groups={},
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
        **kwargs
    ):
        """
        Make line chart/s of chat files data using file extension as its categorical variable.

        See stats.Media.Files.plot.line
        """

        cat = by_ext and "ext" or "sender_name"
        per_chat = by_ext or False
        include = by_ext and include_ext or []
        exclude = by_ext and exclude_ext or []
        n = by_ext and n_ext or 0
        chat_groups = by_ext and ext_groups or {}
        return super(chat_files_plot, self).line(
            per_chat,
            n,
            aggr,
            cumulative,
            cum_aggr,
            window,
            include,
            exclude,
            chat_groups,
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
            **kwargs
        )

    def pie(
        self,
        by_ext=False,
        n_ext=5,
        include_ext=[],
        exclude_ext=[],
        ext_groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs
    ):
        """
        Make pie chart of chat files data using file extension as its categorical variable.

        See stats.Media.Files.plot.pie
        """
        cat = by_ext and "ext" or "sender_name"
        per_chat = by_ext or False
        include = by_ext and include_ext or []
        exclude = by_ext and exclude_ext or []
        n = by_ext and n_ext or 0
        chat_groups = by_ext and ext_groups or {}
        return super(chat_files_plot, self).pie(
            n,
            include,
            exclude,
            chat_groups,
            others,
            return_data,
            aggr,
            label,
            cat=cat,
            **kwargs
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
        include_ext=[],
        n_ext=5,
        exclude_ext=[],
        ext_groups={},
        legend=False,
        others=False,
        return_data=False,
        style=False,
        dt_range="last year",
        **kwargs
    ):
        """
        Make dot plot of chat files data using file extension as its categorical variable.

        See stats.Media.Files.plot.dot
        """
        by_ext = x == "ext"
        cat = by_ext and "ext" or "sender_name"
        include = by_ext and include_ext or []
        exclude = by_ext and exclude_ext or []
        n = by_ext and n_ext or 0
        chat_groups = by_ext and ext_groups or {}
        return super(chat_files_plot, self).dot(
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
            **kwargs
        )

    def gantt_chart(
        self,
        by_ext=False,
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
        **kwargs
    ):
        """
        Make gantt chart of chat files data using file extension as its categorical variable.

        See stats.Media.Files.plot.gantt_chart
        """
        return super(chat_files_plot, self).gantt_chart(
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
            cat="ext",
            **kwargs
        )

    @doc(chat_plot.ranking)
    def ranking(
        self,
        n=15,
        dt_hrchy="day",
        cat="sender_name",
        aggr="count",
        dt_range=None,
        show_name=False,
        legend=True,
        rank=False,
        **kwargs
    ):
        metric = metric_correct(self._metric, aggr)
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
            **kwargs
        )
