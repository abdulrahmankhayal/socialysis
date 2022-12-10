from .common_plots import plot, multicat_plot, chat_plot
from .plot_helper import top_timely, metric_correct
from .media import media_plot
from pandas.util._decorators import doc


class share_plot(multicat_plot):
    @doc(media_plot.bar, cat="domain", alias="link domain", metric="links")
    def bar(
        self,
        by_domain=False,
        n_domain=5,
        include_domain=[],
        exclude_domain=[],
        domain_groups={},
        domain_others=False,
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
        cat = "domain"
        return super(share_plot, self).bar(
            by_domain,
            n_domain,
            include_domain,
            exclude_domain,
            domain_groups,
            domain_others,
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

    @doc(media_plot.line, cat="domain", alias="link domain", metric="links")
    def line(
        self,
        by_domain=False,
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
        cat = "domain"
        return super(share_plot, self).line(
            by_domain,
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

    @doc(media_plot.pie, cat="domain", alias="link domain", metric="links")
    def pie(
        self,
        by_domain=False,
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
        cat = "domain"
        return super(share_plot, self).pie(
            by_domain,
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
        n_domain=5,
        include_domain=[],
        exclude_domain=[],
        domain_groups={},
        domain_others=False,
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

        cat = "domain"
        return super(share_plot, self).dot(
            n_domain,
            include_domain,
            exclude_domain,
            domain_groups,
            domain_others,
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

    @doc(media_plot.gantt_chart, cat="domain", alias="link domain", metric="links")
    def gantt_chart(
        self,
        by_domain=False,
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
        return super(share_plot, self).gantt_chart(
            by_domain,
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
            cat="domain",
            **kwargs
        )


class chat_share_plot(plot):
    def bar(
        self,
        by_domain=False,
        over_time=False,
        n_domain=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_domain=[],
        exclude_domain=[],
        domain_groups={},
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
        Make vertical bar plot/s of chat links data using link domain as its categorical variable.

        See stats.Links.plot.bar
        """
        cat = by_domain and "domain" or "sender_name"
        per_chat = by_domain or False
        include = by_domain and include_domain or []
        exclude = by_domain and exclude_domain or []
        n = by_domain and n_domain or 0
        chat_groups = by_domain and domain_groups or {}

        return super(chat_share_plot, self).bar(
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
        by_domain=False,
        n_domain=5,
        aggr="count",
        cumulative=False,
        cum_aggr="sum",
        window="default",
        include_domain=[],
        exclude_domain=[],
        domain_groups={},
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
        Make line chart/s of chat links data using link domain as its categorical variable.

        See stats.Links.plot.line
        """
        cat = by_domain and "domain" or "sender_name"
        per_chat = by_domain or False
        include = by_domain and include_domain or []
        exclude = by_domain and exclude_domain or []
        n = by_domain and n_domain or 0
        chat_groups = by_domain and domain_groups or {}
        return super(chat_share_plot, self).line(
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
        by_domain=False,
        n_domain=5,
        include_domain=[],
        exclude_domain=[],
        domain_groups={},
        others=True,
        return_data=False,
        aggr="count",
        label=True,
        **kwargs
    ):
        """
        Make pie chart of chat links data using link domain as its categorical variable.

        See stats.Links.plot.pie
        """
        cat = by_domain and "domain" or "sender_name"
        per_chat = by_domain or False
        include = by_domain and include_domain or []
        exclude = by_domain and exclude_domain or []
        n = by_domain and n_domain or 0
        chat_groups = by_domain and domain_groups or {}
        return super(chat_share_plot, self).pie(
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
        include_domain=[],
        n_domain=5,
        exclude_domain=[],
        domain_groups={},
        legend=False,
        others=False,
        return_data=False,
        style=False,
        dt_range="last year",
        **kwargs
    ):
        """
        Make dot plot of chat links data using link domain as its categorical variable.

        See stats.Links.plot.dot
        """
        by_domain = x == "domain"
        cat = by_domain and "domain" or "sender_name"
        include = by_domain and include_domain or []
        exclude = by_domain and exclude_domain or []
        n = by_domain and n_domain or 0
        chat_groups = by_domain and domain_groups or {}
        return super(chat_share_plot, self).dot(
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
        include=[],
        n=5,
        exclude=[],
        groups={},
        others=False,
        return_data=False,
        style=False,
        **kwargs
    ):
        """
        Make gantt chart of chat links data using link domain as its categorical variable.

        See stats.Links.plot.gantt_chart
        """
        return super(chat_share_plot, self).gantt_chart(
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
            cat="domain",
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
