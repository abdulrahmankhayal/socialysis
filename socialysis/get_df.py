import numpy as np
import cv2
import os
import json
import pandas as pd
import librosa
from multiprocessing.pool import ThreadPool as Pool
import emoji
from tqdm import tqdm
import math
import regex

RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")


def remove_bad_chars(text):
    return RE_BAD_CHARS.sub("", text)


time_units = ["sec", "minute", "hour", "day"]
unit_factors = [60, 60, 24]


def get_unit_factor(cur_unit, unit):
    if unit not in time_units:
        raise ValueError(
            f"{unit} is not a valid time unit; valid units are {time_units}"
        )
    indces = [time_units.index(cur_unit), time_units.index(unit)]
    factor = math.prod(unit_factors[min(indces) : max(indces)])
    return factor, indces


def set_dur_unit(data, cur_unit, unit):

    factor, indces = get_unit_factor(cur_unit, unit)
    if indces[1] > indces[0]:
        data = data.map(lambda dur: round(dur / factor, 2))
    else:
        data = data.map(lambda dur: round(dur * factor, 2))
    return data


def get_video_dur(filename):
    try:
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = fps > 0 and round(frame_count / fps, 2) or 0
        cap.release()
        return duration
    except:
        return 0


def get_audio_length(uri):
    try:
        return librosa.get_duration(filename=uri)
    except:
        return 0


def parallel_audio(base, folders, max_workers=2):

    directory = base + "messages/inbox"
    s = (
        pd.Series(
            list(
                map(
                    lambda direc: os.path.isdir(direc)
                    and list(map(lambda txt: direc + "/" + txt, os.listdir(direc)))
                    or np.NaN,
                    list(map(lambda txt: directory + "/" + txt + "/audio", folders)),
                )
            )
        )
        .dropna()
        .explode()
    )

    # folders = os.listdir(directory)
    results = []

    with Pool(processes=max_workers) as p:
        max_ = len(s)
        with tqdm(total=max_) as pbar:
            for _ in p.imap(get_audio_length, s):
                results.append(_)
                pbar.update()

    s.index = results
    s = s.str.replace(base, "", regex=False)
    s = s = pd.Series(s.index, index=s.values)
    return s


def parse_json(base, parallel=False, process_audio=True, dur_unit="sec", **kwargs):

    jsons = {}
    base = base.replace("\\", "/")
    if base[-1] != "/":
        base += "/"
    directory = base + "messages/inbox"

    folders = os.listdir(directory)
    info = json.load(
        open(os.path.join(base, "messages\\autofill_information.json"), "r")
    )
    user = info[list(info)[-1]]["FULL_NAME"][0]
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")
    unit_factor = (
        (dur_unit == "sec" or parallel) and 1 or get_unit_factor("sec", dur_unit)[0]
    )
    if parallel and process_audio:
        print("Getting Audio Files Duration In Parallel ...")
        max_workers = kwargs.get("max_workers", 2)
        s = parallel_audio(base, folders, max_workers)
        if dur_unit != "sec":
            s = set_dur_unit(s, "sec", dur_unit)

    print("Parsing Data ...")
    for folder in tqdm(folders):
        for filename in os.listdir(os.path.join(directory, folder)):
            if filename.startswith("message"):
                data = json.load(open(os.path.join(directory, folder, filename), "r"))

                if len(data["participants"]) != 2:
                    continue
                if data["participants"][0]["name"] != user:
                    data["participants"] = {"name": data["participants"][0]["name"]}
                else:
                    data["participants"] = {"name": data["participants"][1]["name"]}
                if '"' in data["participants"]["name"]:
                    data["participants"]["name"] = data["participants"]["name"].replace(
                        '"', ""
                    )
                data["participants"]["name"] = r'"{}"'.format(
                    data["participants"]["name"]
                )
                data["participants"]["name"] = (
                    json.loads(data["participants"]["name"])
                    .encode("latin1")
                    .decode("utf8")
                )

                for indx, message in enumerate(data["messages"]):

                    if "audio_files" in data["messages"][indx].keys():
                        for aud_idx, audio in enumerate(
                            data["messages"][indx]["audio_files"]
                        ):
                            dirc = (
                                base
                                + data["messages"][indx]["audio_files"][aud_idx]["uri"]
                            )
                            try:
                                data["messages"][indx]["audio_files"][aud_idx][
                                    "length"
                                ] = (
                                    process_audio
                                    and (
                                        parallel
                                        and s[
                                            data["messages"][indx]["audio_files"][
                                                aud_idx
                                            ]["uri"]
                                        ]
                                        or librosa.get_duration(filename=dirc)
                                    )
                                    or 0
                                )
                            except:
                                data["messages"][indx]["audio_files"][aud_idx][
                                    "length"
                                ] = 0
                            if unit_factor != 1:
                                data["messages"][indx]["audio_files"][aud_idx][
                                    "length"
                                ] = round(
                                    data["messages"][indx]["audio_files"][aud_idx][
                                        "length"
                                    ]
                                    / unit_factor,
                                    2,
                                )
                    if "videos" in data["messages"][indx].keys():
                        for vid_idx, vid in enumerate(data["messages"][indx]["videos"]):
                            dirc = (
                                base + data["messages"][indx]["videos"][vid_idx]["uri"]
                            )
                            data["messages"][indx]["videos"][vid_idx][
                                "duration"
                            ] = get_video_dur(dirc)
                            if unit_factor != 1:
                                data["messages"][indx]["videos"][vid_idx][
                                    "duration"
                                ] = round(
                                    data["messages"][indx]["videos"][vid_idx][
                                        "duration"
                                    ]
                                    / unit_factor,
                                    2,
                                )

                    if "reactions" in data["messages"][indx].keys():
                        data["messages"][indx]["reactions"] = (
                            message["reactions"][0]["reaction"]
                            .encode("latin1")
                            .decode("utf8")
                        )

                    if "content" in data["messages"][indx].keys():
                        data["messages"][indx]["content"] = (
                            message["content"].encode("latin1").decode("utf8")
                        )

                    if "sender_name" in data["messages"][indx].keys():
                        data["messages"][indx]["sender_name"] = (
                            message["sender_name"].encode("latin1").decode("utf8")
                        )

                if data["participants"]["name"] in jsons.keys():

                    jsons[data["participants"]["name"]]["messages"] += data["messages"]
                else:

                    jsons[data["participants"]["name"]] = data
    return jsons


def df_from_jsons(base, parallel=False, process_audio=True, dur_unit="sec", **kwargs):

    tqdm.pandas()

    jsons = parse_json(base, parallel, process_audio, dur_unit, **kwargs)
    print("Building df ...")
    dflist = []
    for chat, json in jsons.items():
        temp = pd.DataFrame(json["messages"])
        temp["chat"] = chat
        dflist.append(temp)
    df = pd.concat(dflist).reset_index(drop=True)
    # df.timestamp_ms=df.timestamp_ms.apply(pd.to_datetime,unit='ms')
    df["call_status"] = df["call_duration"].map(
        lambda val: val > 0 and "replied" or "missed", na_action="ignore"
    )
    xemojis = (
        df.content.dropna()
        .apply(lambda txt: "".join([emo["emoji"] for emo in emoji.emoji_list(txt)]))
        .reset_index()
    )
    xemojis.columns = ["index", "emoji"]
    df = df.reset_index().merge(xemojis, how="left", on="index").replace("", np.NaN)
    df = df.join(
        df.dropna(subset="call_duration")["content"]
        .map(
            lambda txt: any(video_kw in txt for video_kw in ["فيديو", "video"])
            and "video"
            or "voice"
        )
        .rename("call_type"),
        how="left",
    )
    df.content = df.content.map(lambda txt: remove_bad_chars(txt), na_action="ignore")
    if dur_unit != "sec":
        df.call_duration = set_dur_unit(df.call_duration, "sec", dur_unit)
    print("Convert Timestamps to DateTime format ...")
    df.timestamp_ms = df.timestamp_ms.progress_apply(pd.to_datetime, unit="ms")
    cols_of_interest = [
        "index",
        "chat",
        "sender_name",
        "timestamp_ms",
        "content",
        "is_unsent",
        "share",
        "photos",
        "reactions",
        "call_duration",
        "audio_files",
        "gifs",
        "videos",
        "sticker",
        "files",
        "call_status",
        "emoji",
        "call_type",
    ]
    for col in cols_of_interest:
        if not hasattr(df, col):
            df[col] = np.nan
    return df[cols_of_interest]
