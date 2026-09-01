"""Shared evaluation constants and helpers."""

import numpy as np

FS = 256


def to_mask(annotations_obj):
    mask = np.zeros(int(annotations_obj.events[0]["recordingDuration"] * FS))
    for event in annotations_obj.events:
        if event["eventType"].value != "bckg":
            mask[
                round(event["onset"] * FS) : round(
                    (event["onset"] + event["duration"]) * FS
                )
            ] = 1
    return mask
