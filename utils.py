from datetime import datetime, timedelta

from parser import NGINX_Line


def get_paths_from_timestamp(
    data: list[NGINX_Line], after: datetime, before: datetime
) -> list[str]:
    """
    Returns paths between given datetimes
    """
    return [line.path for line in data if after <= line.timestamp <= before]


def get_status_from_timestamp(
    data: list[NGINX_Line], after: datetime, before: datetime
) -> list[str]:
    """
    Returns status codes between given datetimes
    """
    return [line.status for line in data if after <= line.timestamp <= before]


def get_paths_from_delta(data: list[NGINX_Line], delta: timedelta) -> list[str]:
    """
    Return paths within a given time delta (between x hours, days, etc.)
    """
    return [
        line.path
        for line in data
        if line.timestamp - delta <= line.timestamp <= line.timestamp + delta
    ]


def get_status_from_delta(data: list[NGINX_Line], delta: timedelta) -> list[str]:
    """
    Return status codes within a given time delta (between x hours, days, etc.)
    """
    return [
        line.status
        for line in data
        if line.timestamp - delta <= line.timestamp <= line.timestamp + delta
    ]
