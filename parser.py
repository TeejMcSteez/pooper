import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Counter, NamedTuple

# NGINX uses Apache HTTP Server Version 2.4 log format (I think)
# https://httpd.apache.org/docs/2.4/logs.html
pattern = re.compile(
    r"^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "
    r'"(?:(?P<method>\S+) (?P<path>\S+) [^"]+|[^"]*)" '
    r'(?P<status>\d+) (?P<bytes>\d+) "[^"]*" "(?P<useragent>[^"]*)"'
)


class NGINX_Line(NamedTuple):
    ip: str
    timestamp: datetime
    method: str
    path: str
    status: str
    bytes: str
    useragent: str


@dataclass
class IP_Data:
    requests: int = 0
    errors: int = 0
    paths: Counter[str] = field(default_factory=Counter)
    user_agents: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    status_codes: Counter[str] = field(default_factory=Counter)


# Parse line using regex pattern
def parse_nginx_line(line: str):
    """
    Parse line using regex pattern
    """
    if m := pattern.match(line):
        return NGINX_Line(
            ip=m["ip"],
            timestamp=datetime.strptime(m["timestamp"], "%d/%b/%Y:%H:%M:%S %z"),
            method=m["method"] or "",
            path=m["path"] or "",
            status=m["status"],
            bytes=m["bytes"],
            useragent=m["useragent"],
        )


def analyze_NGINX_ip_data(lines: list[str]) -> defaultdict[str, IP_Data]:
    """
    Analyzed log data returning the IP and used data
    """
    ip_data = defaultdict(IP_Data)

    for line in lines:
        entry = parse_nginx_line(line)
        if not entry:
            continue
        ip = entry.ip
        ip_data[ip].requests += 1
        ip_data[ip].paths[entry.path] += 1
        ip_data[ip].user_agents.add(entry.useragent)
        ip_data[ip].methods.add(entry.method)
        ip_data[ip].status_codes[entry.status] += 1

        if int(entry.status) >= 400:
            ip_data[ip].errors += 1
    return ip_data
