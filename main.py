import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Counter, NamedTuple

import matplotlib.pyplot as plt
import numpy

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


# Analyzed log data returning the IP and used data
def analyze_NGINX_ip_data(lines: list[str]) -> defaultdict[str, IP_Data]:
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


# Turns given path list into a frequency set
def vectorize_global_paths(paths: list[str]):
    endpoint_counts = Counter(paths)
    total = sum(endpoint_counts.values())
    return {ep: count / total for ep, count in endpoint_counts.items()}


# Turns given status codes into a frequency set
def vectorize_global_status_codes(codes: list[str]) -> dict[str, float]:
    code_counts = Counter(codes)
    total = sum(code_counts.values())
    return {ep: count / total for ep, count in code_counts.items()}


def get_paths_from_timestamp(
    data: list[NGINX_Line], after: datetime, before: datetime
) -> list[str]:
    return [line.path for line in data if after <= line.timestamp <= before]


def get_status_from_timestamp(
    data: list[NGINX_Line], after: datetime, before: datetime
) -> list[str]:
    return [line.status for line in data if after <= line.timestamp <= before]


# Builds vector from given client paths and all clients path frequency
def build_ip_vector_from_paths(
    ip_paths: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    vocab = list(global_freq.keys())
    total = len(ip_paths) or 1
    ip_counts = Counter(ip_paths)
    ip_freq = {ep: ip_counts.get(ep, 0) / total for ep in vocab}

    raw_dev = numpy.array([ip_freq[ep] - global_freq[ep] for ep in vocab])

    # confidence weight — shrink deviation toward zero for low volume IPs
    confidence = numpy.log1p(total) / numpy.log1p(
        100
    )  # normalize against reasonable max (100%)
    confidence = min(confidence, 1.0)

    return raw_dev * confidence


# Builds vector from given client status codes and all client status code frequencies
def build_ip_vector_from_status_codes(
    ip_codes: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    pass
    codes = list(global_freq.keys())
    total = len(ip_codes) or 1
    status_counts = Counter(ip_codes)
    code_freq = {ep: status_counts.get(ep, 0) / total for ep in codes}
    return numpy.array([code_freq[ep] - global_freq[ep] for ep in codes])


# filepath check
if len(sys.argv) < 2:
    print("specify filename")
    exit(1)
filepath = sys.argv[1]
lines = pathlib.Path(filepath).read_text().splitlines()
# Grabs all unique IP's from each line
unique_ips = {p.ip for line in lines if (p := parse_nginx_line(line))}
# Grabs all unqiue paths in each line
unique_paths = {p.path for line in lines if (p := parse_nginx_line(line))}

# Gets IP data from file lines
ip_data = analyze_NGINX_ip_data(lines=lines)
# Grabs all unique status codes
statuses = [c for data in ip_data.values() for c in data.status_codes]
# Collect all paths from each IP
paths = [path for data in ip_data.values() for path in data.paths.elements()]
# Used for frequency comparison (can be select endpoint or compared against all other IP's)
global_path_freq = vectorize_global_paths(paths=paths)
global_status_freq = vectorize_global_status_codes(codes=statuses)

path_deviations = []
status_deviations = []
for entry, data in ip_data.items():
    path_vec = build_ip_vector_from_paths(
        list(data.paths.elements()), global_freq=global_path_freq
    )
    status_vec = build_ip_vector_from_status_codes(
        list(data.status_codes.elements()), global_freq=global_status_freq
    )
    path_deviations.append(float(numpy.linalg.norm(path_vec)))
    status_deviations.append(float(numpy.linalg.norm(status_vec)))

fig, ax = plt.subplots()
ax.scatter(path_deviations, status_deviations, alpha=0.4, s=10)
ax.set_xlabel("path deviation (L2)")
ax.set_ylabel("status code deviation (L2)")
ax.set_title("Path vs status code deviation per IP")
plt.tight_layout()
plt.show()

ips = list(ip_data.keys())

for i, (ip, path_dev, status_dev) in enumerate(
    zip(ips, path_deviations, status_deviations)
):
    if path_dev > 0.5 or status_dev > 1.1:
        print(f"{ip:20s}  path={path_dev:.3f}  status={status_dev:.3f}")
        print(f"  paths:   {dict(ip_data[ip].paths)}")
        print(f"  codes:   {dict(ip_data[ip].status_codes)}")
        print(f"  agents:  {ip_data[ip].user_agents}")
        print()
