import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Counter, NamedTuple

import matplotlib.pyplot as plt
import numpy
from numpy._core.multiarray import ndarray

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


def vectorize_global_paths(paths: list[str]):
    """
    Empirical PMF: p(endpoint) = count(endpoint) / N
    """
    endpoint_counts = Counter(paths)
    total = sum(endpoint_counts.values())
    return {ep: count / total for ep, count in endpoint_counts.items()}


def vectorize_global_status_codes(codes: list[str]) -> dict[str, float]:
    """
    Empirical PMF: p(status) = count(status) / N
    """
    code_counts = Counter(codes)
    total = sum(code_counts.values())
    return {ep: count / total for ep, count in code_counts.items()}


def vectorize_global_user_agents(agents: list[str]) -> dict[str, float]:
    """
    Empirical PMF: p(agents) = count(agents) / N
    """
    agent_counts = Counter(agents)
    total = sum(agent_counts.values())
    return {ep: agent / total for ep, agent in agent_counts.items()}


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


def build_ip_vector_from_paths(
    ip_paths: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    """
    deviation[i] = p_ip(i) - p_global(i), scaled by log confidence weight

    confidence = log(1+n) / log(1+100), clamped to [0,1] — shrinks sparse IPs toward zero

    final vector: dev * confidence  (low-traffic IPs get pulled toward origin)
    """
    vocab = list(global_freq.keys())
    total = len(ip_paths) or 1
    ip_counts = Counter(ip_paths)
    ip_freq = {ep: ip_counts.get(ep, 0) / total for ep in vocab}

    # dev[i] = p_ip(i) - p_global(i)
    raw_dev = numpy.array([ip_freq[ep] - global_freq[ep] for ep in vocab])

    # log(1+n) / log(1+100) in [0,1]
    confidence = min(numpy.log1p(total) / numpy.log1p(100), 1.0)

    return raw_dev * confidence


def build_ip_vector_from_status_codes(
    ip_codes: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    """
    deviation[i] = p_ip(i) - p_global(i), no confidence weighting
    """
    codes = list(global_freq.keys())
    total = len(ip_codes) or 1
    status_counts = Counter(ip_codes)
    # dev[i] = p_ip(i) - p_global(i)
    code_freq = {ep: status_counts.get(ep, 0) / total for ep in codes}
    return numpy.array([code_freq[ep] - global_freq[ep] for ep in codes])


def build_ip_vector_from_agents(
    ip_agents: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    agents = list(global_freq.keys())
    total = len(ip_agents) or 1
    agents_count = Counter(ip_agents)

    agent_freq = {ep: agents_count.get(ep, 0) / total for ep in agents}
    return numpy.array([agent_freq[ep] - global_freq[ep] for ep in agents])


def plot_path_status_deviations(path_dev, status_dev):
    """
    Display 2D standard deviations (path_deviation x, status_deviation y)
    """
    fig, ax = plt.subplots()
    ax.scatter(path_dev, status_dev, alpha=0.4, s=10)
    ax.set_xlabel("path deviation (L2)")
    ax.set_ylabel("status code deviation (L2)")
    ax.set_title("Path vs status code deviation per IP")
    plt.tight_layout()
    plt.show()


def report(ip_data: defaultdict[str, IP_Data]):
    """
    Prints IP data
    """
    ips = list(ip_data.keys())

    for i, (ip, path_dev, status_dev) in enumerate(
        zip(ips, path_deviations, status_deviations)
    ):
        print(f"{ip:20s}  path={path_dev:.3f}  status={status_dev:.3f}")
        print(f"  paths:   {dict(ip_data[ip].paths)}")
        print(f"  codes:   {dict(ip_data[ip].status_codes)}")
        print(f"  agents:  {ip_data[ip].user_agents}")
        print()


# TODO: Update to be more general, build raw scores from path and status vectors based on what is passed not requiring global vars
def build_raw_scores(data: defaultdict[str, IP_Data]):
    raw_scores = []  # (n_ips, 8): global_path, global_status, hour_path, hour_status, day_path, day_status, week_path, week_status

    for entry, ip_data in data.items():
        global_path_vec = build_ip_vector_from_paths(
            list(ip_data.paths.elements()), global_path_freq
        )
        global_status_vec = build_ip_vector_from_status_codes(
            list(ip_data.status_codes.elements()), global_status_freq
        )

def build_raw_scores(
    data: defaultdict[str, IP_Data],
    distributions: list[Distribution],
) -> list[list[float]]:
    """Returns (n_ips, n_distributions) matrix of L2 deviation norms."""
    raw_scores = []
    for _, ip_data in data.items():
        row = [
            float(numpy.linalg.norm(build_vec(extract(ip_data), global_freq)))
            for extract, build_vec, global_freq in distributions
        ]
        raw_scores.append(row)
    return raw_scores


def build_z_score(score_matrix: ndarray):
    """
    Calculate z-score from score matrix
    """
    return (score_matrix - score_matrix.mean(axis=0)) / (
        score_matrix.std(axis=0) + 1e-9
    )


# filepath check
if len(sys.argv) < 2:
    print("specify filename")
    exit(1)
filepath = sys.argv[1]
lines = pathlib.Path(filepath).read_text().splitlines()
# Parse all valid lines into NGINX_Line
parsed_lines = [p for line in lines if (p := parse_nginx_line(line))]
# Gets IP data from file lines
ip_data = analyze_NGINX_ip_data(lines=lines)
# Grabs all unique status codes
statuses = [c for data in ip_data.values() for c in data.status_codes.elements()]
# Collect all paths from each IP
paths = [path for data in ip_data.values() for path in data.paths.elements()]
# Collects all user agents from each IP
agents = [agents for data in ip_data.values() for agents in data.user_agents]
# Used for frequency comparison (can be select endpoints or compared against all other IP's)
# Comparing against all other paths derives deviations within the data
# Comparing against specific paths shows deviation from intended paths (or simulated ones)
global_path_freq = vectorize_global_paths(paths=paths)
# Used for frequnecy comparison (can be select status codes or compared against all other status codes)
# Comparing against all other status codes derives deviations within the data
# Comparing against specific codes shows deviation from intended codes (or simulated ones)
global_status_freq = vectorize_global_status_codes(codes=statuses)
# Used for frequency comparison (can be select user agents or compared against all other user agents)
# Comparing against all other user agents derives deviations within the data
# Comparing against specific agents shows deviation from intended agents (or simulated ones)
global_agent_freq = vectorize_global_user_agents(agents=agents)

ips = list(ip_data.keys())

distributions: list[Distribution] = [
    (lambda d: list(d.paths.elements()), build_ip_vector_from_paths, global_path_freq),
    (lambda d: list(d.status_codes.elements()), build_ip_vector_from_status_codes, global_status_freq),
    (lambda d: list(d.user_agents), build_ip_vector_from_agents, global_agent_freq),
]

raw_scores = build_raw_scores(ip_data, distributions)

score_matrix = numpy.array(raw_scores)  # (n_ips, n_distributions)
z = build_z_score(score_matrix=score_matrix)
anomaly_scores = numpy.linalg.norm(z, axis=1)  # one scalar per IP

path_deviations = score_matrix[:, 0].tolist()
status_deviations = score_matrix[:, 1].tolist()

plot_path_status_deviations(path_deviations, status_deviations)
report(ip_data)
