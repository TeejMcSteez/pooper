import pathlib
import sys
from collections import defaultdict
from typing import Callable, Counter

import matplotlib.pyplot as plt
import numpy
from numpy._core.multiarray import ndarray

from parser import IP_Data, analyze_NGINX_ip_data, parse_nginx_line


def build_global_paths_vector(paths: list[str]):
    """
    Empirical PMF: p(endpoint) = count(endpoint) / N
    """
    endpoint_counts = Counter(paths)
    total = sum(endpoint_counts.values())
    return {ep: count / total for ep, count in endpoint_counts.items()}


def build_global_status_codes_vector(codes: list[str]) -> dict[str, float]:
    """
    Empirical PMF: p(status) = count(status) / N
    """
    code_counts = Counter(codes)
    total = sum(code_counts.values())
    return {ep: count / total for ep, count in code_counts.items()}


def build_global_user_agents_vector(agents: list[str]) -> dict[str, float]:
    """
    Empirical PMF: p(agents) = count(agents) / N
    """
    agent_counts = Counter(agents)
    total = sum(agent_counts.values())
    return {ep: agent / total for ep, agent in agent_counts.items()}


def build_path_vector(
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


def build_status_code_vector(
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


def build_agents_vector(
    ip_agents: list[str], global_freq: dict[str, float]
) -> numpy.ndarray:
    agents = list(global_freq.keys())
    total = len(ip_agents) or 1
    agents_count = Counter(ip_agents)

    agent_freq = {ep: agents_count.get(ep, 0) / total for ep in agents}
    raw_dev = numpy.array([agent_freq[ep] - global_freq[ep] for ep in agents])

    confidence = min(numpy.log1p(total) / numpy.log1p(100), 1.0)
    return raw_dev * confidence


def plot_2d_deviations(x: list[float], y: list[float], x_label: str, y_label: str):
    _, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.4, s=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{x_label} vs {y_label} per IP")
    plt.tight_layout()
    plt.show()


def plot_3d_deviations(
    score_matrix: numpy.ndarray,
    anomaly_scores: numpy.ndarray,
    labels: list[str],
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        score_matrix[:, 0].tolist(),
        score_matrix[:, 1].tolist(),
        score_matrix[:, 2].tolist(),
        c=anomaly_scores.tolist(),
        cmap="hot",
        alpha=0.6,
        s=10,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title("IP anomaly score across all distributions")
    fig.colorbar(sc, label="anomaly score (z-norm)")
    plt.tight_layout()
    plt.show()


# Each Distribution is (extractor, vector_builder, global_freq):
#   extractor:      IP_Data -> list[str]        — pulls the raw values for this IP
#   vector_builder: (list[str], dict) -> ndarray — computes deviation vector
#   global_freq:    dict[str, float]             — the baseline PMF to deviate from
Distribution = tuple[
    Callable[[IP_Data], list[str]],
    Callable[[list[str], dict[str, float]], numpy.ndarray],
    dict[str, float],
]


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
global_path_freq = build_global_paths_vector(paths=paths)
# Used for frequnecy comparison (can be select status codes or compared against all other status codes)
# Comparing against all other status codes derives deviations within the data
# Comparing against specific codes shows deviation from intended codes (or simulated ones)
global_status_freq = build_global_status_codes_vector(codes=statuses)
# Used for frequency comparison (can be select user agents or compared against all other user agents)
# Comparing against all other user agents derives deviations within the data
# Comparing against specific agents shows deviation from intended agents (or simulated ones)
global_agent_freq = build_global_user_agents_vector(agents=agents)

ips = list(ip_data.keys())

distributions: list[Distribution] = [
    (lambda d: list(d.paths.elements()), build_path_vector, global_path_freq),
    (
        lambda d: list(d.status_codes.elements()),
        build_status_code_vector,
        global_status_freq,
    ),
    (lambda d: list(d.user_agents), build_agents_vector, global_agent_freq),
]

raw_scores = build_raw_scores(ip_data, distributions)

score_matrix = numpy.array(raw_scores)  # (n_ips, n_distributions)

z = build_z_score(score_matrix=score_matrix)

anomaly_scores = numpy.linalg.norm(z, axis=1)  # one scalar per IP

plot_2d_deviations(
    score_matrix[:, 0].tolist(),
    score_matrix[:, 1].tolist(),
    x_label="path deviation (L2)",
    y_label="status deviation (L2)",
)

plot_2d_deviations(
    score_matrix[:, 1].tolist(),
    score_matrix[:, 2].tolist(),
    x_label="status deviation (L2)",
    y_label="user agent deviation (L2)",
)

plot_2d_deviations(
    score_matrix[:, 0].tolist(),
    score_matrix[:, 2].tolist(),
    x_label="path deviation (L2)",
    y_label="user agent deviation (L2)",
)

plot_3d_deviations(
    score_matrix,
    anomaly_scores,
    labels=["path deviation", "status deviation", "agent deviation"],
)
