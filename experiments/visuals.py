"""Generate publication figures and a local Folium comparison map."""

import argparse
import csv
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from branca.element import Element
from threadpoolctl import threadpool_limits

from pathsnap.data import DEFAULT_DATA, load_reference, load_trip, trip_ids
from pathsnap.service import Matcher, visible_roads

SHORT_NAMES = ["Logistic", "Gaussian NB", "Tree", "Forest", "Hist. boosting"]
GREEN = "#176a52"
INK = "#18392e"
GOLD = "#cf863d"
BLUE = "#3760a8"
GRID = "#e4ece6"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
        "xtick.color": INK, "ytick.color": INK, "grid.color": GRID, "savefig.facecolor": "white",
    })


def save(fig, output):
    fig.savefig(output, dpi=170, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def models(results, output):
    rows = read_csv(results / "model_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.0), layout="constrained")
    for ax, metric, title, color in zip(axes, ("cv_macro_f1", "top1", "log_loss"),
                                           ("Grouped CV macro F1", "Student-test quality", "Student-test log loss"),
                                           (GREEN, GREEN, GOLD)):
        x = np.arange(len(rows))
        values = [float(r[metric]) for r in rows]
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        if metric == "top1":
            ax.bar(x - .18, values, .35, color=color, label="Top-1")
            ax.bar(x + .18, [float(r["macro_f1"]) for r in rows], .35, color="#8bbbaa", label="Macro F1")
            ax.legend(frameon=False, loc="lower right", fontsize=8)
        elif metric == "cv_macro_f1":
            ax.bar(x, values, .60, color=color, yerr=[float(r["cv_macro_f1_sd"]) for r in rows],
                   capsize=3, error_kw={"lw": .9})
        else:
            ax.bar(x, values, .60, color=color)
            ax.set_yscale("log")
            ax.set_ylim(.045, 3.4)
        if metric != "log_loss":
            ax.set_ylim(0, 1.04)
        ax.set_xticks(x, SHORT_NAMES, rotation=28, ha="right")
    save(fig, output / "model_comparison.png")

    fig, axes = plt.subplots(1, 5, figsize=(11.1, 2.5), layout="constrained")
    for row, name, ax in zip(rows, SHORT_NAMES, axes):
        counts = np.array([[int(row[k]) for k in ("tn", "fp")],
                           [int(row[k]) for k in ("fn", "tp")]])
        rates = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(rates, vmin=0, vmax=1, cmap="Greens")
        ax.set_title(name, fontweight="bold", pad=8)
        for (i, j), count in np.ndenumerate(counts):
            ax.text(j, i, f"{count:,}\n{rates[i, j]:.1%}", ha="center", va="center",
                    fontsize=7, color="white" if rates[i, j] > .65 else INK)
        ax.set_xticks([0, 1], ["Negative", "Positive"], fontsize=7)
        ax.set_yticks([0, 1], ["Negative", "Positive"] if ax == axes[0] else ["", ""], fontsize=7)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Reference class", fontsize=8)
    save(fig, output / "confusions.png")


def tuning(results, output):
    rows = read_csv(results / "bayesian_trials.csv")
    fig, ax = plt.subplots(figsize=(8.5, 2.55), layout="constrained")
    trials = [int(r["trial"]) for r in rows]
    scores = [float(r["cv_macro_f1"]) for r in rows]
    ax.plot(trials, np.maximum.accumulate(scores), color=GREEN, lw=1.7, label="Best so far")
    ax.scatter(trials[:5], scores[:5], c=GOLD, label="Startup")
    ax.scatter(trials[5:], scores[5:], c="#64a590", label="TPE")
    ax.set(xlabel="Trial", ylabel="Grouped CV macro F1", xticks=trials)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    save(fig, output / "bayesian_search.png")


def routes(results, output):
    all_rows = read_csv(results / "routes_by_trip.csv")
    base = [row for row in all_rows if row["method"] == "Classifier"]
    hmm = [row for row in all_rows if row["method"] == "Viterbi"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.2), layout="constrained", sharey=True)
    for ax, metric, title in zip(axes, ("point_accuracy", "sequence_similarity"),
                                ("Point accuracy", "Route sequence similarity")):
        y = np.arange(len(base))
        a, b = [np.asarray([float(row[metric]) for row in rows]) for rows in (base, hmm)]
        ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color="#a8bbb1", lw=2)
        ax.scatter(a, y, color=GOLD, label="Classifier")
        ax.scatter(b, y, color=GREEN, label="Viterbi")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_yticks(y, [row["trip_id"][-2:] for row in base])
        ax.set_xlim(max(0, min(a.min(), b.min()) - .07), 1.01)
        ax.grid(axis="x")
    axes[0].set_ylabel("Student-test trip ID suffix")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, output / "routes.png")


def eda(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2), layout="constrained")
    trip = load_trip(data / "train", "00000014")
    road_xy = trip.graph.project(np.asarray(visible_roads(trip, limit=5000)))
    axes[0].add_collection(LineCollection(road_xy / 1000, linewidths=.48, color="#a3bbb0", alpha=.78))
    axes[0].scatter(*(trip.xy / 1000).T, s=6, c=GOLD, label="GPS", zorder=3)
    axes[0].set(xlabel="East of trip origin (km)", ylabel="North of trip origin (km)", aspect="equal")
    axes[0].autoscale()
    axes[0].set_title("Training GPS and road geometry · trip 14", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    bins = [1, 1.1, 2.1, 5.1, 10.1, 30.1, 60.1, 300.1, 3000]
    for partition, color in (("train", GREEN), ("student_test", GOLD)):
        gaps = np.concatenate([
            np.diff(load_trip(data / partition, trip_id).track[:, 2])
            for trip_id in trip_ids(data / partition)
        ])
        counts, _ = np.histogram(np.maximum(gaps, 1), bins)
        axes[1].plot(range(len(counts)), counts, "o-", color=color, label=partition.replace("_", " "))
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(8), ["1", "2", "3–5", "6–10", "11–30", "31–60", "61–300", "301+"],
                     rotation=28, ha="right")
    axes[1].set(xlabel="Sampling interval (seconds)", ylabel="Interval count (log scale)")
    axes[1].set_title("Timestamp gaps by partition", loc="left", fontweight="bold")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, output / "eda.png")


def segment_group(map_, name, color, roads, weight, opacity, *, show=True, dash=None):
    group = folium.FeatureGroup(name=name, show=show).add_to(map_)
    lines = np.asarray(roads, dtype=float)[..., ::-1].tolist()
    if lines:
        folium.PolyLine(lines, color=color, weight=weight, opacity=opacity, dash_array=dash).add_to(group)


def folium_map(data, artifacts, output, trip_id):
    partition = data / "student_test"
    trip, candidates, prediction = Matcher(artifacts).match(partition, trip_id)
    reference = load_reference(partition, trip)
    graph = trip.graph
    gps = trip.track[:, :2][:, ::-1]
    map_ = folium.Map(location=np.mean(gps, axis=0), tiles=None, control_scale=True,
                      prefer_canvas=True, attribution_control=False, zoom_snap=.25,
                      zoom_animation=False, fade_animation=False)
    map_.get_root().header.add_child(Element(
        "<style>.leaflet-container{background:#f1f5ef!important}"
        ".legend{position:fixed;bottom:24px;right:18px;z-index:1000;background:white;"
        "padding:9px 13px;border-radius:7px;border:1px solid #c5d1c8;color:#254236;"
        "font:13px system-ui;line-height:1.65}</style>"
    ))
    segment_group(map_, "Road network", "#9eb0a6", visible_roads(trip, candidates, 6500), 1.4, .80)
    segment_group(map_, "Reference route", BLUE, graph.lonlat[graph.arcs[reference]], 6, .63)
    segment_group(map_, "Viterbi route", GREEN,
                  graph.lonlat[graph.arcs[np.concatenate(prediction.route_parts)]], 3, 1, dash="6 3")
    group = folium.FeatureGroup(name="GPS observations").add_to(map_)
    stride = max(1, len(gps) // 175)
    for lat, lon in gps[::stride]:
        folium.CircleMarker((lat, lon), radius=1.5, fill=True, fill_opacity=.6, color=GOLD,
                            opacity=.75, weight=1).add_to(group)
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()], padding=(36, 32))
    map_.get_root().html.add_child(Element(
        f'<div class="legend"><strong>Student test · {trip_id}</strong><br>'
        '<span style="color:#728b7c">━</span> Road network<br>'
        f'<span style="color:{BLUE}">━</span> Reference route<br>'
        f'<span style="color:{GREEN}">┄</span> Viterbi route<br>'
        f'<span style="color:{GOLD}">●</span> GPS observations</div>'
    ))
    map_.save(output / "student_trip_folium.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("local/figures"))
    parser.add_argument("--trip", default="00000091")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    with threadpool_limits(limits=4):
        models(args.results, args.output)
        tuning(args.results, args.output)
        routes(args.results, args.output)
        eda(args.data, args.output)
        folium_map(args.data, args.artifacts, args.output, args.trip)
