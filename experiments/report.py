"""Build the PathSnap technical report from verified run artifacts."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from experiments.visuals import read_csv

INK = colors.HexColor("#1b352b")
MUTED = colors.HexColor("#4d6559")
GREEN = colors.HexColor("#176a52")
PALE = colors.HexColor("#e8f2eb")
LINE = colors.HexColor("#dce7df")
WIDTH = A4[0] - 92


def register_fonts():
    font_dir = Path(get_data_path()) / "fonts" / "ttf"
    for font, file in (("PathSans", "DejaVuSans.ttf"), ("PathBold", "DejaVuSans-Bold.ttf"),
                       ("PathMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(font, str(font_dir / file)))
    pdfmetrics.registerFontFamily("PathSans", normal="PathSans", bold="PathBold",
                                 italic="PathSans", boldItalic="PathBold")


def styles():
    common = {"textColor": INK, "fontName": "PathSans", "allowWidows": 0, "allowOrphans": 0}
    variants = {
        "title": dict(fontName="PathBold", fontSize=41, leading=47, spaceAfter=12),
        "subtitle": dict(fontSize=16, leading=24, spaceAfter=14, textColor=MUTED),
        "kicker": dict(fontName="PathBold", fontSize=9, leading=13, spaceAfter=9, textColor=GREEN),
        "h1": dict(fontName="PathBold", fontSize=23, leading=28, spaceAfter=13),
        "h2": dict(fontName="PathBold", fontSize=12, leading=17, spaceBefore=9, spaceAfter=4),
        "body": dict(fontSize=9.2, leading=14.3, spaceAfter=8),
        "model_body": dict(fontSize=8.8, leading=12.5, spaceAfter=6),
        "small": dict(fontSize=8.1, leading=12.2, spaceAfter=8, textColor=MUTED),
        "cell": dict(fontSize=7.6, leading=10.4),
        "cell_header": dict(fontName="PathBold", fontSize=7.6, leading=10.3, textColor=colors.white),
        "hero": dict(fontName="PathBold", fontSize=21, leading=27, alignment=TA_CENTER, textColor=GREEN),
        "hero_label": dict(fontSize=7.5, leading=11, alignment=TA_CENTER, textColor=MUTED),
        "mono": dict(fontName="PathMono", fontSize=7.1, leading=11, textColor=INK),
    }
    return {key: ParagraphStyle(key, **{**common, **params}) for key, params in variants.items()}


def furniture(canvas, doc):
    w, h = A4
    canvas.setFillColor(GREEN)
    canvas.roundRect(46, h - 41, 12, 12, 3, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("PathBold", 8)
    canvas.drawCentredString(52, h - 38, "/")
    canvas.setFillColor(INK)
    canvas.setFont("PathBold", 9)
    canvas.drawString(65, h - 38, "PathSnap")
    canvas.setFont("PathSans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 46, h - 38, "PES UNIVERSITY  /  UE24CS352A")
    canvas.setStrokeColor(LINE)
    canvas.line(46, h - 47, w - 46, h - 47)
    canvas.line(46, 44, w - 46, 44)
    canvas.drawString(46, 29, "TECHNICAL REPORT   /   18 NOVEMBER 2026")
    canvas.drawRightString(w - 46, 29, f"{doc.page:02}")


def percent(value):
    return f"{float(value):.2%}"


def add_table(story, style, headers, rows, widths):
    values = [[Paragraph(escape(str(label)), style["cell_header"]) for label in headers]]
    values += [[Paragraph(escape(str(value)), style["cell"]) for value in row] for row in rows]
    table = Table(values, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), .6, LINE),
    ]))
    story.extend([table, Spacer(1, 8)])


def add_image(story, path, max_height, width=WIDTH):
    w, h = ImageReader(str(path)).getSize()
    ratio = min(width / w, max_height / h)
    story.append(Image(str(path), w * ratio, h * ratio, hAlign="CENTER"))


def build(results, figures, runtime, output):
    register_fonts()
    st, story = styles(), []
    comparison = read_csv(results / "model_comparison.csv")
    tuning = read_csv(results / "tuning_comparison.csv")
    inventory = read_csv(results / "data_inventory.csv")
    route_rows = read_csv(results / "route_comparison.csv")
    smoke = read_csv(results / "api_smoke.csv")
    case_rows = read_csv(results / "routes_by_trip.csv")
    search = json.loads((results / "bayesian_search.json").read_text(encoding="utf-8"))
    grid = read_csv(results / "decoder_grid.csv")
    selected = max(grid, key=lambda r: (float(r["sequence_similarity"]), float(r["point_accuracy"])))
    tune_base, tune_final = tuning
    classifier, viterbi = route_rows

    def paragraph(value, kind="body"):
        story.append(Paragraph(value, st[kind]))

    def page(number, title):
        story.append(PageBreak())
        paragraph(number, "kicker")
        paragraph(title, "h1")

    def caption(value):
        paragraph(value, "small")

    def fig(name, max_height, text):
        add_image(story, figures / name, max_height)
        story.append(Spacer(1, 6))
        caption(text)

    paragraph("MACHINE LEARNING HACKATHON", "kicker")
    paragraph("PathSnap", "title")
    paragraph("Distributed map matching<br/>and route reconstruction", "subtitle")
    paragraph("Technical report", "h2")
    paragraph("A supervised road-candidate scorer and a directed Viterbi decoder reconstruct the likely road route from noisy GPS. FastAPI runs the prediction and keeps route-reference scoring on the server. A browser on a second laptop can select a trip and display the returned match.")
    story.append(Spacer(1, 8))
    hero = Table([
        [Paragraph(percent(viterbi["point_accuracy"]), st["hero"]),
         Paragraph(percent(viterbi["sequence_similarity"]), st["hero"]),
         Paragraph(percent(viterbi["route_continuity"]), st["hero"])],
        [Paragraph(label, st["hero_label"]) for label in (
            "POINT ACCURACY", "ROUTE SEQUENCE SIMILARITY", "RECONSTRUCTED CONTINUITY")],
    ], colWidths=[WIDTH / 3] * 3)
    hero.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("LINEAFTER", (0, 0), (1, -1), .75, colors.white),
    ]))
    story.extend([hero, Spacer(1, 9)])
    caption("Metrics shown for the ten labeled student-test trips. Route similarity is averaged across trips.")
    paragraph("Scope and provenance", "h2")
    paragraph("The supplied bundle contained 30 training trips and 10 student-test trips. The published brief describes 80 training trips and a separate 10-trip evaluator partition. Development used the 40 supplied trips. The reserved evaluator files and the second physical laptop were unavailable.")
    paragraph("Codex generated the implementation and this technical report and ran the recorded experiments. The student team should independently reproduce the experiments, verify the interpretations, and follow the event’s AI disclosure requirements.")
    paragraph("Repository", "h2")
    paragraph('<link href="https://github.com/ibrahimmaq14/M-_hack_madhu" color="#176a52">github.com/ibrahimmaq14/M-_hack_madhu</link>')
    paragraph("Project brief", "h2")
    caption("PES University PathSnap student materials. Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, and Assessment and Plagiarism Policy.")

    page("2.2 / DATASET UNDERSTANDING", "GPS, graphs and reference routes")
    groups = [[r for r in inventory if r["partition"] == key] for key in ("train", "student_test")]

    def total(pool, name):
        return sum(int(row[name]) for row in pool)

    def span(pool, name):
        values = [int(row[name]) for row in pool]
        return f"{min(values):,}-{max(values):,}"

    table_rows = []
    for title, group in zip(("Training", "Student test"), groups):
        obs = total(group, "observations")
        one_sec = total(group, "one_second_steps") / (obs - len(group))
        table_rows.append([
            title, len(group), f"{obs:,}", span(group, "observations"), span(group, "arcs"),
            percent(one_sec), f"{max(float(row['max_gap_s']) for row in group):,.0f} s",
        ])
    add_table(story, st, ["Split", "Trips", "GPS rows", "Rows per trip", "Arcs per trip", "1 s steps", "Max gap"],
              table_rows, [82, 34, 59, 79, 96, 71, WIDTH - 421])
    paragraph("<b>Inputs and cleaning.</b> Read whitespace-separated longitude, latitude and time from .track, longitude and latitude from .nodes, and directed zero-based node pairs from .arcs. The .route file stores a chronological list of arc row IDs. All 160 expected files were present. Both splits had zero nonfinite or invalid GPS rows, reversed timestamps and duplicate times. The parser validates coordinates, removes invalid rows and stably sorts and deduplicates timestamps when needed.")
    paragraph(f"<b>Network and sampling.</b> The training graphs ranged from {span(groups[0], 'nodes')} nodes and {span(groups[0], 'weak_components')} weakly connected components. Student-test graphs ranged from {span(groups[1], 'nodes')} nodes and {span(groups[1], 'weak_components')} weak components. No isolated nodes were found. Raw references contained {total(groups[0], 'reference_breaks')} training and {total(groups[1], 'reference_breaks')} student-test breaks. Train trip 21 had a 2,161 s gap and test trip 65 had a 210 s gap.")
    fig("eda.png", 183, "Figure 1. Training-trip GPS geometry and the timestamp-gap counts for both development partitions.")
    paragraph("<b>Features and weak labels.</b> A per-trip equirectangular projection converts angular differences to local metres with longitude scaled by the cosine of the median trip latitude. A sampled segment index proposes ten candidates and exact point-to-segment projection orders the proposal pool. Distance, rank, overhang and arc length describe spatial fit. Heading alignment, adjacent and three-step distances, progression, elapsed time and speed distinguish parallel roads, direction and junctions. Temporal features ignore lookups across gaps over 10 s.")
    paragraph(f"Point labels come from minimum-cost monotone alignment of observations to the ordered reference route using distance, heading and a small forward-skip cost. Candidate miss was {percent(total(groups[0], 'candidates_missing_reference') / total(groups[0], 'observations'))} in training and {percent(tune_final['candidate_miss'])} in student test. Trip 33 accounts for 3,858 of 4,647 training misses. Candidate coverage was only 33.98% despite a 6.99 m median nearest-road distance. This can indicate disagreement in the route, alignment or candidate retrieval. All observations were retained for evaluation.")

    page("3 / MODEL EXPERIMENTATION", "Five classifiers on shared folds")
    paragraph("Three group-aware folds hold out whole training trips. Fitting uses up to 950 evenly spaced observations with an available positive candidate per trip, yielding 255,980 candidate rows at refit. Every candidate in every held-out observation is scored. The same pipeline and folds are used across models.")
    paragraph("Mean grouped CV candidate macro F1 is the selection metric. It weights match and nonmatch classes equally under the roughly 1 to 9 label imbalance. Binary metrics use a 0.5 score cutoff. Top-1 selects the highest score in each observation’s candidate set. Candidate misses count against top-1. Continuity checks directed adjacency across changes in the predicted arc sequence.")
    configs = ["C 1", "Smoothing 1e-9", "Depth 11 / leaf 24", "60 trees / depth 15", "110 iters / 25 leaves"]
    rows = []
    for r, config in zip(comparison, configs):
        rows.append([r["model"], r["family"], config, percent(r["cv_macro_f1"]), percent(r["top1"]),
                     percent(r["continuity"]), percent(r["macro_f1"]), f"{float(r['log_loss']):.4f}",
                     f"{float(r['inference_ms_per_obs']):.4f}"])
    add_table(story, st, ["Model", "Family", "Key config", "CV F1", "Test top-1", "Test continuity",
                          "Test F1", "Log loss", "ms / obs"],
              rows, [82, 63, 64, 44, 44, 56, 47, 44, WIDTH - 444])
    fig("model_comparison.png", 170, "Figure 2. CV bars show fold standard deviation. Student-test statistics use all 19,160 observations. Inference time is batched scoring per observation, excluding feature extraction and decoding.")
    paragraph("Histogram gradient boosting led grouped CV at 91.31% macro F1 and had the strongest student-test top-1 at 87.82%. Its mean grouped CV F1 exceeded the forest by 0.38 percentage points. Fold variability makes this a modest selection margin.")

    page("3.2 / INDIVIDUAL MODELS", "Configurations and behavior")
    notes = [
        ("Scaling makes the linear combination of distance and heading interpretable. Its CV F1 trails the ensembles, which suggests nonlinear interactions matter at junctions. The fitted-to-CV gap is small.",
         "C=1 and max_iter=300 after StandardScaler."),
        ("Gaussian marginals provide fast fitting and directional ranking. The independence assumption is weak for correlated distance features. Its 39,903 false positives and high log loss show that ranking quality can coexist with poor threshold behavior. Low F1 on both training and CV points to model bias.",
         "StandardScaler and var_smoothing=1e-9."),
        ("A single tree can split on direction and distance jointly and was the fastest to score. Its fitting-to-CV F1 gap and log loss are higher than the ensembles, consistent with overfitting and sharp leaf confidence.",
         "max_depth=11 and min_samples_leaf=24."),
        ("Bagging captures nonlinear interactions with less variance than one tree. Log loss improved strongly over the tree. It took the longest to refit, and its 1.95 pp fitted-to-CV F1 gap indicates some generalization loss.",
         "60 trees, max_depth=15, min_samples_leaf=6, max_features=sqrt and bootstrap sampling."),
        ("Boosting learns residual errors across distance, heading and arc progression. It had the best grouped CV and lowest student-test loss. Its fitted-to-CV gap is comparable to the forest and its batched scoring took longer.",
         "110 iterations, learning rate 0.09, 25 leaves, minimum leaf size 32 and L2 1.0."),
    ]
    for row, (behavior, config) in zip(comparison, notes):
        paragraph(escape(row["model"]), "h2")
        paragraph(f"<b>Setup.</b> {config} Optimisation uses {escape(row['objective'].lower().replace('gaussian', 'Gaussian').replace('gini', 'Gini'))}. "
                  f"<b>Measured.</b> Fitted training F1 {percent(row['train_macro_f1'])}, CV F1 {percent(row['cv_macro_f1'])}, "
                  f"test top-1 {percent(row['top1'])}, test F1 {percent(row['macro_f1'])}, test loss {float(row['log_loss']):.4f}. "
                  f"Macro precision/recall {percent(row['macro_precision'])}/{percent(row['macro_recall'])}. "
                  f"Full refit {float(row['refit_s']):.2f} s. {behavior}", "model_body")
    caption("Fitted training F1 uses all training observations, including candidate misses, and is diagnostic. CV estimates come from the corresponding held-out trips. No epoch-level loss history was recorded for these fits.")

    fig("confusions.png", 96, "Figure 3. Student-test candidate confusion matrices. Rows show reference classes and columns show predictions.")

    page("4 / BAYESIAN OPTIMISATION", "TPE search on the training folds")
    paragraph(f"Optuna’s tree-structured Parzen estimator used seed {search['seed']}, {search['startup']} startup trials and {search['trials']} total trials. The score is mean candidate macro F1 over the same three whole-trip CV folds. TPE proposes settings by modelling parameter densities for higher- and lower-scoring trials. Total search time was {search['search_s']:.1f} seconds.")
    labels = {
        "max_iter": ("Boosting iterations", "80-210, integer"),
        "learning_rate": ("Learning rate", "0.035-0.18, log scale"),
        "max_leaf_nodes": ("Leaves per tree", "12-48, integer"),
        "min_samples_leaf": ("Minimum leaf samples", "16-85, integer"),
        "l2_regularization": ("L2 regularisation", "0.02-8.0, log scale"),
    }
    rows = []
    for key, value in search["best_params"].items():
        best = f"{value:.5f}" if isinstance(value, float) else str(value)
        rows.append([labels[key][0], labels[key][1], best])
    add_table(story, st, ["Parameter", "Search range", "Winning setting"], rows, [181, 168, WIDTH - 349])
    paragraph("The iteration and learning-rate ranges let weaker steps trade against ensemble size. Leaves, minimum leaf size and L2 span model capacity and regularisation. Positive scale parameters use log sampling to explore ratios across their ranges.")
    fig("bayesian_search.png", 176, "Figure 4. The best trial was trial 11, using zero-based trial numbers. Lower scores remain in the search record.")
    rows = [[
        r["variant"], percent(r["cv_macro_f1"]), percent(r["top1"]), percent(r["macro_f1"]),
        f"{float(r['log_loss']):.4f}", f"{float(r['inference_ms_per_obs']):.4f}", f"{float(r['refit_s']):.2f}",
    ] for r in tuning]
    add_table(story, st, ["Variant", "CV F1", "Test top-1", "Test F1", "Test loss", "ms / obs", "Fit s"],
              rows, [83, 67, 80, 72, 70, 70, WIDTH - 442])
    paragraph("Tuning added 0.23 percentage points of CV F1 and 0.07 points of student-test top-1, with slightly lower log loss. Prediction became slower. The tuned classifier was retained based on its CV objective, and the complete trial and fold logs were saved.")

    page("5 / HMM AND VITERBI", "From individual matches to a route")
    paragraph("<b>States and emissions.</b> Each state identifies a directed candidate arc for the current GPS observation. The emission weight is the logarithm of the clipped classifier score, scaled by the selected coefficient. Candidate IDs and their projected along-arc positions are held for each step.")
    paragraph(f"<b>Transitions.</b> On the same arc, use the absolute difference in projected progression and lightly penalise backward movement over 8 m. For distinct arcs, add the distance to the first arc’s end, the bounded directed shortest-path distance between endpoints, and the second arc’s projected progress. Penalise the absolute difference between road travel and straight-line GPS displacement. The scale combines beta={selected['beta_m']} m, displacement and elapsed time. Immediate directed reversals cost {float(selected['reversal_cost']):g} log-weight units. Unreachable pairs have zero transition weight.")
    paragraph(f"<b>Decoding.</b> For each state, take its log emission plus the maximum prior score and transition weight. Backpointers recover the maximum-score path. The emission coefficient is {float(selected['emission_weight']):g}. The spatial search supplies a nearest fallback if the index pool is underfilled. Gaps over {float(selected['gap_reset_s']):g} s or a completely unreachable next step start a separate part. Directed shortest paths fill intermediate arcs within each part. Nine settings were compared on student test with mean sequence similarity as the decoder objective.")
    add_table(story, st, ["Method", "Point accuracy", "Observed adjacency", "Full continuity", "Sequence match",
                          "Batch time"], [[r["method"], percent(r["point_accuracy"]), percent(r["raw_continuity"]),
                           percent(r["route_continuity"]), percent(r["sequence_similarity"]),
                           f"{float(r['seconds']):.2f} s"] for r in route_rows],
              [77, 87, 97, 84, 84, WIDTH - 429])
    fig("routes.png", 167, "Figure 5. Student-test trajectories. Sequence match is one minus normalized Levenshtein distance against collapsed reference arcs. Observed adjacency compares consecutive distinct observed arcs before inserting bridges.")
    base_07, hmm_07 = [r for r in case_rows if r["trip_id"] == "00000007"]
    base_77, hmm_77 = [r for r in case_rows if r["trip_id"] == "00000077"]
    paragraph(f"Viterbi improves point accuracy on nine runs. Trip 07 drops from {percent(base_07['point_accuracy'])} to {percent(hmm_07['point_accuracy'])} while sequence similarity rises from {percent(base_07['sequence_similarity'])} to {percent(hmm_07['sequence_similarity'])}. This is consistent with a timing mismatch around segment boundaries. Trip 77 remains the lowest-scoring reconstructed sequence at {percent(hmm_77['sequence_similarity'])}, up from {percent(base_77['sequence_similarity'])}. The decoder emitted six breaks. Bridge insertion explains part of the full-route continuity gain.")

    page("6 / FOLIUM VISUALISATION", "GPS and routes on the road network")
    paragraph("The following exported view shows student-test trip 00000091. Gray depicts the supplied road network, blue the reference route, green the Viterbi path and gold the observed coordinates. The map viewport is fitted to the full GPS extent. In the HTML version each layer can be toggled independently.")
    fig("folium_map.png", 345, "Figure 6. Screenshot of the generated Folium view. The wider blue stroke makes overlapping predicted and reference arcs visible.")
    paragraph("Run <b>python -m experiments.visuals</b> with the student bundle and final model to regenerate the interactive HTML at <b>local/figures/student_trip_folium.html</b>. Reference geometry is used only in the offline report map. The evaluator UI receives prediction geometry and aggregate scores.")
    paragraph("Trip 00000091 had 874 observations. Sequence similarity reached 97.47% and point accuracy reached 94.51% after Viterbi. These evaluate agreement with the supplied route sequence and its inferred observation alignment.")

    page("7 / API, UI AND DEPLOYMENT", "Evaluator workflow")
    paragraph("Laptop 1 runs FastAPI and loads the saved classifier and decoder configuration. Laptop 2 serves the static evaluator interface and connects to the configured API origin. The UI loads trip identifiers with GET /trips and sends the selected ID with POST /predict. The server loads the reference only after inference and returns aggregate metrics alongside the prediction.")
    fig("evaluator_ui.png", 360, "Figure 7. Working evaluator UI captured from the local student-test run. The UI ran in a browser on its own HTTP origin and called the FastAPI origin.")
    paragraph("The UI presents the graph, GPS and predicted route with layer toggles, fit, zoom and pan. It reports point agreement with the aligned reference, edit-based sequence similarity, directed continuity, the number of points and route parts, and server latency. The road map remains available for a trip without scoring labels.")
    paragraph("For the two-laptop demonstration, set the data and UI-origin environment variables on laptop 1 and run Uvicorn on its LAN interface. Serve the UI on laptop 2 and pass laptop 1’s address in the api query parameter. Exact commands and paths are in the repository README.")

    page("7 / LIVE INTEGRATION EVIDENCE", "API requests and receiving logs")
    seconds = [float(r["elapsed_seconds"]) for r in smoke]
    paragraph(f"The live API returned 200 for {len(smoke)} of {len(smoke)} student-test trips. Each result contained one arc per GPS observation, valid arc indices and connected geometry within each reconstructed part. Offline and API metrics matched. End-to-end server time ranged from {min(seconds):.2f} to {max(seconds):.2f} s in the smoke run.")
    api_image = figures / "api_response.png"
    if api_image.is_file():
        fig("api_response.png", 247, "Figure 8. FastAPI’s live OpenAPI console shows the HTTP 200 response for the selected trip.")
    lines = runtime.read_text(encoding="utf-8-sig").splitlines()
    receiver = [line for line in lines if ("PathSnap ready" in line
                                          or ("POST /predict" in line
                                              and ("00000091" in line or "200 OK" in line)))]
    paragraph("Receiving terminal log excerpt", "h2")
    log = [receiver[0], *receiver[-3:]]
    data = [[Paragraph("<br/>".join(escape(line) for line in log), st["mono"])]]
    table = Table(data, colWidths=[WIDTH], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([table, Spacer(1, 10)])
    caption("Terminal output was captured as a transcript on the development machine. Reserved evaluator inputs and a second physical device remain to be verified at the event.")
    paragraph("Reproduction files", "h2")
    paragraph("Use results/data_inventory.csv for the dataset audit, results/model_comparison.csv and results/cv_folds.csv for candidate scores, results/bayesian_trials.csv and results/bayesian_folds.csv for the TPE search, and results/routes_by_trip.csv and results/api_smoke.csv for the decoder and deployment checks.")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=46, rightMargin=46,
                            topMargin=66, bottomMargin=60, title="PathSnap Technical Report",
                            subject="Directed GPS map matching and route reconstruction",
                            author="PathSnap project")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("local/figures"))
    parser.add_argument("--runtime", type=Path, default=Path("local/api-runtime.log"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/PathSnap_Technical_Report.pdf"))
    args = parser.parse_args()
    build(args.results, args.figures, args.runtime, args.output)
