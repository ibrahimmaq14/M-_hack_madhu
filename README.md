# PathSnap

PathSnap reconstructs a directed vehicle route from GPS observations and a trip-specific road graph. The FastAPI server performs candidate generation, classifier inference, Viterbi decoding and private reference scoring. The static browser UI can run on a second laptop.

## Data

The supplied bundle had 30 training trips and 10 labeled student-test trips. The brief describes 80 training trips and 10 separately reserved evaluator trips. Results in this repository use the supplied 40 trips. The reserved partition was unavailable during development. Raw files, derived feature caches, reference routes and map exports are kept local.

Keep filenames zero-padded and preserve the original subdirectories.

    student-bundle/
      train/00000001/00000001.{track,nodes,arcs,route}
      student_test/00000000/00000000.{track,nodes,arcs,route}

Each file is whitespace-separated. Inference reads .track (longitude, latitude, seconds), .nodes (longitude, latitude), and .arcs (directed from/to node row IDs). Scoring reads .route (ordered arc row IDs) independently after prediction. Arc identifiers are local to each trip.

## Reproduce the experiments

Run from the project root with Python 3.11. Replace DATA_ROOT with the directory that contains train and student_test. The default matches the layout of the original local bundle.

    python -m pip install -r requirements-dev.txt
    python -m experiments.prepare --data "DATA_ROOT"
    python -m experiments.train compare --data "DATA_ROOT"
    python -m experiments.train tune --data "DATA_ROOT"
    python -m experiments.routes --data "DATA_ROOT"
    python -m experiments.visuals --data "DATA_ROOT"

Preparation records per-trip counts, graph components, timestamp gaps and candidate coverage. The three CV folds hold out entire trips. Models are selected by grouped CV candidate macro F1. Student-test trips are used for development comparison and decoder selection. Use a fresh .cache when working with a different copy of the data.

| Path | Contents |
| --- | --- |
| pathsnap/ | Parsing, geometry, candidate features, Viterbi, metrics and FastAPI |
| experiments/ | EDA, training, model comparisons, Bayesian TPE, decoder experiments, Folium and report generation |
| results/ | Trip inventory, fold assignments, candidate metrics, search trials and decoder comparisons |
| artifacts/ | Serialized final classifier and its two configuration files |
| ui/ | Static evaluator interface |

## Run the student-test demo on one laptop

Open two shells in the project. The server defaults to the student_test directory in the original bundle. Set PATHSNAP_TRIPS when using a different folder.

    python -m pip install -r requirements.txt

PowerShell terminal 1

    $env:PATHSNAP_TRIPS = 'C:\path\to\student_test'
    $env:OMP_NUM_THREADS = '4'
    python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000

Terminal 2

    python -m http.server 8080 --bind 127.0.0.1 --directory ui

Open http://127.0.0.1:8080. Use GET /health, GET /trips and POST /predict with a JSON body such as {"trip_id":"00000009"}. The OpenAPI console is at http://127.0.0.1:8000/docs.

## Two-laptop evaluator deployment

1. On laptop 1 install the inference dependencies. Set PATHSNAP_TRIPS to the evaluator-provided directory containing the ten trip folders. Set PATHSNAP_REFERENCES to the server-only reference directory when supplied separately, using the same trip-folder layout. It defaults to PATHSNAP_TRIPS. Set PATHSNAP_MODE to evaluator. Keep these files on laptop 1.
2. Set PATHSNAP_UI_ORIGINS on laptop 1 to the UI origin, for example http://192.168.1.21:8080 for laptop 2. Start uvicorn with --host 0.0.0.0 --port 8000. Use a trusted LAN and allow TCP 8000 for laptop 2.
3. On laptop 2 run python -m http.server 8080 --bind 0.0.0.0 --directory ui. Open http://192.168.1.21:8080/?api=http://192.168.1.20:8000, substituting the actual IP addresses. The UI retrieves trip IDs dynamically. Keep the receiving terminal visible to show the inference logs.
4. Verify GET /health reports the evaluator mode and ten trips. Select a trip in the UI. The API returns predicted arc IDs, geometry and aggregate metrics. If reference files have yet to be installed, matching still runs and the score is unavailable.

The offline evaluator UI uses network geometry returned by the API. The separate Folium map is generated for the report with an interactive GPS, context, prediction and reference layer for a student-test trip.

## Reproduce the report evidence

Start Uvicorn and the static UI in separate shells as above. Capture the real server output by appending a PowerShell pipeline to the Uvicorn command.

    New-Item -ItemType Directory -Force local
    python -u -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000 2>&1 | Tee-Object local/api-runtime.log

In a third shell, run the checks and report generator. The browser check uses installed Microsoft Edge by default. Pass --channel chrome for Google Chrome. The interactive Folium map and the OpenAPI console load their library assets over the network.

    python -m experiments.verify --data "DATA_ROOT"
    python -m experiments.report

The verification checks all ten trips against the recorded offline scores and captures the browser UI and live HTTP response. The report is written to output/pdf.

## Experimental evidence and AI assistance

Codex generated the implementation and report and executed the recorded experiments on the supplied development data. The student team is responsible for reproducing the runs, reviewing the analysis in its own words and following the event's AI disclosure rules. Observation-level references are inferred by monotonic alignment to the ordered reference routes. Scores quantify agreement with those derived labels. The student-test results are development estimates, including the decoder grid selection. They do not establish reserved-evaluator performance.
