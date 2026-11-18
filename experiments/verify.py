"""Verify live prediction for every student-demo trip and capture browser evidence."""

import argparse
import csv
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

from experiments.train import write_rows
from pathsnap.data import DEFAULT_DATA, load_trip


def verify(api, data):
    records = []
    with httpx.Client(base_url=api, timeout=30) as client:
        health = client.get("/health").raise_for_status().json()
        index = client.get("/trips").raise_for_status().json()
        assert health["mode"] == index["mode"] == "student_demo"
        assert health["trip_count"] == len(index["trips"]) == 10
        assert client.post("/predict", json={"trip_id": "../api"}).status_code == 422
        assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
        for trip_id in index["trips"]:
            response = client.post("/predict", json={"trip_id": trip_id}).raise_for_status()
            result = response.json()
            trip = load_trip(data / "student_test", trip_id)
            assert result["trip_id"] == trip_id
            assert len(result["predicted_arcs"]) == result["observations"] == len(trip.xy)
            assert all(0 <= arc < len(trip.graph.arcs) for arc in result["predicted_arcs"])
            assert set(result["metrics"]) == {"point_accuracy", "sequence_similarity", "route_continuity"}
            assert set(result["geometry"]) == {"gps", "route", "roads"}
            assert len(result["geometry"]["gps"]) == len(trip.xy)
            for part in result["route_parts"]:
                arcs = trip.graph.arcs[part]
                assert (arcs[1:, 0] == arcs[:-1, 1]).all()
            records.append({"trip_id": trip_id, "http_status": response.status_code,
                            "observations": result["observations"], **result["metrics"],
                            "parts": result["decoder"]["parts"], "elapsed_seconds": result["elapsed_seconds"]})
            print(f"POST /predict {trip_id} -> {response.status_code} "
                  f"({result['observations']} points, {result['metrics']['point_accuracy']:.2%})")
    return records


def capture(ui, api, output, channel):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(ui + "/?api=" + api, wait_until="domcontentloaded")
        page.wait_for_function("document.querySelectorAll('#trip option').length === 10")
        page.select_option("#trip", "00000091")
        with page.expect_response(lambda r: r.url.endswith("/predict")) as pending:
            page.locator("#run").click()
        assert pending.value.ok
        page.wait_for_function("document.querySelector('#accuracy').textContent.includes('%')")
        page.screenshot(path=output / "evaluator_ui.png")
        el = page.locator("#route")
        el.uncheck()
        el.check()
        page.locator("#zoom-in").click()
        page.locator("#reset").click()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.body.scrollWidth") == 390
        page.screenshot(path=output / "evaluator_ui_mobile.png", full_page=True)
        assert not errors, errors

        map_path = (output / "student_trip_folium.html").resolve()
        if map_path.is_file():
            geo = browser.new_page(viewport={"width": 1000, "height": 610}, ignore_https_errors=True)
            geo.goto(map_path.as_uri(), wait_until="load", timeout=50000)
            geo.wait_for_selector(".leaflet-control-layers-overlays", timeout=20000)
            assert geo.locator(".leaflet-control-layers-overlays label").count() == 4
            geo.wait_for_function("""() => Object.keys(window).some(key => /^map_[a-f0-9]+$/.test(key) && window[key]?.getZoom?.() > 11)""")
            geo.screenshot(path=output / "folium_map.png")


        docs = browser.new_page(viewport={"width": 1150, "height": 800}, ignore_https_errors=True)
        docs.goto(api + "/docs", wait_until="domcontentloaded")
        post = docs.locator(".opblock-post")
        post.locator(".opblock-summary").click(timeout=18000)
        post.get_by_role("button", name="Try it out").click()
        post.locator("textarea").fill('{"trip_id": "00000091"}')
        with docs.expect_response(lambda r: r.url.endswith("/predict")) as pending:
            post.get_by_role("button", name="Execute").click()
        assert pending.value.status == 200
        live = post.locator(".live-responses-table")
        live.get_by_text("200", exact=True).wait_for()
        live.screenshot(path=output / "api_response.png")
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--ui", default="http://127.0.0.1:8080")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("local/figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = verify(args.api, args.data)
    write_rows(args.results / "api_smoke.csv", records)
    expected = {row["trip_id"]: row for row in csv.DictReader(
        (args.results / "routes_by_trip.csv").open(encoding="utf-8")
    ) if row["method"] == "Viterbi"}
    for row in records:
        for field in ("point_accuracy", "sequence_similarity", "route_continuity"):
            assert abs(row[field] - float(expected[row["trip_id"]][field])) < 1e-10
    capture(args.ui, args.api, args.output, args.channel)
