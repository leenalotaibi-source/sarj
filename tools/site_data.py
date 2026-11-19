"""Refresh the demo page's data block (site/index.html) from a real offline run of OUR system.

Run from the kit root (where ttbench/ and data/ are):

    python /path/to/repo/tools/site_data.py /path/to/repo/site/index.html

What it replaces, per call (everything else in the block is kept: audio envelopes, labels, baseline):
    ours       decision times (STOP / TAKE_TURN), scenario-relative ms
    events[i].ours   outcome, credit and detail from the OFFICIAL scorer
    evidence   per 100 ms: caller speech in the current run as % of what the rules needed for STOP
    heard      what the words layer (Whisper tiny) returned, when, and the verdict
and the summary ("ours": Core, R, S, EOT, false stops, misses, latency, categories).

Words are run inline and released after their measured compute time (same as tools/kit_eval.py).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttbench.dataset import lay_out_stream, load_split  # noqa: E402
from ttbench.harness import FRAME_MS, FRAME_SAMPLES, build_stream  # noqa: E402
from ttbench.score import load_labels, score  # noqa: E402

import my_system  # noqa: E402
from tt.eot import words_verdict  # noqa: E402
from tt.words import WordsWorker, classify  # noqa: E402

BLOCK = re.compile(r'(<script type="application/json" id="data">)(.*?)(</script>)', re.S)


def run(data_dir: Path):
    labels, manifest = load_labels(data_dir)
    scen = lay_out_stream(load_split(data_dir))
    stream, controls = build_stream(scen)
    m = my_system.MySystem()
    if m.words is not None:
        m.words = WordsWorker(m.words.backend, sync=True)

    per = {}            # scenario -> dict(decisions, evidence, heard)
    cur = None
    start = 0
    last_f = {}
    log = []
    ev_acc = []

    def hook(dec):
        fe_process = dec.fe.process

        def process(a, c):
            f = fe_process(a, c)
            last_f["f"] = f
            return f
        dec.fe.process = process

    if m.words is not None:            # one worker for the whole run: wrap it once
        poll = m.words.poll

        def wpoll(now):
            r = poll(now)
            if r is not None and cur is not None:
                req, text = r
                t = round(now)            # the decider clock restarts with every call
                if req and req[0] == "eot":
                    v = words_verdict(text, m.dec.eot.digits_needed) or "no objection"
                    per[cur]["heard"].append([t, text, "eot:" + v])
                elif text:
                    per[cur]["heard"].append([t, text, classify(text) or "none"])
            return r
        m.words.poll = wpoll

    n = len(stream) // FRAME_SAMPLES
    for i in range(n):
        for c in controls.get(i, []):
            if c["type"] == "scenario_start":
                if cur is not None:
                    per[cur]["evidence"] = ev_acc
                m.on_scenario_start(c.get("context", {}))
                hook(m.dec)
                cur = c["scenario_id"]
                start = next(s.start_ms for s in scen if s.scenario_id == cur)
                per[cur] = {"decisions": [], "heard": []}
                ev_acc = []
            elif c["type"] == "agent_utterance_start":
                m.on_agent_start(c.get("text", ""), c.get("duration_ms", 0))
            elif c["type"] == "agent_utterance_end":
                m.on_agent_end()
        fr = stream[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        d = m.on_frame(fr[:, 0], fr[:, 1])
        t_abs = (i + 1) * FRAME_MS
        if d:
            log.append({"dir": "rx", "stream_ms": t_abs, "msg": {"type": "decision", "decision": d}})
            per[cur]["decisions"].append([round(t_abs - start), d])
        # evidence curve: one value per 100 ms (max over its 5 frames)
        dec = m.dec
        val = 0.0
        if dec.agent_on and dec.run_start is not None and "f" in last_f and dec.run_gap <= 1500:
            need = dec._needed_ms(last_f["f"])
            val = 100.0 * dec.run_speech / max(need, 1.0)
        k = int((t_abs - start - 1) // 100)
        while len(ev_acc) <= k:
            ev_acc.append(0.0)
        ev_acc[k] = max(ev_acc[k], min(val, 250.0))
    per[cur]["evidence"] = ev_acc
    meta = {"scenarios": [{"scenario_id": s.scenario_id, "start_ms": s.start_ms, "duration_ms": s.duration_ms}
                          for s in scen]}
    res = score(labels, meta, log, manifest.get("thresholds"))
    starts = {s.scenario_id: s.start_ms for s in scen}
    return res, per, starts, m.demo_note


def main():
    site = Path(sys.argv[1])
    html = site.read_text(encoding="utf-8")
    mt = BLOCK.search(html)
    D = json.loads(mt.group(2))
    res, per, starts, note = run(Path("data/dev"))

    def rel(detail, sid):
        return re.sub(r"(\d+) ms(?! after| before)", lambda g: f"{int(g.group(1)) - starts[sid]} ms", detail)

    evs = {e["event_id"]: e for e in res["events"]}
    for c in D["calls"]:
        p = per[c["id"]]
        c["ours"] = p["decisions"]
        c["evidence"] = [round(v) for v in p["evidence"][: len(c["env_caller"])]]
        c["heard"] = p["heard"]
        for e in c["events"]:
            r = evs.get(e["id"])
            if r:
                e["ours"] = {"outcome": r["outcome"], "credit": round(r["credit"], 3),
                             "detail": rel(r.get("detail", ""), c["id"])}
    core, rates, lat = res["core"], res["rates"], res["latency_ms"]
    D["ours"] = {"core": round(core["core_score"], 1), "R": round(core["R"], 3), "S": round(core["S"], 3),
                 "eot": round(res["eot"]["eot_score"], 1), "false_stops": rates["false_stops"],
                 "misses": rates["misses"], "lat_median": lat["median"], "lat_p90": lat["p90"],
                 "categories": res["categories"], "slices": res["slices"], "note": note}
    D["threshold"] = 100
    new = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    site.write_text(html[:mt.start(2)] + new + html[mt.end(2):], encoding="utf-8")
    print(f"Core {core['core_score']:.1f}  R {core['R']:.3f}  S {core['S']:.3f}  EOT {res['eot']['eot_score']:.1f}"
          f"  false stops {rates['false_stops']}  misses {rates['misses']}  ({note}) -> {site}")


if __name__ == "__main__":
    main()
