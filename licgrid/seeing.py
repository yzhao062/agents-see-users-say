"""Does an agent stop for a second record only once it has seen it? Under page (one match per page, with total_matches
and has_more, and an offset route to the next page), split K = 2 rollouts by whether the agent fetched the second
record before any action (score()["seen"] >= 2), without and with the fiduciary line. Reads the archived rollouts.

  python seeing.py [--work $WTA_WORK] [--latex rows.tex]

Columns (80 rollouts per cell): K1/K2 listed and page acting; two|page, share of page count answers that state two;
saw K2, share of page K = 2 rollouts that fetched the second record before acting; then acted/n among those that saw
both and those that saw one; the same three with the line. Also prints the gpt-oss reasoning-effort sweep under page.
--latex writes the rows of the appendix table: K2 listed, K1 page, K2 page, two|page, saw K2, acted/n after
seeing both and after seeing one, and with the line saw K2 and K2 page.
"""
import os
import argparse
from pathlib import Path

from all_agents import ART, fmt, frontier_manifest, load, manifest
from mechanism import classify


def cell(rows, c, k, q="irreversible"):
    return [r for r in rows if r["contract"] == c and r["k"] == k and r["consequence"] == q]


def share(xs):
    return fmt(sum(xs) / len(xs)) if xs else "-"


def split(rows):
    p2 = cell(rows, "page", 2)
    saw = [r["s"]["acted"] for r in p2 if r["s"]["seen"] >= 2]
    one = [r["s"]["acted"] for r in p2 if r["s"]["seen"] < 2]
    return saw, one


def corr(pts):
    xs, ys = zip(*pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    return num / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5


def tex(x):
    return x.replace("0.", ".") if x != "-" else "--"


def effort_sweep():
    """gpt-oss at three reasoning efforts (scale sweep): page at K = 2 and K = 1."""
    for m in ("gptoss120b", "gptoss20b"):
        for e in ("low", "medium", "high"):
            n = load([ART / f"b2-{m}_{e}__none.jsonl.gz", ART / f"b2-{m}_{e}__none__scope.jsonl.gz"])["none"]
            saw, one = split(n)
            p2 = cell(n, "page", 2)
            print(f"{m} {e}: saw K2 {share([r['s']['seen'] >= 2 for r in p2])}, act K2 {share(saw + one)}, "
                  f"act K1 {share([r['s']['acted'] for r in cell(n, 'page', 1)])}, saw: acted {sum(saw)}/{len(saw)}, "
                  f"one: acted {sum(one)}/{len(one)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(os.environ.get("WTA_WORK", "data/work"))))
    ap.add_argument("--latex", default="")
    a = ap.parse_args()
    lines = []
    hist = {lab: load(fs) for lab, fs in manifest(a.work).items()}
    front = {lab + " (frontier)": load(fs) for lab, fs in frontier_manifest().items()}
    head = ["agent", "K1 listed", "K2 listed", "K1 page", "K2 page", "two|page", "saw K2", "saw: acted", "one: acted",
            "line: saw K2", "line: K1 page", "line: K2 page", "line saw: acted", "line one: acted"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    pool = {k: [0, 0] for k in ("saw", "one", "line saw", "line one")}
    both = [0, 0]  # page K = 2 rollouts that acted after fetching: how many acted on both records
    within, lower, pts, looked = 0, 0, [], {}
    for lab, rows in {**hist, **front}.items():
        n, f = rows["none"], rows["fiduciary"]
        if not cell(n, "page", 2):
            continue
        saw, one = split(n)
        lsaw, lone = split(f)
        cnt = [classify(r["final"]) for r in cell(n, "page", 2, "count")]
        act = lambda rs, c, k: share([r["s"]["acted"] for r in cell(rs, c, k)])
        print(f"| {lab} | {act(n, 'all', 1)} | {act(n, 'all', 2)} | {act(n, 'page', 1)} | {act(n, 'page', 2)} | "
              f"{share([c == 'two' for c in cnt])} | {share([r['s']['seen'] >= 2 for r in cell(n, 'page', 2)])} | "
              f"{sum(saw)}/{len(saw)} | {sum(one)}/{len(one)} | "
              f"{share([r['s']['seen'] >= 2 for r in cell(f, 'page', 2)])} | {act(f, 'page', 1)} | {act(f, 'page', 2)} | "
              f"{sum(lsaw)}/{len(lsaw)} | {sum(lone)}/{len(lone)} |")
        if lab not in hist and not any("midrule" in x for x in lines):
            lines.append("\\midrule\n")
        lines.append(lab.replace(" (frontier)", "") + " & " + " & ".join(
            [tex(act(n, "all", 2)), tex(act(n, "page", 1)), tex(act(n, "page", 2)), tex(share([c == "two" for c in cnt])),
             tex(share([r["s"]["seen"] >= 2 for r in cell(n, "page", 2)])), f"{sum(saw)}/{len(saw)}", f"{sum(one)}/{len(one)}",
             tex(share([r["s"]["seen"] >= 2 for r in cell(f, "page", 2)])), tex(act(f, "page", 2))]) + " \\\\\n")
        if lab in hist:
            for k, xs in (("saw", saw), ("one", one), ("line saw", lsaw), ("line one", lone)):
                pool[k][0] += sum(xs); pool[k][1] += len(xs)
            pts.append((len(saw) / (len(saw) + len(one)), (sum(saw) + sum(one)) / (len(saw) + len(one))))
            acted_saw = [r for r in cell(n, "page", 2) if r["s"]["acted"] and r["s"]["seen"] >= 2]
            both[0] += sum(r["s"]["acted_all"] for r in acted_saw); both[1] += len(acted_saw)
            if len(saw) >= 5 and len(one) >= 5:
                within += 1
                lower += sum(saw) / len(saw) < sum(one) / len(one)
            if lsaw or lone:
                looked[lab] = (len(saw) / 80, len(lsaw) / 80)
    print(f"\nhistorical agents: {len(pts)}")
    for k, (x, m) in pool.items():
        print(f"page K = 2, {k}: acted {x}/{m} = {x / m:.3f}")
    print(f"agents with >= 5 rollouts in each group (no line): {within}; acting lower after seeing both: {lower}")
    print(f"acted after fetching: {both[1]} rollouts, of which acted on both records: {both[0]}")
    print(f"cross-agent correlation, share that saw both vs acting under page: r = {corr(pts):.2f}")
    print("line raised seeing by >= 0.3:", [(lab, fmt(x), fmt(y)) for lab, (x, y) in looked.items() if y - x >= 0.3])
    print()
    effort_sweep()
    if a.latex:
        Path(a.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.latex).write_text("".join(lines))


if __name__ == "__main__":
    main()
