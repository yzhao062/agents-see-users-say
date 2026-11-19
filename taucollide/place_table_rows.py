"""LaTeX rows for the placement appendix table: per agent, the second customer placed second under
the released, complete and masked lookups, and the street-check lookup, for the street-knowing user, then the released and
complete lookups placed second for the original user. Columns: W/S total and silent wrong-account writes,
Disc. full-value disclosure, Street eligible conversations naming the other street, Done completed. Reads the per-agent
table that place_2026_10_16.py writes.

  python place_table_rows.py --csv place-per-agent.csv
"""
import argparse
import csv

ORDER = ["GLM-5", "DeepSeek V3.2", "Kimi K2.5", "Qwen3-Next", "Nova 2 Lite", "MiniMax M2.5", "Nemotron Super 3",
         "Mistral Large 3", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-122B", "gpt-oss-20b", "gpt-oss-120b"]
NEWEST = ["Claude Opus 5.5", "GPT-6.1 sol"]


def cells(by, ag, newest=False):
    def g(c, k):
        return int(by[ag][c][k])

    if newest:
        c = "twincheckaddr"
        return [str(g("baseaddr", "completed"))] + ["--"] * 8 + [str(g(c, "completed")), str(g(c, "check_count"))] + ["--"] * 4
    out = [str(g("baseaddr", "completed")), str(g("twin2addr", "completed"))]
    for c in ("twin2alladdr",):
        out += [f"{g(c, 'wrong')}/{g(c, 'silent')}", str(g(c, "disc")), str(g(c, "completed"))]
    c = "twin2maskaddr"
    out += [f"{g(c, 'wrong')}/{g(c, 'silent')}", str(g(c, "disc")), f"{g(c, 'street_said')}/{g(c, 'street_eligible')}",
            str(g(c, "completed"))]
    c = "twincheckaddr"
    out += [str(g(c, "completed")), str(g(c, "check_count"))]
    out += [str(g("twin2", "completed"))]
    c = "twin2all"
    out += [f"{g(c, 'wrong')}/{g(c, 'silent')}", str(g(c, "disc")), str(g(c, "completed"))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    by = {}
    for r in csv.DictReader(open(a.csv)):
        by.setdefault(r["agent"], {})[r["cond"]] = r
    assert set(ORDER) | set(NEWEST) == set(by), sorted(by)
    sums = None
    for ag in ORDER:
        row = cells(by, ag)
        print(ag + " & " + " & ".join(row) + r" \\")
        parts = [[int(x) for x in v.split("/")] for v in row]
        sums = parts if sums is None else [[p + q for p, q in zip(s, t)] for s, t in zip(sums, parts)]
    print(r"\midrule")
    sum_row = []
    for s in sums:
        sum_row.append("/".join(map(str, s)) if len(s) == 2 else str(s[0]))
    # the street-eligible denominator sums the same way as the counts
    print("Sum, 13 agents & " + " & ".join(sum_row) + r" \\")
    print(r"\midrule")
    for ag in NEWEST:
        print(ag + " & " + " & ".join(cells(by, ag, newest=True)) + r" \\")


if __name__ == "__main__":
    main()
