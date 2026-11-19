"""LaTeX rows for the experiment B per-agent appendix table, from b-addr-per-agent.csv (b_addr_2026_10_15.py)."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    by.setdefault(r["agent"], {})[r["cond"]] = r
order = list(by)


def c(r, k):
    return int(r[k])


out = []
for i, ag in enumerate(order):
    if i == 13:
        out.append("\\midrule")
    d = by[ag]
    n = c(d["baseaddr"], "n")
    cells = [ag,
             str(c(d["baseaddr"], "completed")), str(c(d["basenotepoladdr"], "completed"))]
    for cond in ("twinaddr", "twinnotepoladdr", "twinalladdr", "twinmaskaddr"):
        r = d[cond]
        cells += [f"{c(r, 'wrong')}/{c(r, 'silent')}", str(c(r, "disc")), str(c(r, "completed"))]
    m = d["twinmaskaddr"]
    cells += [str(c(m, "intended")), str(c(m, "opened_second")), f"{c(m, 'street_said')}/{c(m, 'street_eligible')}"]
    out.append(" & ".join(cells) + " \\\\")
tot = {}
main = order[:13]
s = lambda cond, k: sum(c(by[a][cond], k) for a in main)
cells = ["Sum, 13 agents", str(s("baseaddr", "completed")), str(s("basenotepoladdr", "completed"))]
for cond in ("twinaddr", "twinnotepoladdr", "twinalladdr", "twinmaskaddr"):
    cells += [f"{s(cond, 'wrong')}/{s(cond, 'silent')}", str(s(cond, "disc")), str(s(cond, "completed"))]
cells += [str(s("twinmaskaddr", "intended")), str(s("twinmaskaddr", "opened_second")),
          f"{s('twinmaskaddr', 'street_said')}/{s('twinmaskaddr', 'street_eligible')}"]
out.insert(13, "\\midrule\n" + " & ".join(cells) + " \\\\")
print("\n".join(out))
