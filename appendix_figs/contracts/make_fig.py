"""Appendix schematic of the six search-tool contracts as small cards (2 columns x 3 rows).

Reads data.csv beside this file and writes fig.pdf, fig.png (300 dpi) and drawn_text.json (the strings drawn, to
check against data.csv). Layout units are points: the axes cover the whole figure, 1 data unit = 1 pt. The figure is included
at its native width (6.75 in), so every font size below is the printed size; all text is at least 7 pt.

Result mocks use the real tool's keys (results, total_matches, has_more, note; run_grid.py, Env.call) and the real note
wording (run_grid.py, contract "firstnote") with the entity noun e = "record". Ids, the name, and the "..." (the other
record fields) are illustrative; line breaks and indentation are added for legibility (the tool returns json.dumps).
"""
import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

HERE = Path(__file__).resolve().parent
TEAL, ORANGE, GRAY, MINT, CORAL, INK, LGRAY = "#2F7F93", "#D9692F", "#9A9A9A", "#5FAF8F", "#ED8D5A", "#333333", "#C9C9C9"
GROUP_COLOR = {"mint": MINT, "teal": TEAL, "coral": CORAL}
GROUP_TINT = {"mint": 0.55, "teal": 0.62, "coral": 0.55}  # share of white blended into the header strip
ENTITY = "record"  # e in the mocks

W = 486.0  # 6.75 in in points
MX, GX, GY = 1.5, 6.0, 5.0  # outer margin, column gap, row gap
NCOL, NROW = 2, 3
PAD = 6.0  # card inner padding
# every text at least 7 pt at final size
DESC_PT, NAME_PT, MONO_PT, LABEL_PT, CHIP_PT = 7.5, 8.5, 7.0, 7.0, 7.0
DESC_LEAD, MONO_LEAD = 9.2, 8.8
HEAD_H = 15.0
BOX_TOP, BOX_BOT = 6.5, 5.0  # padding inside the mock box (the top leaves room for the label on the border)
DIM = "#555555"  # text of labels and of the dashed (not returned) chips; dark enough for 7 pt text
MOCK_BG, INSTR_INK = "#F3F3F3", "#8A3E12"

SERIF = dict(family="DejaVu Serif")
MONO = dict(family="DejaVu Sans Mono")
_t2p = TextToPath()


def fp(kind, size, bold=False):
    if kind == "m":
        return FontProperties(size=size, **MONO)
    return FontProperties(size=size, style="italic" if kind == "i" else "normal",
                          weight="bold" if bold else "normal", **SERIF)


def tw(s, kind, size, bold=False):
    return _t2p.get_text_width_height_descent(s, fp(kind, size, bold), False)[0]


def blend(c, share):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * share, g + (1 - g) * share, b + (1 - b) * share)


def parse_words(s):
    """'a *e*s `m`' -> words (split on spaces); a word is a list of (text, kind) pieces, kind in r, i, m."""
    words = []
    for w in s.split(" "):
        pieces = []
        for part in re.split(r"(\*[^*]+\*|`[^`]+`)", w):
            if not part:
                continue
            if part.startswith("*"):
                pieces.append((part[1:-1], "i"))
            elif part.startswith("`"):
                pieces.append((part[1:-1], "m"))
            else:
                pieces.append((part, "r"))
        words.append(pieces)
    return words


def wrap_rich(s, width, size):
    """Greedy wrap of marked-up text; returns lines, each a list of words (lists of pieces)."""
    words, lines, cur, cw = parse_words(s), [], [], 0.0
    sp = tw(" ", "r", size)
    for w in words:
        wl = sum(tw(t, k, size) for t, k in w)
        add = wl if not cur else sp + wl
        if cur and cw + add > width:
            lines.append(cur)
            cur, cw = [w], wl
        else:
            cur.append(w)
            cw += add
    if cur:
        lines.append(cur)
    return lines


def draw_rich_line(ax, x, y, line, size, color=INK):
    sp = tw(" ", "r", size)
    for wi, w in enumerate(line):
        if wi:
            x += sp
        for t, k in w:
            ax.text(x, y, t, fontproperties=fp(k, size), color=color, ha="left", va="baseline")
            x += tw(t, k, size)


def plain(line):
    return " ".join("".join(t for t, _ in w) for w in line)


def mock_lines(row, max_chars):
    """Return [(text, emphasis)] lines of the K = 2 result mock; emphasis in None, 'page', 'note'."""
    n = int(row["n_returned"])
    rec = lambda i: f'{{"id": "R-{i}", "name": "Ana Lee", ...}}'  # noqa: E731
    extras = []  # (key, value-as-json, emphasis)
    if row["total_matches"]:
        extras.append(("total_matches", row["total_matches"], "page"))
    if row["has_more"]:
        extras.append(("has_more", row["has_more"], "page"))
    if row["note"]:
        extras.append(("note", '"' + row["note"].replace("*e*", ENTITY) + '"', "note"))
    out = [('{"results": [', None)]
    for i in range(1, n + 1):
        end = "," if i < n else ("]," if extras else "]}")
        out.append(("  " + rec(i) + end, None))
    # one line per extra field; consecutive fields of one kind share a line when they fit (total_matches, has_more)
    groups = []
    for k, v, emph in extras:
        item = f'"{k}": {v}'
        if groups and groups[-1][1] == emph and len(" " + groups[-1][0] + ", " + item + "}") <= max_chars:
            groups[-1][0] += ", " + item
        else:
            groups.append([item, emph])
    for j, (item, emph) in enumerate(groups):
        text = " " + item + ("}" if j == len(groups) - 1 else ",")
        if len(text) <= max_chars:
            out.append((text, emph))
        else:  # wrap the long string value, hanging indent under the opening quote
            head = text[: text.index(": ") + 2]
            parts = textwrap.wrap(text[len(head):], width=max_chars - len(head), break_long_words=False)
            out.append((head + parts[0], emph))
            for p in parts[1:]:
                out.append((" " * (len(head) + 1) + p.lstrip(), emph))
    return out


def main():
    rows = list(csv.DictReader(open(HERE / "data.csv", newline="", encoding="utf-8")))
    # reading order of the 2 x 3 grid: listed, page / bare, documented / noted, compound
    assert [r["contract"] for r in rows] == ["listed", "page", "bare", "documented", "noted", "compound"]
    plt.rcParams.update({"font.size": 6.5, "font.family": "serif", "pdf.fonttype": 42})

    cw = (W - 2 * MX - (NCOL - 1) * GX) / NCOL
    inner = cw - 2 * PAD
    mono_w = tw("M", "m", MONO_PT)
    box_pad = 3.5  # left padding inside the mock box
    max_chars = int((inner - 2 * box_pad) / mono_w)

    # ---- content and heights ----
    cards = []
    for r in rows:
        desc = wrap_rich(r["description"], inner, DESC_PT)
        mock = mock_lines(r, max_chars)
        instr = ""
        if r["instruction"]:
            s = re.sub(r"^the ", "", r["instruction"])
            instr = s[0].upper() + s[1:]
        cards.append(dict(row=r, desc=desc, mock=mock, instr=instr))
    for c in cards:  # the mock must fit its box
        for t, _ in c["mock"]:
            assert len(t) <= max_chars, (c["row"]["contract"], len(t), max_chars, t)

    instr_h = 12.5
    row_geo = []
    for ri in range(NROW):
        grp = cards[NCOL * ri: NCOL * ri + NCOL]
        n_desc = max(len(c["desc"]) for c in grp)
        desc_h = n_desc * DESC_LEAD + (instr_h + 3.0 if any(c["instr"] for c in grp) else 0)
        n_mock = max(len(c["mock"]) for c in grp)
        box_h = BOX_TOP + 0.8 * MONO_PT + (n_mock - 1) * MONO_LEAD + BOX_BOT
        h = HEAD_H + 4.0 + desc_h + 8.0 + box_h + 5.0
        row_geo.append(dict(h=h, desc_h=desc_h, box_h=box_h))
    key_h = 13.0
    H = MX + sum(g["h"] for g in row_geo) + NROW * GY - 1 + key_h + MX
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    drawn = {}
    top = H - MX
    for ri in range(NROW):
        geo = row_geo[ri]
        for ci in range(NCOL):
            c = cards[NCOL * ri + ci]
            r = c["row"]
            col = GROUP_COLOR[r["group"]]
            x0 = MX + ci * (cw + GX)
            y1 = top  # card top
            y0 = top - geo["h"]
            card = FancyBboxPatch((x0, y0), cw, geo["h"], boxstyle="round,pad=0,rounding_size=3.0", fc="white",
                                  ec=col, lw=1.1, zorder=2)
            ax.add_patch(card)
            hd = Rectangle((x0, y1 - HEAD_H), cw, HEAD_H, fc=blend(col, GROUP_TINT[r["group"]]), ec="none", zorder=3)
            ax.add_patch(hd)
            hd.set_clip_path(card)
            ax.plot([x0, x0 + cw], [y1 - HEAD_H] * 2, color=col, lw=0.8, zorder=4, solid_capstyle="butt")
            ax.text(x0 + PAD, y1 - HEAD_H / 2 - 0.2, r["contract"], fontproperties=fp("r", NAME_PT, True), color=INK,
                    ha="left", va="center", zorder=5)
            # returned-record chips (solid: returned by this call; dashed: matches but not returned)
            n_ret = int(r["n_returned"])
            chip_w, chip_h, chip_gap = 20.0, 10.0, 3.0
            for k in (2, 1):
                cx = x0 + cw - PAD - chip_w - (2 - k) * (chip_w + chip_gap)
                returned = k <= n_ret
                chip = FancyBboxPatch((cx, y1 - HEAD_H / 2 - chip_h / 2), chip_w, chip_h,
                                      boxstyle="round,pad=0,rounding_size=1.8",
                                      fc="white" if returned else "none", ec=INK if returned else "#6E6E6E",
                                      lw=0.7, ls="-" if returned else (0, (1.6, 1.2)), zorder=5)
                ax.add_patch(chip)
                ax.text(cx + chip_w / 2, y1 - HEAD_H / 2 - 0.2, f"R-{k}", fontproperties=fp("m", CHIP_PT),
                        color=INK if returned else DIM, ha="center", va="center", zorder=6)
            # description
            yb = y1 - HEAD_H - 4.0 - DESC_PT * 0.82
            for line in c["desc"]:
                draw_rich_line(ax, x0 + PAD, yb, line, DESC_PT)
                yb -= DESC_LEAD
            drawn[r["contract"]] = dict(desc=" ".join(plain(l) for l in c["desc"]), mock=[t for t, _ in c["mock"]],
                                        instr=c["instr"])
            # compound: the instruction change, in the user's orange
            if c["instr"]:
                iy = y1 - HEAD_H - 4.0 - len(c["desc"]) * DESC_LEAD - 0.8
                ax.add_patch(FancyBboxPatch((x0 + PAD, iy - instr_h), inner, instr_h,
                                            boxstyle="round,pad=0,rounding_size=2.0", fc=blend(ORANGE, 0.86),
                                            ec=ORANGE, lw=0.7, zorder=3))
                ax.text(x0 + PAD + 4, iy - instr_h / 2 - 0.2, c["instr"], fontproperties=fp("i", DESC_PT),
                        color=INSTR_INK, ha="left", va="center", zorder=5)
            # result mock, anchored at the card bottom
            by0 = y0 + 5.0
            ax.add_patch(FancyBboxPatch((x0 + PAD, by0), inner, geo["box_h"], boxstyle="round,pad=0,rounding_size=2.0",
                                        fc=MOCK_BG, ec=LGRAY, lw=0.7, zorder=3))
            ty = by0 + geo["box_h"] - BOX_TOP - MONO_PT * 0.80
            for text, emph in c["mock"]:
                if emph:
                    tint = blend(TEAL if emph == "page" else CORAL, 0.78)
                    ax.add_patch(Rectangle((x0 + PAD + 1.2, ty - 2.4), inner - 2.4, MONO_LEAD, fc=tint, ec="none",
                                           zorder=3.5))
                ax.text(x0 + PAD + box_pad, ty, text, fontproperties=fp("m", MONO_PT), color=INK, ha="left",
                        va="baseline", zorder=5)
                ty -= MONO_LEAD
            lab = "result at K = 2"
            if r["result_table"].startswith("As "):
                lab += ", " + r["result_table"].split(";")[0].split(",")[0].rstrip(".").lower().replace("as ", "as ")
                if "plus" in r["result_table"]:
                    lab += " plus note"
            lw_ = tw(lab, "r", LABEL_PT) + 6
            ax.add_patch(Rectangle((x0 + PAD + 5, by0 + geo["box_h"] - 4.2), lw_, 8.4, fc="white", ec="none", zorder=4))
            ax.text(x0 + PAD + 8, by0 + geo["box_h"] - 0.2, lab, fontproperties=fp("r", LABEL_PT), color=DIM,
                    ha="left", va="center", zorder=6)
            drawn[r["contract"]]["label"] = lab
        top -= geo["h"] + GY

    # ---- key ----
    ky = MX + key_h / 2 - 0.3
    items = [("R-1", True, "returned by this call"), ("R-2", False, "also matches, not returned")]
    texts = [tw(t, "r", LABEL_PT) for _, _, t in items]
    total = sum(20 + 4 + w for w in texts) + 18
    kx = (W - total) / 2
    for (name, ret, label), w in zip(items, texts):
        ax.add_patch(FancyBboxPatch((kx, ky - 5.0), 20, 10.0, boxstyle="round,pad=0,rounding_size=1.8",
                                    fc="white" if ret else "none", ec=INK if ret else "#6E6E6E", lw=0.7,
                                    ls="-" if ret else (0, (1.6, 1.2)), zorder=5))
        ax.text(kx + 10, ky - 0.2, name, fontproperties=fp("m", CHIP_PT), color=INK if ret else DIM, ha="center",
                va="center", zorder=6)
        ax.text(kx + 24, ky - 0.2, label, fontproperties=fp("r", LABEL_PT), color=DIM, ha="left", va="center")
        kx += 20 + 4 + w + 18
    drawn["_key"] = [t for _, _, t in items]

    fig.savefig(HERE / "fig.pdf")
    fig.savefig(HERE / "fig.png", dpi=300)
    (HERE / "drawn_text.json").write_text(json.dumps(drawn, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"size {W / 72:.2f} x {H / 72:.2f} in; card {cw:.1f} pt wide; mock max {max_chars} chars")


if __name__ == "__main__":
    main()
