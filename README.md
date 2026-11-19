# What LLM Agents See and What Users Say

[![arXiv](https://img.shields.io/badge/arXiv-2611.17639-b31b1b.svg)](https://arxiv.org/abs/2611.17639)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](LICENSE)

Code for the paper [*What LLM Agents See and What Users Say*](https://arxiv.org/abs/2611.17639) by Yue Zhao, Angela Huyue Zhang, S. Alex Yang, Hua Wei, and Xiyang Hu.

Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? The paper varies what a lookup returns about a second matching record (nothing, a warning, a count, or the record itself) and whether the user states a claim about the record. Under a role-only prompt, a count of a second record leaves most of 36 earlier agents acting on the one they see, whereas listing both stops most. In a retail benchmark, when a first-match lookup returned an added customer with the user's name and zip code, earlier agents wrote to the wrong account in about a quarter of conversations.

> [!NOTE]
> Appendix figures A1 to A9 regenerate now from the CSV files included here. This first release also holds the code, the prompts and tool fixtures, the lists that map each table row to its rollout files, and the scripts that compute the paper's tables and data figures. The rollouts and the survey's per-tool labels will follow in a data release. Run the scripts that read them only on the complete data release, since missing inputs can produce zero counts or incomplete tables. A few numbers in the running text come from short checks that will be added with the data release.

## Contents

| Folder | What it holds |
|---|---|
| `licgrid/` | The grid and the empty-claim tasks. `templates.py` holds the 16 which-record tasks and `absence_templates.py` the 16 empty-claim tasks, with their prompts and tool fixtures; the people, companies, and contact details in the fixtures are invented. `run_grid.py` and `run_absence.py` run an agent through each lookup variant, and `score_grid.py` and `score_absence.py` label the rollouts. `all_agents.py` lists the rollout files behind each agent's rows (`manifest(work)`, `frontier_manifest()`). The remaining scripts compute the tables and figures listed below. |
| `taucollide/` | The tau-bench retail collision: a second customer with the requester's name and zip code. `run_tau.py` accepts 14 base conditions. The released, complete, and masked lookups test both placements of the added customer, and the street check tests the added customer first. Other conditions change the lookup description, the policy, or the prompt, and a condition name ending in `addr` gives the simulated user its street address. The other scripts compute the tau-bench tables; module constants such as `AGENTS` in `tau_table.py` list the rollout files behind each row. |
| `survey/`, `apisurvey/` | Aggregation of the survey of lookup tools in MCP servers, agent toolkits, and agent benchmarks. `apisurvey/aggregate.py` is the first pass, which `survey/reaggregate_2026_10_13.py` re-derives with the spot-check corrections. |
| `appendix_figs/<name>/` | One folder per figure for A1 to A9, each with `make_fig.py` and the `data.csv` it reads. |

## Setup

```bash
pip install -r requirements.txt
```

The code was run with Python 3.12. Run every command from the repository root, since the data folders below are relative to it.

The tau-bench collision also needs a tau-bench checkout at commit `59a200c`:

```bash
git clone https://github.com/sierra-research/tau-bench data/tau-bench
git -C data/tau-bench checkout 59a200c
```

Three environment variables say where data live:

| Variable | Default | Holds |
|---|---|---|
| `WTA_WORK` | `data/work` | Rollout folders that the analysis scripts read (for example `stated/`, `onrecord/`, `results/`) and the `logs/` folder some of them write to |
| `WTA_ARTIFACTS` | `data/artifacts` | Archived rollout bundles (`licgrid-rollouts-2026-10-12/`, `taucollide-rollouts-2026-10-12/`), the survey labels (`api-survey-2026-10-12/`), and the per-agent tables of the tau-bench analyses |
| `WTA_TAU_BENCH` | `data/tau-bench` | The tau-bench checkout (`run_tau.py` and `analyze_tau.py` also take `--tau-root`) |

The data release will unpack into this layout, and the run scripts below write wherever `--out` points.

## What Runs Without the Rollouts

The tests check the task templates, tool contracts, and scoring:

```bash
python licgrid/test_grid.py                    # prints "1920 rollouts, 0 problems"
python -m pytest -q licgrid/test_absence.py    # 418 passed
```

Each of the appendix figures A1 to A9 regenerates from the CSV beside its script, for example:

```bash
python appendix_figs/survey/make_fig.py   # writes fig.pdf and fig.png next to the script
```

Most figure scripts set Times New Roman. Where it is not installed, as on many Linux machines, matplotlib falls back to its default font.

## Running an Agent

`run_grid.py` calls any OpenAI-compatible endpoint (for example a local vLLM server) or Amazon Bedrock:

```bash
python licgrid/run_grid.py --base-url http://127.0.0.1:8000/v1 --model <model> --out runs/<model>.jsonl
python licgrid/run_grid.py --bedrock-region us-east-1 --model <bedrock-model-id> --out runs/<model>.jsonl
```

A hosted API takes `--api-key-file <path>`, and Bedrock reads credentials from the standard AWS credential chain. Rerunning with the same `--out` skips the rollouts already written and refuses a file written by another model or prompt norm. If any rollout fails, the run exits with status 1 and keeps the errors, so the next run retries those cells. Resume does not check other settings, such as reasoning effort or token limits, so give each setting its own `--out`.

By default, `run_grid.py` runs the listed (`all`) and first-match (`first`) contracts under three consequences (`irreversible`, `reversible`, `report`). The paper's other cells use `--contracts` (`page`, `pagerec`, `reclist`, `firstdoc`, `firstnote`, `compound`), `--consequences` (`count`, `save`, `share`), and `--norm fiduciary` for the prompt check. `--stated` adds the user's claim; it takes only the `irreversible` and `reversible` consequences and no `compound` contract. `templates.py` and `run_grid.py` define each variant.

The empty-claim tasks run the same way, with `--unstated` for the runs without the user's claim:

```bash
python licgrid/run_absence.py --bedrock-region us-east-1 --model <bedrock-model-id> --out runs/absence-<model>.jsonl
```

For the tau-bench collision, the agent runs on Bedrock or on an OpenAI-compatible endpoint (`--agent-base-url`). The simulated user always runs on Bedrock (`--user-model`, default `deepseek.v3.2`), so these runs need AWS credentials:

```bash
python taucollide/run_tau.py --agent-model <bedrock-model-id> --conditions base,twin,twindoc,twinall --out runs/tau-<model>.jsonl
```

Each run script has `--help` for the remaining options.

## Paper Tables and Figures

The grid and tau-bench analyses read the rollouts, and the survey analyses read the per-tool labels; both come with the data release. Commands with explicit output paths below write to `out/` (create it with `mkdir -p out`), which git ignores. Other scripts print their tables or write to their documented defaults.

| Paper item | Script |
|---|---|
| Grid table: acting by lookup and claim | `licgrid/table1_exact_2026_10_15.py`, `licgrid/all_agents.py` |
| Grid figure: opening the second record against acting | `licgrid/panel_r2_2026_10_15.py --out out`, then `licgrid/grid_figure.py --art out --out out/grid.pdf` |
| Summary table: count against listing, for both task families | `licgrid/itt.py`, `licgrid/crossover.py` |
| Selectivity figure: acting when the claim is true minus when it is false | `licgrid/selectivity_cells_2026_11_08.py out/cells.csv`, then `licgrid/selectivity_hist_2026_11_08.py --csv out/cells.csv --out out/selectivity.pdf` |
| Stating the claim in both worlds | `licgrid/panel_r2_2026_10_15.py --out out`, `licgrid/two_world_all_2026_10_15.py` |
| tau-bench table: wrong-account writes, leaks, and completion by lookup | `taucollide/place_2026_10_16.py --out-dir out`; the Clean column from `taucollide/clean_2026_10_16.py` |
| Per-agent street-check figure (A10) | `taucollide/place_2026_10_16.py --out-dir out`, then `taucollide/clean_nostreet_2026_10_20.py --csv out/place-per-agent.csv --out out/place-ns.csv`, then `taucollide/place_figure_radar.py --csv out/place-ns.csv --layout strip --right predef --out out/street-check.pdf` |
| Per-agent grid tables (all earlier agents; newest models) | `licgrid/all_agents.py --latex out/agents.tex --frontier-latex out/frontier.tex`, `licgrid/itt.py --frontier` |
| Empty-claim tasks: templates and per-agent results | `licgrid/absence_templates.py`; `licgrid/score_absence.py --final "${WTA_WORK:-data/work}/absence"` (acting when the claim is true, and probe answers, per agent); `licgrid/crossover.py --per-agent` (acting when the claim is false, with and without the claim) |
| Paired gaps and record-count crossover tables | `licgrid/itt.py`, `licgrid/crossover.py --per-agent`, `licgrid/onrecord.py --per-agent` |
| Opening the second record, per agent | `licgrid/seeing.py` |
| Survey of lookup tools | `survey/reaggregate_2026_10_13.py`, `survey/cap_sizes_2026_10_15.py` |
| tau-bench per-agent, newest-model, reasoning, and Gemini tables | `taucollide/tau_table.py --paper out/tau-rows.tex`; the Gemini table's name-only columns from `taucollide/audit_2026_10_13.py` |
| Street-knowing user tables | `taucollide/b_addr_2026_10_15.py --out-dir out`, then `taucollide/b_table_rows_2026_10_15.py out/b-addr-per-agent.csv`; `taucollide/place_table_rows.py --csv out/place-per-agent.csv` |
| Leaks of the other customer's data | `taucollide/disclosure_2026_10_15.py --out-dir out`, `taucollide/disclosure_conditions_2026_10_15.py --out-dir out` |
| Checks behind numbers in the text | `licgrid/compound_both.py` (acting on both matches), `licgrid/remedy.py` (the prompt check of Proposition A.2), `licgrid/reversible.py`, `licgrid/reclist.py`, `licgrid/cross_model.py`, `licgrid/in_episode.py`, `licgrid/mechanism.py`, `taucollide/audit_2026_10_13.py`, `taucollide/wrong_writes.py`, `taucollide/tau_summary.py` |

Without `--out-dir`, `place_2026_10_16.py` and `b_addr_2026_10_15.py` write to `$WTA_ARTIFACTS/tau-place/` and `$WTA_ARTIFACTS/tau-b-addr/`.

## Citation

```bibtex
@article{zhao2026agentssee,
  title   = {What {LLM} Agents See and What Users Say},
  author  = {Zhao, Yue and Zhang, Angela Huyue and Yang, S. Alex and Wei, Hua and Hu, Xiyang},
  journal = {arXiv preprint arXiv:2611.17639},
  year    = {2026}
}
```

## License

The code is released under the BSD 2-Clause License ([LICENSE](LICENSE)). `taucollide/run_tau.py` adapts prompts, a policy excerpt, a tool description, and parts of the agent loop and reward check from tau-bench, which is under the MIT License; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
