# Third-Party Notices

## tau-bench

`taucollide/run_tau.py` adapts the simulated-user prompt of tau-bench's `tau_bench/envs/user.py` and the authentication rule of the retail policy (`tau_bench/envs/retail/wiki.md`). The lookup description and the name and zip matching adapt `tau_bench/envs/retail/tools/find_user_id_by_name_zip.py`. Its agent loop follows `tau_bench/agents/tool_calling_agent.py`, and its state hash and reward check follow `tau_bench/envs/base.py`. The rest of the tau-bench collision code imports the `tau_bench` package from a checkout (commit `59a200c`), which is not part of this repository.

Source: <https://github.com/sierra-research/tau-bench>

```text
MIT License

Copyright (c) 2024 Sierra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
