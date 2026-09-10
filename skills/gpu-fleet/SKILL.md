---
name: gpu-fleet
description: Check GPU availability and running jobs across the team's GPU hosts before launching, scheduling or debugging GPU work. Use when asked which GPUs are free, what is running where, whether a host is busy, or where a job should run.
---

# GPU fleet check

The fleet is monitored by mnvitop. Prefer its MCP tools; fall back to the CLI.

## Which tool

1. Placement question ("where can I run this", "find me 2 GPUs"): call `find_gpus(count, min_free_gb)`.
   Read `summary`; use `placements[0].host` and `placements[0].gpus`. If `found` is false, read
   `alternatives` and say clearly that only shared GPUs are available.
2. Overview ("what is running", "is h20 busy"): call `fleet_status()` or `fleet_status(hosts=["h20"])`.
   Read `summary`, then the per host `summary` lines; open `gpu[].procs` only when asked who is running what.
3. Unknown host names: call `list_hosts()` once and reuse the names.
4. No MCP server configured: run `mnvitop --once --json` (hosts as arguments if needed) and read
   `hosts[].gpus` and `hosts[].procs`; `mnvitop --once` prints the same as a table.

## Reading the result

- `busy` means a process holds the GPU or more than 5 percent of its VRAM is taken; `free` means neither.
- `vram_free_gb` is what a new job could take; a busy GPU with free VRAM is shareable, not free.
- `ok: false` or `age_s` above 30 means the numbers are not current: say so instead of guessing.
- `cmd` is truncated; `user` and `pid` identify a job.

## Rules

- Answer with the numbers that matter (host, GPU indices, free VRAM), not the whole JSON.
- These hosts are shared. Never stop, renice or otherwise touch processes you did not start.
- Prefer one host with all requested GPUs idle over splitting a job across hosts.
- Before launching a job, check again: a placement is only valid at the moment it was returned.
