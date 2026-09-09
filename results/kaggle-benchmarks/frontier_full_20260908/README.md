# Fresh frontier campaign

This directory records the fresh Kaggle Benchmark campaign started on 2026-09-08.
It is separate from the historical frontier artifacts.

## Admission

The private admission task completed 18 requested routes. Eight routes reached
the server's completed state, but the corresponding run metadata was unavailable
through the downloader at collection time. Ten routes errored. The downloaded
error manifests are retained below. No route from this admission batch is
promoted to behavioral evidence until its output and admission summary are
available locally.

## Behavioral baseline

The private confirmatory baseline task uses the frozen `ai-race-fairgame-v3`
protocol. The Gemini 3 Flash run is locally complete and validated: 30 races,
60 player rows, 558 decisions, and zero parse failures. Other requested routes
remain server-side until their run status and artifacts are collected.

The campaign is therefore an evidence container, not a result table. Downstream
figures and manuscript claims must be generated only from locally validated
manifests in this directory.
