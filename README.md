# tei_rerank

The self-hosted TEI **reranker** service (ARCHITECTURE.md block 7) — decoupled like `../tei`, `../postgres` and `../qdrant`. The agent talks to `POST /rerank`.

This is a second TEI process, not a second model inside `../tei`: **TEI serves one model per process.** The plan originally said "on the same TEI deployment", which was never possible.

```bash
./build_model.sh          # once: downloads ~2.3 GB into model/bge-reranker-v2-m3
docker compose up -d

curl -s localhost:8081/health
curl -s localhost:8081/rerank -H 'Content-Type: application/json' \
  -d '{"query": "Zinssätze", "texts": ["Die EZB hat die Zinsen erhöht.", "Kochrezept"]}'
```

## Why rerank at all

Retrieval embeds the query and each passage *independently* and compares vectors, so nothing in the passage ever sees the query. A cross-encoder scores the pair jointly, letting the query attend to the passage token by token. That is why block 7 calls it the single biggest precision lever: retrieve top-30 hybrid, rerank, hand top-5 to the LLM.

The cost is that it is O(candidates), not O(1) — every passage is a full forward pass.

## Why there is a local model directory

Same two reasons as `../tei`: it caps `max_seq_length`, which is what bounds memory on the CPU backend, and it removes the startup weight download that `HF_HUB_OFFLINE=1` then enforces.

Three differences from the embedder's copy:

- **It downloads from the hub directly** (pinned revision) instead of scraping the cache a hub-backed TEI run left behind. `../tei/build_model.sh` bootstraps its cache by starting TEI **uncapped** — and this backbone uncapped is precisely the ~12 GB warmup block 5 documents, which is what the cap exists to prevent. Six files over HTTPS avoid triggering the bug in order to fix it, and pinning the revision makes a rebuild reproducible.
- **No ONNX export is published** — only `model.safetensors`, which TEI's CPU backend loads directly.
- **No `sentence_bert_config.json` ships at all**, so `build_model.sh` *writes* one instead of overwriting it. Without that file TEI falls back to `config.json`'s `max_position_embeddings` (8194) and the cap silently never applies. Confirm it took:

```bash
docker logs tei_rerank-tei-rerank-1 | grep -i 'max'
```

The cap is **1536**, and unlike the embedder that budget covers *both* halves — the reranker tokenizes query and passage together. The chunker's longest measured output is 1257 tokens, leaving ~280 for the query.

## Memory

Same XLM-R-large backbone as bge-m3 (24 layers, 16 heads, 1024 hidden, upstream limit 8192), so block 5's formula carries over unchanged:

```
weights + seqs_per_pass x max_seq_length^2 x n_heads x 4 bytes
seqs_per_pass = min(max_batch_tokens / max_seq_length, max_batch_requests)
```

The one deliberate difference from `../tei` is `--max-batch-tokens 8192` (TEI's default is 16384). Query-time load is ~30 passages, not thousand-chunk batches, so halving it costs no throughput here and buys back memory:

```
seqs_per_pass = 8192 / 1536 = 5
5 x 1536^2 x 16 x 4 B ~= 755 MB  over ~2.3 GB of weights  ~= 3 GB
```

against bge-m3's ~4.7 GB. `mem_limit: 6g` is the safety net around that. ONNX Runtime's arena only ever grows, so measure after real traffic, not right after warmup.

## Dev-box reality (measured 2026-07-31)

It is **correct but slow here**. Ranking quality is right — a German query scored an English passage above its German paraphrase and gave the irrelevant one 0.0:

```
"Zinssätze der EZB"
  0.6223  The ECB raised interest rates.
  0.5618  Die EZB hat die Leitzinsen erhöht.
  0.0000  Ein Kochrezept für Kuchen.
```

Latency is the problem:

| Workload | Measured |
|---|---|
| Hybrid retrieval, no rerank, 30 chunks | **0.68 s** |
| Rerank 5 passages (~3,500 tok) | **222 s** (~16 tok/s) |
| Rerank 30 passages (~16,700 tok) | ~17 min, extrapolated |

So `RAG_RERANK_ENABLED` is `false` in `../agent/docker-compose.yml`; the agent's `--rerank` flag turns it on per query.

**The cause is known and fixable — it is the backend, not the hardware.** TEI prefers ONNX Runtime and falls back to Candle:

```
ERROR Could not start ORT backend: File at `/model/bge-reranker-v2-m3/onnx/model.onnx` does not exist
INFO  Starting Bert model on Cpu
WARN  Backend does not support a batch size > 4
```

`BAAI/bge-reranker-v2-m3` publishes no ONNX export, where `BAAI/bge-m3` does — which is why the embedder manages ~100–160 tok/s on the same box and the same backbone, an ~8x gap, and why Candle also caps the batch at 4 against the embedder's 8. Supplying an ONNX export (a community one, or exported with `optimum`) is the obvious first optimization; **deliberately not done yet** — correctness first, tuning once the pipeline is exercised end to end.

The cluster budgets 8 cores per TEI replica and block 11 names a GPU as the escape hatch, so this is a dev-box story either way.

## Notes

- Running both TEIs at once costs ~4.7 GB (`../tei`) + ~3 GB here. On a 15 GB box with an IDE open that is tight — check `free -h` first.
- Swapping the reranker model does **not** invalidate the Qdrant collection: unlike the embedder, nothing it produces is stored. Reranking is pure query-time compute, so the model can change whenever.
- On the cluster this becomes a standing Nomad service alongside the embedding TEI (block 11's resource plan: 4 cores / 8 GB).
