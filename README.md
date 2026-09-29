# Papers — Replicações bilíngues de papers de memória de agentes
# Bilingual replications of agent-memory papers

Cada pasta `papers/pNN_*` replica o **mecanismo** de um paper, rodando offline
com um `MockLLM` determinístico (sem chave de API). Os resultados demonstram o
mecanismo — **não** reproduzem os números dos papers. Leia
`docs/ARQUITETURA_ARCHITECTURE.md` para a regra de honestidade e como trocar
para um LLM real (`PAPERS_LLM=openai`).

Each `papers/pNN_*` folder replicates a paper's **mechanism**, running offline
with a deterministic `MockLLM` (no API key). Results demonstrate the mechanism —
they do **not** reproduce the papers' numbers. See
`docs/ARQUITETURA_ARCHITECTURE.md` for the honesty rule and how to switch to a
real LLM (`PAPERS_LLM=openai`).

```bash
pip install -e ".[dev]"
pytest -q
python -m papers.p02_reflexion.run --write-results
```

## Índice / Index

| pasta / folder | paper | status |
|---|---|---|
| `p01_clin` | CLIN (2310.10134) | ✅ |
| `p02_reflexion` | Reflexion (2303.11366) | ✅ |
| `p03_memrl` | MemRL (2601.03192) | ✅ |
| `p04_muse` | MUSE (2411.13537) | ✅ |
| `p05_cer` | CER (2506.06698) | ✅ |
| `p06_voyager` | Voyager (2305.16291) | ✅ |
| `p07_muse_autoskill` | MUSE-Autoskill (2605.27366) | ✅ |
| `p08_coala` | CoALA (2309.02427) | ✅ |
| `p09_generative_agents` | Generative Agents (2304.03442) | em breve / coming soon |
| `p10_memgpt` | MemGPT (2310.08560) | em breve / coming soon |
| `p11_a_mem` | A-MEM (2502.12110) | em breve / coming soon |
| `p12_zep` | Zep/Graphiti (2501.13956) | em breve / coming soon |
| `p13_mem0` | Mem0 (2504.19413) | em breve / coming soon |
| `p14_memory_r1` | Memory-R1 (2508.19828) | em breve / coming soon |
| `p15_longmemeval` | LongMemEval (2410.10813) | em breve / coming soon |
| `p16_memoryagentbench` | MemoryAgentBench (2507.05257) | em breve / coming soon |
| `p17_longmemeval_v2` | LongMemEval-V2 (2605.12493) | em breve / coming soon |
| `p18_survey` | Survey (2512.13564) | em breve / coming soon |
