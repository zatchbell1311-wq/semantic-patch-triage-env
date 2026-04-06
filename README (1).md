# Semantic Patch Triage — OpenEnv Environment

## Description
Agent learns to triage semantic memory patches under token budget constraints.
Based on the DSPM paper (Dubey, 2026) for LLM context compression.

## Tasks
| Task   | Patches | Budget | Critical |
|--------|---------|--------|----------|
| easy   | 6       | 150    | 3        |
| medium | 10      | 100    | 5        |
| hard   | 15      | 80     | 8        |

## Reward
`(CRR * 0.6) + (TRR * 0.4) - 0.3 if over budget`

## Setup
```bash
docker build -t spt-env .
docker run -p 7860:7860 spt-env
```

## Endpoints
- POST /reset?task_id=easy
- POST /step
- GET  /state
- GET  /health
