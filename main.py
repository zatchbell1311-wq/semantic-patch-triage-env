
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="Semantic Patch Triage", version="1.0.0")

class Patch(BaseModel):
    id: str
    type: str
    payload: str
    token_cost: int
    is_critical: bool

class Observation(BaseModel):
    patches: List[Patch]
    token_budget: int
    total_tokens: int
    turn: int
    task_id: str
    instruction: str

class Action(BaseModel):
    keep_patch_ids: List[str]

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: Observation

TASKS = {
    "easy": {
        "token_budget": 150,
        "patches": [
            {"id":"p1","type":"constraint","payload":"Response latency must be under 200ms","token_cost":12,"is_critical":True},
            {"id":"p2","type":"decision","payload":"Use Redis for caching instead of Memcached","token_cost":11,"is_critical":True},
            {"id":"p3","type":"code","payload":"r = redis.Redis(host=localhost, port=6379)","token_cost":18,"is_critical":False},
            {"id":"p4","type":"entity","payload":"Team lead is Priya, backend dev is Arjun","token_cost":10,"is_critical":False},
            {"id":"p5","type":"constraint","payload":"System must handle 10000 concurrent users","token_cost":11,"is_critical":True},
            {"id":"p6","type":"structure","payload":"Architecture follows microservices with API gateway","token_cost":13,"is_critical":False},
        ]
    },
    "medium": {
        "token_budget": 100,
        "patches": [
            {"id":"p1","type":"constraint","payload":"API rate limit is 1000 requests per minute","token_cost":13,"is_critical":True},
            {"id":"p2","type":"decision","payload":"Database is PostgreSQL 15 with connection pooling","token_cost":12,"is_critical":True},
            {"id":"p3","type":"code","payload":"SELECT * FROM users WHERE created_at > NOW() - INTERVAL 7 days","token_cost":20,"is_critical":False},
            {"id":"p4","type":"entity","payload":"Primary stakeholder is the CTO office","token_cost":9,"is_critical":False},
            {"id":"p5","type":"constraint","payload":"Data must be encrypted at rest using AES-256","token_cost":11,"is_critical":True},
            {"id":"p6","type":"decision","payload":"Auth uses JWT tokens with 1 hour expiry","token_cost":13,"is_critical":True},
            {"id":"p7","type":"structure","payload":"Services communicate via REST over internal VPC","token_cost":13,"is_critical":False},
            {"id":"p8","type":"equation","payload":"Cache hit ratio target >= 0.85","token_cost":17,"is_critical":False},
            {"id":"p9","type":"entity","payload":"Deployment is on AWS us-east-1","token_cost":10,"is_critical":False},
            {"id":"p10","type":"constraint","payload":"Zero downtime deployment required","token_cost":11,"is_critical":True},
        ]
    },
    "hard": {
        "token_budget": 80,
        "patches": [
            {"id":"p1","type":"constraint","payload":"ML inference must complete within 50ms p99","token_cost":14,"is_critical":True},
            {"id":"p2","type":"decision","payload":"Use ONNX Runtime instead of TorchServe","token_cost":13,"is_critical":True},
            {"id":"p3","type":"code","payload":"session = ort.InferenceSession(model.onnx)","token_cost":22,"is_critical":False},
            {"id":"p4","type":"constraint","payload":"Model accuracy must stay above 94.5 percent F1","token_cost":15,"is_critical":True},
            {"id":"p5","type":"decision","payload":"Feature store uses Feast with Redis online store","token_cost":12,"is_critical":True},
            {"id":"p6","type":"equation","payload":"Drift threshold: KL divergence > 0.1","token_cost":18,"is_critical":False},
            {"id":"p7","type":"structure","payload":"Pipeline: ingest preprocess extract infer postprocess","token_cost":18,"is_critical":False},
            {"id":"p8","type":"constraint","payload":"Training data cannot include PII without consent","token_cost":14,"is_critical":True},
            {"id":"p9","type":"entity","payload":"Model v2.3.1 is current production checkpoint","token_cost":11,"is_critical":False},
            {"id":"p10","type":"decision","payload":"Shadow deployment for AB testing new models","token_cost":14,"is_critical":True},
            {"id":"p11","type":"code","payload":"outputs = model.run(None, input_dict)","token_cost":21,"is_critical":False},
            {"id":"p12","type":"structure","payload":"Monitoring: Prometheus metrics to Grafana","token_cost":14,"is_critical":False},
            {"id":"p13","type":"constraint","payload":"GPU memory must not exceed 80 percent VRAM","token_cost":13,"is_critical":True},
            {"id":"p14","type":"entity","payload":"Infra managed by ML platform team on Slack","token_cost":16,"is_critical":False},
            {"id":"p15","type":"decision","payload":"Canary rollout at 5 percent before full promotion","token_cost":13,"is_critical":True},
        ]
    }
}

_state = {"task_id":"easy","patches":[],"token_budget":0,"turn":0,"done":False}
INSTRUCTION = "You are a memory manager. Pick patches to keep. Prioritise constraint and decision types. Stay within token budget."

def make_obs(task_id, patches, budget, turn):
    return Observation(patches=[Patch(**p) for p in patches],
        token_budget=budget, total_tokens=sum(p["token_cost"] for p in patches),
        turn=turn, task_id=task_id, instruction=INSTRUCTION)

def compute_reward(kept_ids, patches, budget):
    pm = {p["id"]:p for p in patches}
    kept = [pm[i] for i in kept_ids if i in pm]
    crit = [p for p in patches if p["is_critical"]]
    kept_crit = [p for p in kept if p["is_critical"]]
    crr = len(kept_crit)/len(crit) if crit else 1.0
    total = sum(p["token_cost"] for p in patches)
    kept_tok = sum(p["token_cost"] for p in kept)
    trr = max(0.0, 1.0 - kept_tok/total) if total else 0.0
    penalty = 0.3 if kept_tok > budget else 0.0
    reward = round(max(0.0, min(1.0, crr*0.6 + trr*0.4 - penalty)), 4)
    return reward, crr, trr, kept_tok

@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = Query(default="easy")):
    task = TASKS.get(task_id, TASKS["easy"])
    _state.update({"task_id":task_id,"patches":task["patches"],
                   "token_budget":task["token_budget"],"turn":0,"done":False})
    return ResetResult(observation=make_obs(task_id, task["patches"], task["token_budget"], 0))

@app.post("/step", response_model=StepResult)
def step(action: Action):
    if _state["done"]:
        return StepResult(observation=make_obs(_state["task_id"],_state["patches"],
            _state["token_budget"],_state["turn"]), reward=0.0, done=True,
            info={"message":"already done"})
    _state["turn"] += 1
    _state["done"] = True
    reward, crr, trr, kt = compute_reward(action.keep_patch_ids, _state["patches"], _state["token_budget"])
    return StepResult(
        observation=make_obs(_state["task_id"],_state["patches"],_state["token_budget"],_state["turn"]),
        reward=reward, done=True,
        info={"crr":round(crr,4),"trr":round(trr,4),"kept_tokens":kt,"budget":_state["token_budget"]})

@app.get("/state")
def state():
    return _state

@app.get("/health")
def health():
    return {"status":"ok","version":"1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
