import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

# Safe client init — won't crash if token is missing
try:
    client = OpenAI(api_key=HF_TOKEN if HF_TOKEN else "dummy", base_url=API_BASE_URL)
    USE_LLM = bool(HF_TOKEN)
except Exception as ex:
    print(f"[DEBUG] OpenAI client init failed: {ex}", flush=True)
    client = None
    USE_LLM = False

def log_start(task, env, model):
    print(json.dumps({"event":"[START]","task":task,"env":env,"model":model}), flush=True)

def log_step(step, action, reward, done, error=None):
    print(json.dumps({"event":"[STEP]","step":step,"action":str(action)[:200],
                      "reward":reward,"done":done,"error":error}), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({"event":"[END]","success":success,"steps":steps,
                      "score":score,"rewards":rewards}), flush=True)

def smart_greedy_action(patches, budget):
    """
    Greedy algorithm — no LLM needed:
    1. Always keep critical patches first (sorted by token_cost ascending)
    2. Then fill remaining budget with constraint > decision > others
    """
    critical = [p for p in patches if p["is_critical"]]
    non_critical = [p for p in patches if not p["is_critical"]]

    # Sort critical by token cost (cheapest first to fit more)
    critical.sort(key=lambda p: p["token_cost"])

    # Sort non-critical by priority: constraint > decision > rest, then cheapest
    type_priority = {"constraint": 0, "decision": 1, "equation": 2, "structure": 3, "entity": 4, "code": 5}
    non_critical.sort(key=lambda p: (type_priority.get(p["type"], 9), p["token_cost"]))

    kept = []
    used_tokens = 0

    # First pass: keep all critical patches that fit
    for p in critical:
        if used_tokens + p["token_cost"] <= budget:
            kept.append(p)
            used_tokens += p["token_cost"]

    # Second pass: fill with non-critical if budget allows
    for p in non_critical:
        if used_tokens + p["token_cost"] <= budget:
            kept.append(p)
            used_tokens += p["token_cost"]

    return [p["id"] for p in kept]

def get_action_llm(patches, budget):
    """Try LLM, fall back to greedy if it fails."""
    lines = [
        f"ID={p['id']} type={p['type']} tokens={p['token_cost']} critical={p['is_critical']} | {p['payload']}"
        for p in patches
    ]
    prompt = (
        f"You are a memory manager for an AI agent.\n"
        f"Token budget remaining: {budget}\n"
        f"You MUST stay within the token budget (sum of token_cost of kept patches <= {budget}).\n"
        f"Priority order: keep is_critical=True patches first, then constraint type, then decision type.\n"
        f"Return ONLY a valid JSON array of patch IDs to keep. Example: [\"p1\",\"p3\"]\n\n"
        f"Patches:\n" + "\n".join(lines)
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=200,
            messages=[{"role":"user","content":prompt}]
        )
        t = r.choices[0].message.content.strip()
        s, e = t.find("["), t.rfind("]") + 1
        if s != -1 and e > s:
            ids = json.loads(t[s:e])
            # Validate: check budget not exceeded
            id_set = {p["id"]: p for p in patches}
            total = sum(id_set[i]["token_cost"] for i in ids if i in id_set)
            if total <= budget:
                return ids
            else:
                # LLM exceeded budget — fall back to greedy
                print("[DEBUG] LLM exceeded budget, using greedy fallback", flush=True)
                return smart_greedy_action(patches, budget)
    except Exception as ex:
        print(f"[DEBUG] LLM error: {ex}", flush=True)
    return smart_greedy_action(patches, budget)

def get_action(patches, budget):
    if USE_LLM and client:
        return get_action_llm(patches, budget)
    return smart_greedy_action(patches, budget)

def run_task(task_id):
    log_start(task=task_id, env="semantic-patch-triage", model=MODEL_NAME)
    rewards, steps_taken, score, success = [], 0, 0.0, False
    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()["observation"]

        keep_ids = get_action(obs["patches"], obs["token_budget"])

        result = requests.post(f"{ENV_URL}/step", json={"keep_patch_ids": keep_ids}, timeout=30)
        result.raise_for_status()
        result = result.json()

        reward = result.get("reward", 0.0)
        rewards.append(reward)
        steps_taken = 1
        log_step(step=1, action=keep_ids, reward=reward, done=result.get("done", True), error=None)
        score = round(min(max(reward, 0.0), 1.0), 4)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(step=1, action=[], reward=0.0, done=True, error=str(e))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

def main():
    scores = {}
    for t in TASKS:
        print(f"\n{'='*40} {t}", flush=True)
        scores[t] = run_task(t)
        time.sleep(1)
    print("\nFINAL SCORES:")
    for t, s in scores.items():
        print(f"  {t}: {s:.4f}")
    print(f"  mean: {sum(scores.values())/len(scores):.4f}")

if __name__ == "__main__":
    main()