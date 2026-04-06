
import os, json, time, requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def log_start(task, env, model):
    print(json.dumps({"event":"[START]","task":task,"env":env,"model":model}), flush=True)

def log_step(step, action, reward, done, error=None):
    print(json.dumps({"event":"[STEP]","step":step,"action":str(action)[:200],
                      "reward":reward,"done":done,"error":error}), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({"event":"[END]","success":success,"steps":steps,
                      "score":score,"rewards":rewards}), flush=True)

def get_action(patches, budget):
    lines = [f"ID={p['id']} type={p['type']} tokens={p['token_cost']} critical={p['is_critical']} | {p['payload']}"
             for p in patches]
    prompt = (f"Token budget: {budget}. Pick patches to KEEP. "
              f"Prioritise constraint and decision types. Stay within budget. "
              f"Return ONLY a JSON array of IDs like: [\"p1\",\"p2\"]\n\n" + "\n".join(lines))
    try:
        r = client.chat.completions.create(model=MODEL_NAME, max_tokens=150,
            messages=[{"role":"user","content":prompt}])
        t = r.choices[0].message.content.strip()
        s, e = t.find("["), t.rfind("]")+1
        if s != -1 and e > s:
            return json.loads(t[s:e])
    except Exception as ex:
        print(f"[DEBUG] LLM error: {ex}", flush=True)
    return [p["id"] for p in patches if p["is_critical"]]

def run_task(task_id):
    log_start(task=task_id, env="semantic-patch-triage", model=MODEL_NAME)
    rewards, steps_taken, score, success = [], 0, 0.0, False
    try:
        obs = requests.post(f"{ENV_URL}/reset", params={"task_id":task_id}, timeout=30).json()["observation"]
        keep_ids = get_action(obs["patches"], obs["token_budget"])
        result   = requests.post(f"{ENV_URL}/step", json={"keep_patch_ids":keep_ids}, timeout=30).json()
        reward   = result.get("reward", 0.0)
        rewards.append(reward)
        steps_taken = 1
        log_step(step=1, action=keep_ids, reward=reward, done=result.get("done",True), error=None)
        score   = round(min(max(reward, 0.0), 1.0), 4)
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
    for t,s in scores.items():
        print(f"  {t}: {s:.4f}")
    print(f"  mean: {sum(scores.values())/len(scores):.4f}")

if __name__ == "__main__":
    main()
