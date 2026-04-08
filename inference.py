import os
import json
import time
import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5


def log_start(task, env, model):
    print(json.dumps({"event": "[START]", "task": task, "env": env, "model": model}), flush=True)


def log_step(step, action, reward, done, error=None):
    print(json.dumps({"event": "[STEP]", "step": step, "action": str(action)[:200],
                      "reward": reward, "done": done, "error": error}), flush=True)


def log_end(success, steps, score, rewards):
    print(json.dumps({"event": "[END]", "success": success, "steps": steps,
                      "score": score, "rewards": rewards}), flush=True)


def optimal_knapsack(patches, budget):
    n = len(patches)
    total_tokens = sum(p["token_cost"] for p in patches)
    num_crit = len([p for p in patches if p["is_critical"]])

    best_score = -1
    best_ids = []

    for mask in range(1 << n):
        kept = [patches[i] for i in range(n) if mask & (1 << i)]
        used = sum(p["token_cost"] for p in kept)

        if used > budget:
            continue

        kept_crit = len([p for p in kept if p["is_critical"]])
        crr = kept_crit / num_crit if num_crit else 1.0
        trr = max(0.0, 1.0 - used / total_tokens) if total_tokens else 0.0
        score = crr * 0.6 + trr * 0.4

        if score > best_score:
            best_score = score
            best_ids = [p["id"] for p in kept]

    print(f"[DEBUG] Knapsack best score: {best_score:.4f}, keeping: {best_ids}", flush=True)
    return best_ids


def run_task(task_id):
    log_start(task=task_id, env="semantic-patch-triage", model="knapsack")
    rewards, steps_taken, score, success = [], 0, 0.0, False

    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()["observation"]

        keep_ids = optimal_knapsack(obs["patches"], obs["token_budget"])

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