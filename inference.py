import os
import time
import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASKS = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5


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

    return best_ids


def run_task(task_id):
    print(f"[START] task={task_id} env=semantic-patch-triage model=knapsack", flush=True)
    score = 0.0
    steps = 0
    success = False

    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()["observation"]

        keep_ids = optimal_knapsack(obs["patches"], obs["token_budget"])

        result = requests.post(f"{ENV_URL}/step", json={"keep_patch_ids": keep_ids}, timeout=30)
        result.raise_for_status()
        result = result.json()

        reward = result.get("reward", 0.0)
        steps = 1
        score = round(min(max(reward, 0.0), 1.0), 4)
        success = score >= SUCCESS_THRESHOLD

        print(f"[STEP] step=1 action={keep_ids} reward={reward} done=true", flush=True)

    except Exception as e:
        print(f"[STEP] step=1 reward=0.0 done=true error={str(e)}", flush=True)

    print(f"[END] task={task_id} score={score} steps={steps} success={success}", flush=True)
    return score


def main():
    scores = {}
    for t in TASKS:
        print(f"\n{'='*40} {t}", flush=True)
        scores[t] = run_task(t)
        time.sleep(1)

    print("\nFINAL SCORES:", flush=True)
    for t, s in scores.items():
        print(f"  {t}: {s:.4f}", flush=True)
    print(f"  mean: {sum(scores.values())/len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()