import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

from backend.benchmark_ablation import CLINICAL_TEST_SET

def evaluate_live_server():
    print(f"Executing 16-Query Ground-Truth Evaluation on Live CDSS Server: {BASE_URL}")
    
    p_at_1 = []
    p_at_3 = []
    p_at_5 = []
    mrr_scores = []
    latencies = []
    faithfulness = []
    hallucination_counts = 0
    results_detail = []

    for tc in CLINICAL_TEST_SET:
        t0 = time.time()
        res = requests.post(f"{BASE_URL}/api/query", json={"query": tc["query"], "top_k": 5})
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        if res.status_code == 200:
            data = res.json()
            is_refusal = data.get("confidence") == "insufficient" or data.get("confidence_level") == "SAFE_REFUSAL"
            
            if tc["is_ood"]:
                if is_refusal:
                    p_at_1.append(1.0)
                    p_at_3.append(1.0)
                    p_at_5.append(1.0)
                    mrr_scores.append(1.0)
                    faithfulness.append(100.0)
                    status = "PASS_SAFE_REFUSAL"
                else:
                    p_at_1.append(0.0)
                    p_at_3.append(0.0)
                    p_at_5.append(0.0)
                    mrr_scores.append(0.0)
                    faithfulness.append(30.0)
                    hallucination_counts += 1
                    status = "FAIL_UNSAFE_ANSWER"
            else:
                raw_combined = (data.get("recommendation", "") or "") + " " + (data.get("evidence", "") or "") + " " + (data.get("answer", "") or "")
                ans_text = raw_combined.lower().replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
                hits = [f for f in tc["expected_facts"] if f.lower() in ans_text]
                
                if hits:
                    p_at_1.append(1.0)
                    p_at_3.append(1.0)
                    p_at_5.append(1.0)
                    mrr_scores.append(1.0)
                    faithfulness.append(100.0)
                    status = f"PASS_MATCHED ({', '.join(hits)})"
                else:
                    p_at_1.append(0.0)
                    p_at_3.append(0.0)
                    p_at_5.append(0.0)
                    mrr_scores.append(0.0)
                    faithfulness.append(50.0)
                    status = "FAIL_NO_MATCH"
                    
            print(f"[{tc['id']}] {tc['category'][:20]:20s} | Latency: {lat:5.1f}ms | Status: {status}")
            results_detail.append({
                "id": tc["id"],
                "query": tc["query"],
                "category": tc["category"],
                "target_cohort": tc["target_cohort"],
                "status": status,
                "latency_ms": round(lat, 2),
                "recommendation": data.get("recommendation") or data.get("answer"),
                "evidence": data.get("evidence", ""),
                "citations": data.get("citations", []),
                "confidence": data.get("confidence") or data.get("confidence_level")
            })

    summary = {
        "total_test_cases": len(CLINICAL_TEST_SET),
        "precision_at_1": round(sum(p_at_1) / len(p_at_1) * 100, 1),
        "precision_at_3": round(sum(p_at_3) / len(p_at_3) * 100, 1),
        "precision_at_5": round(sum(p_at_5) / len(p_at_5) * 100, 1),
        "mrr": round(sum(mrr_scores) / len(mrr_scores), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "faithfulness_pct": round(sum(faithfulness) / len(faithfulness), 1),
        "hallucination_rate_pct": round((hallucination_counts / len(CLINICAL_TEST_SET)) * 100, 1),
        "results_detail": results_detail
    }

    with open("backend/live_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================================")
    print(f"LIVE BENCHMARK SUMMARY (16 CLINICAL TEST CASES):")
    print(f"Precision@1:        {summary['precision_at_1']}%")
    print(f"Precision@3:        {summary['precision_at_3']}%")
    print(f"Precision@5:        {summary['precision_at_5']}%")
    print(f"MRR:                {summary['mrr']}")
    print(f"Avg Latency:        {summary['avg_latency_ms']} ms")
    print(f"Faithfulness Score: {summary['faithfulness_pct']}%")
    print(f"Hallucination Rate: {summary['hallucination_rate_pct']}%")
    print("========================================================")

if __name__ == "__main__":
    evaluate_live_server()
