"""Run labelled validation cases against one or more RAG configurations."""
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rag_pipeline import CURTRagPipeline

def evaluate_case(case, pipeline):
    try:
        result=pipeline.run(case["query"])
        raw=json.loads(result.get("raw_answer") or "{}")
        verdict_ok=raw.get("verdict") == case["expected_verdict"]
        citation_ok=(raw.get("cited_section") or "").upper() == case["expected_section"].upper()
        return {"id":case["id"], "verdict_correct":verdict_ok, "citation_correct":citation_ok,
                "answer":result["answer"], "expected":case}
    except Exception as error:
        return {"id":case["id"], "verdict_correct":False, "citation_correct":False,
                "error":str(error), "expected":case}

def score(cases, pipeline, workers):
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(evaluate_case, case, pipeline): case for case in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            result=future.result()
            results.append(result)
            print(f"[{index}/{len(cases)}] {result['id']}")
    results.sort(key=lambda result: result["id"])
    total=len(results) or 1
    return {"cases":results, "metrics":{"verdict_accuracy":sum(r["verdict_correct"] for r in results)/total,
            "citation_exact_match":sum(r["citation_correct"] for r in results)/total, "case_count":len(results)}}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--cases", default="../data/evaluation_cases.json")
    parser.add_argument("--experiments", default="../data/rag_experiments.json"); parser.add_argument("--output", default="../data/evaluation_results.json")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent LLM requests; increase only within API rate limits.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a quick check.")
    args=parser.parse_args(); base=Path(__file__).resolve().parent
    cases=json.loads((base/args.cases).resolve().read_text(encoding="utf-8"))
    if args.limit: cases=cases[:args.limit]
    experiments=json.loads((base/args.experiments).resolve().read_text(encoding="utf-8"))["experiments"]
    report={}
    project_root=base.parent
    for experiment in experiments:
        settings={k:v for k,v in experiment.items() if k != "name"}
        for key in ("chroma_dir", "bm25_path"):
            if key in settings: settings[key] = project_root / settings[key]
        print(f"Running {experiment['name']} with {len(cases)} cases and {args.workers} workers")
        report[experiment["name"]]=score(cases, CURTRagPipeline(**settings), args.workers)
    target=(base/args.output).resolve(); target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics to {target}")
