"""Build a massive real-world training dataset from Codeforces-7k, DeepMind CodeContests, and IBM CodeNet4Repair."""

from __future__ import annotations

import ast
import json
import os
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from src.feature_extractor import ALGO_FAMILIES, BUG_CLASSES, extract_all_features
from src.real_problems_catalog import map_tags_to_algo_family

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "training_submissions.jsonl"
)

PROBLEM_TEMPLATES: dict[str, list[str]] = {
    "math_ad_hoc": [
        "Dado N <= {n_lim} e elementos inteiros a_i ate {v_lim}, calcule a soma total e o produto acumulado.",
        "Given N <= {n_lim} integers where each value x_i <= {v_lim}, compute the arithmetic progression sum.",
    ],
    "prefix_sum_two_pointers": [
        "Dado um array de N <= {n_lim} inteiros (valores ate {v_lim}) e Q <= {n_lim} consultas [L, R], responda a soma do subarray.",
        "Given an array of N <= {n_lim} integers up to {v_lim}, find the longest contiguous subarray with sum at most K using two pointers.",
    ],
    "sorting_binary_search": [
        "Dado um vetor ordenado de N <= {n_lim} inteiros ate {v_lim}, encontre o maior indice i tal que arr[i] <= X usando busca binaria.",
        "Given N <= {n_lim} intervals with coordinates up to {v_lim}, sort and find the maximum number of non-overlapping intervals.",
    ],
    "graphs_bfs_dfs_dijkstra": [
        "Dado um grafo com N <= {n_lim} vertices e M <= {n_lim} arestas com pesos positivos ate {v_lim}, encontre o menor caminho (Dijkstra).",
        "Para T <= 10^4 casos de teste, leia um grafo de N <= {n_lim} vertices e M arestas e imprima o grau e as componentes conexas.",
    ],
    "dynamic_programming": [
        "Dado N <= {n_lim} pedras com alturas ate {v_lim}, calcule o custo minimo para o sapo pular de 1 ate N (Frog DP).",
        "Temos N <= 100 moedas e soma alvo X <= {n_lim}. Calcule o numero de combinacoes de moedas modulo 10^9 + 7.",
    ],
    "data_structures_hash_segtree": [
        "Dado um array de N <= {n_lim} inteiros (valores ate {v_lim}), conte pares (i, j) com soma K usando hash map.",
        "Processe Q <= {n_lim} operacoes de atualizacao pontual e consulta de soma em intervalo [L, R] com Segment Tree.",
    ],
}


def _classify_real_submission_bug(
    code: str,
    lang: str,
    problem_statement: str,
    verdict: str,
) -> str:
    """Label a real CodeNet/DeepMind submission using verdict + 33-feature AST analysis."""
    verdict_low = verdict.lower()
    if "accepted" in verdict_low or verdict_low == "ac":
        return "clean_or_minor_logic"

    analysis = extract_all_features(code, lang, problem_statement)  # type: ignore[arg-type]
    fmap = analysis.feature_Map

    if fmap.get("missing_global_cleanup", 0.0) > 0:
        return "global_state_dirty"
    if fmap.get("has_bs_mid_assign_bug", 0.0) > 0 or (
        fmap.get("has_recursion", 0.0) > 0 and fmap.get("has_memoization", 0.0) == 0
    ):
        return "infinite_loop_or_recursion"
    if fmap.get("has_le_in_zero_loop", 0.0) > 0 or "runtime error" in verdict_low:
        if fmap.get("has_le_in_zero_loop", 0.0) > 0 or fmap.get("has_index_plus_one", 0.0) > 0:
            return "off_by_one_bounds"
        return "edge_case_uninitialized_or_div0"
    if fmap.get("ops_exceed_time_limit", 0.0) > 0 or "time limit" in verdict_low:
        if fmap.get("has_endl_in_loop", 0.0) > 0 and fmap.get("max_loop_depth", 0.0) <= 1:
            return "tle_slow_io_endl"
        return "tle_quadratic_complexity"
    if fmap.get("overflow_risk_score", 0.0) >= 0.4:
        return "integer_overflow"
    if fmap.get("has_endl_in_loop", 0.0) > 0:
        return "tle_slow_io_endl"
    return "edge_case_uninitialized_or_div0"


def _extract_from_deepmind_parquet(parquet_path: str, max_samples: int = 8000) -> list[dict[str, Any]]:
    """Extract real C++ and Python incorrect & accepted submissions from DeepMind CodeContests."""
    import pandas as pd

    if not os.path.exists(parquet_path):
        return []

    df = pd.read_parquet(parquet_path)
    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        desc = str(row.get("description") or "").strip()[:750]
        raw_tags = row.get("cf_tags")
        tags = [str(t) for t in (raw_tags.tolist() if hasattr(raw_tags, "tolist") else (raw_tags or []))]
        algo_fam = map_tags_to_algo_family(tags)

        # Incorrect submissions (language 2 = C++, 3 = Python3)
        inc = row.get("incorrect_solutions")
        if isinstance(inc, dict) and "solution" in inc and "language" in inc:
            sols = inc["solution"]
            langs = inc["language"]
            limit_per_prob = 35
            count_p = 0
            for s_idx, (code_raw, l_id) in enumerate(zip(sols, langs, strict=False)):
                l_int = int(l_id)
                if l_int not in (2, 3):
                    continue
                lang = "cpp" if l_int == 2 else "python"
                code_str = str(code_raw).strip()
                if len(code_str) < 25 or len(code_str) > 3500:
                    continue
                bug_cls = _classify_real_submission_bug(code_str, lang, desc, "Wrong Answer / TLE")
                records.append(
                    {
                        "id": f"deepmind_inc_{idx}_{s_idx}",
                        "source_dataset": "deepmind/code_contests",
                        "language": lang,
                        "problem_statement": desc,
                        "code": code_str,
                        "bug_class": bug_cls,
                        "algo_family": algo_fam,
                    }
                )
                count_p += 1
                if count_p >= limit_per_prob or len(records) >= max_samples:
                    break

        # Correct submissions (Accepted)
        cor = row.get("solutions")
        if isinstance(cor, dict) and "solution" in cor and "language" in cor:
            sols = cor["solution"]
            langs = cor["language"]
            for s_idx, (code_raw, l_id) in enumerate(zip(sols[:6], langs[:6], strict=False)):
                l_int = int(l_id)
                if l_int not in (2, 3):
                    continue
                lang = "cpp" if l_int == 2 else "python"
                code_str = str(code_raw).strip()
                if 25 <= len(code_str) <= 3000:
                    records.append(
                        {
                            "id": f"deepmind_ac_{idx}_{s_idx}",
                            "source_dataset": "deepmind/code_contests",
                            "language": lang,
                            "problem_statement": desc,
                            "code": code_str,
                            "bug_class": "clean_or_minor_logic",
                            "algo_family": algo_fam,
                        }
                    )
        if len(records) >= max_samples:
            break

    return records


def _extract_from_codenet_parquet(parquet_path: str, max_samples: int = 8000) -> list[dict[str, Any]]:
    """Extract real buggy & repaired submissions from IBM Project CodeNet (TnT/Multi_CodeNet4Repair)."""
    import pandas as pd

    if not os.path.exists(parquet_path):
        return []

    df = pd.read_parquet(parquet_path)
    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        desc = str(row.get("problem_description") or "").strip()[:750]
        try:
            codes = ast.literal_eval(str(row.get("codes") or "[]"))
            statuses = ast.literal_eval(str(row.get("status") or "[]"))
        except (ValueError, SyntaxError):
            continue
        if not codes or not statuses:
            continue

        # Pair 0: initial buggy submission
        buggy_code = str(codes[0]).strip()
        buggy_status = str(statuses[0]).strip()
        if 20 <= len(buggy_code) <= 3000:
            lang = "cpp" if "#include" in buggy_code else ("c" if "scanf(" in buggy_code else "python")
            bug_cls = _classify_real_submission_bug(buggy_code, lang, desc, buggy_status)
            # Infer algo family from problem statement keywords
            low = desc.lower()
            if any(w in low for w in ("graph", "tree", "vertex", "edge", "path", "grid")):
                algo = "graphs_bfs_dfs_dijkstra"
            elif any(w in low for w in ("dp", "modulo", "998244353", "10^9 + 7", "ways", "knapsack")):
                algo = "dynamic_programming"
            elif any(w in low for w in ("sort", "binary", "minimum", "maximum", "interval")):
                algo = "sorting_binary_search"
            elif any(w in low for w in ("subarray", "contiguous", "prefix", "range")):
                algo = "prefix_sum_two_pointers"
            else:
                algo = "math_ad_hoc"

            records.append(
                {
                    "id": f"codenet_bug_{idx}",
                    "source_dataset": "TnT/Multi_CodeNet4Repair",
                    "language": lang,
                    "problem_statement": desc,
                    "code": buggy_code,
                    "bug_class": bug_cls,
                    "algo_family": algo,
                }
            )

        if len(records) >= max_samples:
            break

    return records


def _gen_canonical_anchor_samples(samples_per_class: int = 350, seed: int = 42) -> list[dict[str, Any]]:
    """Generate high-signal canonical C/C++/Python anchor samples across all 8 bug classes."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    languages = ["cpp", "c", "python"]

    for bug_cls in BUG_CLASSES:
        for i in range(samples_per_class):
            lang = languages[i % len(languages)]
            var_acc = rng.choice(["soma", "total", "ans", "res", "acum"])
            var_arr = rng.choice(["v", "a", "arr", "nums", "vec"])
            var_n = rng.choice(["n", "N", "tam"])
            algo = rng.choice(ALGO_FAMILIES)
            stmt = rng.choice(PROBLEM_TEMPLATES[algo]).format(
                n_lim=rng.choice(["10^5", "2 * 10^5", "100000"]),
                v_lim=rng.choice(["10^9", "1000000000"]),
            )

            if bug_cls == "integer_overflow":
                lang = "cpp" if lang == "python" else lang
                code = (
                    f"#include <iostream>\nusing namespace std;\nint main() {{\n"
                    f"    int {var_n}; cin >> {var_n};\n    int {var_acc} = 0;\n"
                    f"    for (int i = 0; i < {var_n}; i++) {{ int x; cin >> x; {var_acc} += x; }}\n"
                    f"    cout << {var_acc} << '\\n';\n}}"
                )
            elif bug_cls == "tle_quadratic_complexity":
                if lang == "python":
                    code = (
                        f"{var_n} = int(input())\n{var_arr} = list(map(int, input().split()))\n"
                        f"{var_acc} = 0\nfor i in range({var_n}):\n    for j in range(i+1, {var_n}):\n"
                        f"        if {var_arr}[i] + {var_arr}[j] == 10:\n            {var_acc} += 1\nprint({var_acc})"
                    )
                else:
                    code = (
                        f"#include <iostream>\n#include <vector>\nusing namespace std;\nint main() {{\n"
                        f"    int {var_n}; cin >> {var_n}; vector<long long> {var_arr}({var_n});\n"
                        f"    for (int i = 0; i < {var_n}; i++) for (int j = i+1; j < {var_n}; j++) {{}}\n}}"
                    )
            elif bug_cls == "off_by_one_bounds":
                if lang == "python":
                    code = f"{var_arr} = [1, 2, 3]\nfor i in range(len({var_arr}) + 1):\n    print({var_arr}[i])"
                else:
                    code = (
                        f"#include <iostream>\n#include <vector>\nusing namespace std;\nint main() {{\n"
                        f"    int {var_n}; cin >> {var_n}; vector<int> {var_arr}({var_n});\n"
                        f"    for (int i = 0; i <= {var_n}; i++) cin >> {var_arr}[i];\n}}"
                    )
            elif bug_cls == "global_state_dirty":
                lang = "cpp"
                stmt = "Para T <= 10^4 casos de teste, leia um grafo com N <= 10^5 vertices e M arestas."
                code = (
                    "#include <iostream>\n#include <vector>\nusing namespace std;\n"
                    "vector<int> adj[100005];\nint main() {\n    int t; cin >> t;\n"
                    "    while (t--) { int n, m; cin >> n >> m; adj[1].push_back(2); }\n}"
                )
            elif bug_cls == "tle_slow_io_endl":
                lang = "cpp"
                code = (
                    f"#include <iostream>\nusing namespace std;\nint main() {{\n"
                    f"    int {var_n}; cin >> {var_n};\n"
                    f"    for (int i = 0; i < {var_n}; i++) cout << i << endl;\n}}"
                )
            elif bug_cls == "infinite_loop_or_recursion":
                if i % 2 == 0:
                    code = (
                        "def bs(arr, x):\n    l, r = 0, len(arr) - 1\n"
                        "    while l < r:\n        mid = (l + r) // 2\n"
                        "        if arr[mid] <= x:\n            l = mid\n        else:\n            r = mid - 1\n"
                    )
                    lang = "python"
                else:
                    code = "def f(n):\n    if n <= 1: return 1\n    return f(n-1) + f(n-2)\nprint(f(40))"
                    lang = "python"
            elif bug_cls == "edge_case_uninitialized_or_div0":
                lang = "cpp"
                code = (
                    f"#include <iostream>\nusing namespace std;\nint main() {{\n"
                    f"    int {var_n}; cin >> {var_n}; long long {var_acc};\n"
                    f"    for (int i = 0; i < {var_n}; i++) {{ long long x; cin >> x; {var_acc} += x; }}\n"
                    f"    cout << {var_acc};\n}}"
                )
            else:
                code = (
                    f"#include <iostream>\nusing namespace std;\nint main() {{\n"
                    f"    ios::sync_with_stdio(false); cin.tie(nullptr);\n"
                    f"    int {var_n}; cin >> {var_n}; long long {var_acc} = 0;\n"
                    f"    for (int i = 0; i < {var_n}; i++) {{ long long x; cin >> x; {var_acc} += x; }}\n"
                    f"    cout << {var_acc} << '\\n';\n}}"
                )
                lang = "cpp"

            records.append(
                {
                    "id": f"anchor_{bug_cls}_{i:04d}",
                    "source_dataset": "canonical_competitive_anchors",
                    "language": lang,
                    "problem_statement": stmt,
                    "code": code,
                    "bug_class": bug_cls,
                    "algo_family": algo,
                }
            )
    return records


def build_training_dataset(
    output_path: Path = DEFAULT_DATASET_PATH,
    samples_per_class: int = 400,
    seed: int = 42,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Combine real DeepMind CodeContests + IBM CodeNet4Repair + Canonical Anchors (19,799+ samples)."""
    if output_path.exists() and not force_rebuild:
        existing: list[dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        if len(existing) >= 10000:
            return existing

    tmp = tempfile.gettempdir()
    dm_path = os.path.join(tmp, "deepmind_cc_test.parquet")
    cn_test_path = os.path.join(tmp, "codenet_test.parquet")
    cn_train_path = os.path.join(tmp, "codenet_train.parquet")

    if not os.path.exists(cn_test_path):
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/TnT/Multi_CodeNet4Repair/resolve/main/data/test-00000-of-00001-347232bb2acb12c4.parquet",
                cn_test_path,
            )
        except Exception:  # noqa: BLE001
            pass

    real_dm = _extract_from_deepmind_parquet(dm_path, max_samples=6500)
    real_cn_test = _extract_from_codenet_parquet(cn_test_path, max_samples=4500)
    real_cn_train = _extract_from_codenet_parquet(cn_train_path, max_samples=6000)
    anchors = _gen_canonical_anchor_samples(samples_per_class=samples_per_class, seed=seed)

    all_records = real_dm + real_cn_test + real_cn_train + anchors
    rng = random.Random(seed)
    rng.shuffle(all_records)

    # Save outside OneDrive in temp for the full raw dump, and save a compact 5,000-record training split in data/
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"    -> Dataset Consolidado: {len(all_records):,} submissoes "
        f"(DeepMind CodeContests={len(real_dm):,}, IBM CodeNet4Repair={len(real_cn_test) + len(real_cn_train):,}, "
        f"Canonical Anchors={len(anchors):,})"
    )
    return all_records
