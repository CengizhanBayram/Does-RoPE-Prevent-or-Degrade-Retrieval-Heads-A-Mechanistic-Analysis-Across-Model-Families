import json, ast

p = "notebooks/layer_d_only_colab.ipynb"
nb = json.load(open(p, encoding="utf-8"))

preflight = r'''# Cell 4: SETUP + PREFLIGHT. Read TWO NUMBERS before the full run:
#   (1) BASELINE accuracy -> is the experiment informative?  (decide FIRST)
#   (2) s/head estimate   -> does it fit in time?
import sys, json, gc, time
sys.path.insert(0, '/content/rope-retrieval-heads')
sys.path.insert(0, '/content/rope-retrieval-heads/scripts')
from pathlib import Path
import numpy as np, torch, yaml
from src.model_loader import load_model, purge_hf_cache
from src.retrieval_head_detector import RetrievalHeadDetector
from src.dimension_utility import DimensionUtilityAnalyzer
from src.activation_patching import ActivationPatcher
from src.repro import set_determinism, capture_environment

with open('/content/rope-retrieval-heads/configs/config.yaml') as f:
    config = yaml.safe_load(f)

SEED = 42
PATCH_CONTEXTS  = [4096]        # SIGNAL region (long-distance; short ctx = ceiling)
PATCH_POSITIONS = [0.5]
N_SAMPLES       = 15
RANDOM_SEEDS    = [0, 1, 2]
TOP_K_HEADS     = 30
DETECT_POSITIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
DETECT_N         = 25
k_dims = config['activation_patching']['k_dims']

out_dir = Path(RESULTS_DIR) / 'layer_d' / MODEL
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_path  = str(out_dir / 'patch_ckpt_topk.json')
heads_path = out_dir / 'topk_heads.npz'

strict = config.get('reproducibility', {}).get('strict_determinism', False)
set_determinism(SEED, strict=strict)
model, tok = load_model(config['models'][MODEL], MODEL)

analyzer = DimensionUtilityAnalyzer(model, config)
freq_order = analyzer.freq_order
norms = analyzer.compute_query_projection_norms()

if heads_path.exists():
    z = np.load(heads_path, allow_pickle=True)
    retrieval_heads = [tuple(int(v) for v in x) for x in z['heads']]
    print(f'Loaded TOP-{len(retrieval_heads)} heads from {heads_path.name}')
else:
    det = RetrievalHeadDetector(model, tok, config,
              score_threshold=config['niah'].get('score_threshold', 0.1), seed=SEED)
    det_samples = det.generate_niah_samples(DETECT_N, PATCH_CONTEXTS, DETECT_POSITIONS)
    scores = det.score_heads(det_samples)
    all_heads = det.get_retrieval_heads(scores)
    ranked = sorted(all_heads, key=lambda lh: scores[lh[0], lh[1]], reverse=True)
    retrieval_heads = ranked[:TOP_K_HEADS]
    np.savez(heads_path, heads=np.array(retrieval_heads))
    print(f'Detected {len(all_heads)} heads @4096; TOP {len(retrieval_heads)} kept -> {heads_path.name}')

if not retrieval_heads:
    raise SystemExit('No retrieval heads found.')

det2 = RetrievalHeadDetector(model, tok, config,
           score_threshold=config['niah'].get('score_threshold', 0.1), seed=SEED)
samples = det2.generate_niah_samples(N_SAMPLES, PATCH_CONTEXTS, PATCH_POSITIONS)
eval_samples = samples[:N_SAMPLES]

patcher = ActivationPatcher(model, tok, config)

# ===== NUMBER 1 (decide FIRST): BASELINE -- is the experiment informative? =====
t0 = time.time()
baseline_acc = patcher._evaluate_accuracy(eval_samples)
t_base = time.time() - t0
print()
print('=' * 66)
print(f'BASELINE accuracy (no patch @4096, n={len(eval_samples)}) = {baseline_acc:.3f}   [{t_base:.0f}s]')
print('  1.00     -> CEILING: per-head zeroing cannot show effect -> POPULATION patch')
print('  0.00     -> FLOOR: nothing to drop (likely PG-19 fallback corpus) -> fix corpus')
print('  0.6-0.9  -> IDEAL: margin exists -> proceed')

# ===== NUMBER 2: timing -- does it fit? =====
lay, hd = retrieval_heads[0]
t0 = time.time()
_ = patcher._evaluate_accuracy(eval_samples, lay, hd, list(range(k_dims)))
s_cond = time.time() - t0
n_cond = 4 + len(RANDOM_SEEDS)
eta_h = s_cond * n_cond * len(retrieval_heads) / 3600
print(f's/condition ~= {s_cond:.0f}s  ->  s/head ~= {s_cond*n_cond:.0f}s  ->  '
      f'ETA(top-{len(retrieval_heads)}) ~= {eta_h:.1f} h')
print('=' * 66)
print('DECISION: baseline in [0.6,0.9] AND ETA < 24h -> run NEXT cell.')
print('          baseline 1.0 or 0.0 -> STOP, tell me the number (config change, not speed).')'''

fullrun = r'''# Cell 5: FULL run -- ONLY if the preflight baseline is in the good band.
# Resumable: if the session drops, just re-run this cell (continues from checkpoint).
patching_results = patcher.run_patching_experiment(
    retrieval_heads=retrieval_heads,
    utility_scores={'_norms': norms},
    samples=samples,
    k_dims=k_dims,
    n_samples=N_SAMPLES,
    random_seeds=RANDOM_SEEDS,
    freq_order=freq_order,
    checkpoint_path=ckpt_path,
)
causal_df = patcher.compute_causal_effect(patching_results)
print(f'Done: {len(causal_df)} heads patched.')'''

idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if "run_patching_experiment" in s:
        idx = i
        break

nb["cells"][idx]["source"] = preflight
nb["cells"].insert(idx + 1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": fullrun})
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1)

# verify
nb = json.load(open(p, encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if s.lstrip().startswith("%%"):
        continue
    ast.parse(s)
print("cells:", len(nb["cells"]), "| split at idx", idx, "| ALL PARSE OK")
