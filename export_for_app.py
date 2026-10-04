# ============================================================
# Run this as a new cell at the END of notebook 03.
# It writes everything the Streamlit app needs into app/.
# ============================================================
import os, json, joblib
import numpy as np
import pandas as pd

APP = 'app'
os.makedirs(APP, exist_ok=True)

# --- the winning model, unwrapped for SHAP ------------------------------
def unwrap(est, depth=5):
    for _ in range(depth):
        if hasattr(est, 'estimator'):
            est = est.estimator
        else:
            break
    return est

kind = models[BEST]['kind']
Xa, _, Xt = FRAMES[kind]

# --- category levels, for the dropdowns ---------------------------------
cat_levels = {c: sorted(pd.concat([Xa[c], Xt[c]]).astype(str).unique().tolist())
              for c in cat_cols}

# --- sensible numeric defaults and ranges -------------------------------
num_stats = {}
for c in num_cols:
    s = pd.concat([Xa[c], Xt[c]]).astype(float)
    num_stats[c] = {
        'min': float(s.min()), 'max': float(s.max()),
        'median': float(s.median()), 'mean': float(s.mean()),
    }

# --- a handful of real test rows, for the "load an example" button ------
examples = []
p_t = P_test[BEST]
yt_arr = y_test.to_numpy()
pred_t = (p_t >= BEST_T).astype(int)

picks = [
    ('High-risk borrower (defaulted)',
     np.where((pred_t == 1) & (yt_arr == 1))[0]),
    ('Low-risk borrower (repaid)',
     np.where((pred_t == 0) & (yt_arr == 0))[0]),
    ('Borderline case',
     np.array([int(np.argmin(np.abs(p_t - BEST_T)))])),
]
for label, idx in picks:
    if len(idx) == 0:
        continue
    i = int(idx[np.argmax(p_t[idx])] if 'High' in label else
            idx[np.argmin(p_t[idx])] if 'Low' in label else idx[0])
    row = Xt.iloc[i]
    examples.append({
        'label': label,
        'values': {c: (str(row[c]) if c in cat_cols else float(row[c]))
                   for c in Xt.columns},
        'actual': int(yt_arr[i]),
        'predicted_p': float(p_t[i]),
    })

bundle = {
    'model':        models[BEST]['model'],   # calibrated
    'raw_model':    unwrap(models[BEST]['model']),
    'kind':         kind,
    'model_name':   BEST,
    'feature_order': Xt.columns.tolist(),
    'cat_cols':     cat_cols,
    'num_cols':     num_cols,
    'cat_levels':   cat_levels,
    'num_stats':    num_stats,
    'p_star':       float(P_STAR),
    'm_median':     float(m_median),
    'lgd':          float(LGD),
    'examples':     examples,
    'test_auc':     float(roc_auc_score(y_test, p_t)),
    'test_cost':    float(total_cost(yt_arr, p_t, BEST_T, L_test, m_test)),
}

joblib.dump(bundle, f'{APP}/model_bundle.joblib')

size = os.path.getsize(f'{APP}/model_bundle.joblib') / 1024**2
print(f'wrote {APP}/model_bundle.joblib  ({size:.1f} MB)')
print(f'model          : {BEST} ({kind})')
print(f'p*             : {P_STAR:.4f}')
print(f'features ({len(Xt.columns)}): {Xt.columns.tolist()}')
print(f'examples       : {[e["label"] for e in examples]}')
