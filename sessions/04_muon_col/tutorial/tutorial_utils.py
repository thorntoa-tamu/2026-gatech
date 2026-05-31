from pyexpat import model

from sklearn import metrics
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_feature_distributions(data, labels):
    """Histogram all cluster features, split by signal vs BIB."""
    feature_keys = data['feature_keys']
    sig_mask = labels == 1
    bib_mask = labels == 0

    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feature_keys):
        ax = axes[i]
        vals = data['features'][feat]
        lo, hi = np.percentile(vals, [1, 99])
        ax.hist(vals[sig_mask], bins=50, range=(lo, hi), alpha=0.6, density=True, label='Signal')
        ax.hist(vals[bib_mask], bins=50, range=(lo, hi), alpha=0.6, density=True, label='BIB')
        ax.set_title(feat.replace('cluster_', ''), fontsize=10)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Cluster Feature Distributions: Signal vs BIB', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_hit_multiplicity(data, labels):
    """Histogram raw-hit counts per cluster, split by signal vs BIB."""
    n_raw = data['raw_hit_counts']
    sig_mask = labels == 1
    bib_mask = labels == 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(n_raw[sig_mask], bins=range(0, 30), alpha=0.6, density=True, label='Signal')
    ax.hist(n_raw[bib_mask], bins=range(0, 30), alpha=0.6, density=True, label='BIB')
    ax.set_xlabel('Number of raw pixel hits per cluster')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Cluster Multiplicity')
    plt.tight_layout()
    plt.show()

    print(f"Raw hits per cluster — median: {np.median(n_raw):.0f}, "
          f"mean: {np.mean(n_raw):.1f}, max: {np.max(n_raw)}")


def plot_feature_importance(bdt, feature_names):
    """Bar chart of BDT feature importances, sorted descending."""
    importances = bdt.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(feature_names)), importances[order])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(
        [feature_names[i].replace('cluster_', '') for i in order],
        rotation=45, ha='right',
    )
    ax.set_ylabel('Feature importance')
    ax.set_title('BDT Feature Importance')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_scores, cut=0.5, title='Confusion Matrix',
                          class_labels=('BIB', 'Signal')):
    """Plot a truth-normalised confusion matrix."""
    y_pred = (np.asarray(y_scores) >= cut).astype(int)
    y_true = np.asarray(y_true)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    bib_total = tn + fp
    sig_total = fn + tp
    tn_pct = 100.0 * tn / bib_total if bib_total > 0 else 0.0
    fp_pct = 100.0 * fp / bib_total if bib_total > 0 else 0.0
    fn_pct = 100.0 * fn / sig_total if sig_total > 0 else 0.0
    tp_pct = 100.0 * tp / sig_total if sig_total > 0 else 0.0

    matrix_pct = np.array([[tn_pct, fn_pct], [fp_pct, tp_pct]])
    matrix_cnt = np.array([[tn, fn], [fp, tp]])
    sig_eff = tp / sig_total if sig_total > 0 else 0.0
    bib_rej = tn / bib_total if bib_total > 0 else 0.0

    cmap = LinearSegmentedColormap.from_list(
        'muc_blue',
        [(0.03, 0.10, 0.35), (0.10, 0.34, 0.78), (0.72, 0.92, 0.98)],
        N=128,
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.imshow(matrix_pct, cmap=cmap, vmin=0, vmax=100, origin='upper', aspect='equal')

    for i in range(2):
        for j in range(2):
            pct = matrix_pct[i, j]
            cnt = matrix_cnt[i, j]
            text_color = 'white' if pct < 35 else 'black'
            ax.text(j, i - 0.06, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color=text_color)
            ax.text(j, i + 0.18, f'({cnt:,})',
                    ha='center', va='center', fontsize=10, color=text_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_labels, fontsize=12)
    ax.set_xlabel('Truth class', fontsize=13)
    ax.set_ylabel('Predicted class', fontsize=13)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    fig.text(0.05, 0.04,
             f'$\\varepsilon_{{sig}}$ = {sig_eff:.3f}',
             ha='left', fontsize=11, color='gray')
    fig.text(0.05, 0.00,
             f'$r_{{BIB}}$ = {bib_rej:.3f}',
             ha='left', fontsize=11, color='gray')

    fig.text(0.95, 0.04, 'Muon Collider Simulation', ha='right',
             fontsize=8, color='gray', style='italic')
    fig.text(0.95, 0.00, 'MAIA Geometry', ha='right',
             fontsize=8, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def plot_roc_comparison(classifiers):
    """Overlay ROC curves.

    Parameters
    ----------
    classifiers : dict of name -> {'fpr': array, 'tpr': array, 'auc': float}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in classifiers.items():
        ax.plot(r['tpr'], 1.0 - r['fpr'],
                label=f"{name} (AUC={r['auc']:.4f})", linewidth=2)
    ax.set_xlabel('Signal Efficiency')
    ax.set_ylabel('1 − BIB Efficiency (Background Rejection)')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=12)
    ax.set_title('Signal vs BIB Separation')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_leaderboard(entries):
    """Print and plot a performance-vs-size leaderboard.

    Parameters
    ----------
    entries : list of (name, auc, size_bytes, fom)
    """
    entries = sorted(entries, key=lambda x: x[3], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<25} {'AUC':>8} {'Size (KB)':>10} {'FoM':>8}")
    print('-' * 60)
    for rank, (name, auc, size, fom) in enumerate(entries, 1):
        print(f"{rank:<5} {name:<25} {auc:>8.4f} {size/1024:>10.1f} {fom:>8.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, auc, size, fom in entries:
        ax.scatter(size / 1024, auc, s=150, zorder=5)
        ax.annotate(name, (size / 1024, auc), textcoords='offset points',
                    xytext=(10, 5), fontsize=10)
    ax.set_xlabel('Model Size (KB)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Performance vs Size — Can you reach the top-left corner?', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, train_dataset, val_dataset, model, lr,
                 optimizer, loss_fn=nn.CrossEntropyLoss, device='cuda'):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.best_model_wts = None

    def _run_epoch(self, dataloader, training=True):
        if training:
            self.model.train()
            losses = []
            for batch in dataloader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        else:
            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in dataloader:
                    X = batch['X'].to(self.device)
                    y = batch['y'].to(self.device)
                    losses.append(self.loss_fn(self.model(X), y).item())
        return np.mean(losses)

    def train(self, num_epochs, patience=10):
        best_loss = np.inf
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(self.train_dataset, training=True)
            val_loss = self._run_epoch(self.val_dataset, training=False)
            print(f'Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                self.best_model_wts = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
        print(f'Training complete. Total time: {time.time() - t0:.1f}s')


def plot_score_distributions(y_true, y_scores, title='Classifier Score Distribution'):
    """Overlay signal and BIB score distributions on the test set."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_scores[y_true == 1], bins=60, range=(0, 1), alpha=0.6,
            density=True, label='Signal', color='tab:blue')
    ax.hist(y_scores[y_true == 0], bins=60, range=(0, 1), alpha=0.6,
            density=True, label='BIB', color='tab:orange')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  Evaluation helpers
# ---------------------------------------------------------------------------

def predict_model(model, test_loader, device='cpu'):
    """Run a trained model on a test loader. Returns (signal_scores, labels)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            all_preds.append(model(X).softmax(-1).cpu().numpy())
            all_labels.append(batch['y'].numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds[:, 1], labels


def evaluate_classifier(y_true, y_scores, name='Classifier',
                        efficiencies=(0.5, 0.8, 0.9, 0.95)):
    """Compute and print classification metrics.

    Returns dict with keys: auc, fpr, tpr, accuracy.
    """
    auc = metrics.roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    acc = metrics.accuracy_score(y_true, (np.asarray(y_scores) > 0.5).astype(int))

    print(f"\n{name} AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    for eff in efficiencies:
        idx = np.argmax(tpr >= eff)
        print(f"  Signal eff = {tpr[idx]:.3f}  ->  1 - BIB eff = {1.0 - fpr[idx]:.4f}")

    return {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'accuracy': acc}


# ---------------------------------------------------------------------------
#  Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(model, test_loader, device='cuda', num_warmup=5, num_runs=50):
    """Compute size, speed, and performance metrics for a model."""
    model.eval()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    size_fp32 = num_params * 4
    size_int8 = num_params * 1

    bops_fp32 = 0
    for _name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bops_fp32 += 2 * module.in_features * module.out_features * 32

    sample_batch = next(iter(test_loader))['X'].to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            model(sample_batch)
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(num_runs):
            model(sample_batch)
        if device == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / num_runs * 1000

    scores, lbls = predict_model(model, test_loader, device)
    auc = metrics.roc_auc_score(lbls, scores)

    fom_fp32 = auc / np.log10(max(size_fp32, 1))
    fom_int8 = auc / np.log10(max(size_int8, 1))

    return {
        'num_params': num_params,
        'trainable_params': trainable_params,
        'size_fp32_bytes': size_fp32,
        'size_int8_bytes': size_int8,
        'bops_fp32': bops_fp32,
        'latency_ms': latency_ms,
        'auc': auc,
        'fom_fp32': fom_fp32,
        'fom_int8': fom_int8,
    }


def benchmark_bdt(bdt, auc):
    """Compute and print BDT benchmark metrics. Returns dict."""
    bdt_bytes = len(pickle.dumps(bdt))
    fom = auc / np.log10(max(bdt_bytes, 1))

    print(f"\n{'='*50}")
    print(f" BDT Benchmark")
    print(f"{'='*50}")
    print(f"  Trees:           {bdt.n_estimators:>10}")
    print(f"  Max depth:       {bdt.max_depth:>10}")
    print(f"  Serialized size: {bdt_bytes:>10,} bytes  ({bdt_bytes/1024:.1f} KB)")
    print(f"  AUC:             {auc:>10.4f}")
    print(f"  FoM:             {fom:>10.4f}")
    print(f"{'='*50}")

    return {'size_bytes': bdt_bytes, 'auc': auc, 'fom': fom}


def print_benchmark(result, name='Model'):
    print(f"\n{'='*50}")
    print(f" {name} Benchmark")
    print(f"{'='*50}")
    print(f"  Parameters:      {result['num_params']:>10,}")
    print(f"  Size (FP32):     {result['size_fp32_bytes']:>10,} bytes  ({result['size_fp32_bytes']/1024:.1f} KB)")
    print(f"  Size (INT8):     {result['size_int8_bytes']:>10,} bytes  ({result['size_int8_bytes']/1024:.1f} KB)")
    print(f"  BOPs (FP32):     {result['bops_fp32']:>10,}")
    print(f"  Latency:         {result['latency_ms']:>10.2f} ms / batch")
    print(f"  AUC:             {result['auc']:>10.4f}")
    print(f"  FoM (FP32):      {result['fom_fp32']:>10.4f}")
    print(f"  FoM (INT8):      {result['fom_int8']:>10.4f}")
    print(f"{'='*50}")

# -------------------------------------------------------------------
# Pretty model card + cartoon diagram
# -------------------------------------------------------------------
from torch.fx import symbolic_trace
from IPython.display import display, Markdown

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def display_model_card(model, config, max_hits=9):
    total_params, trainable_params = count_parameters(model)

    display(Markdown(f"""
### Pixel Transformer

| Setting | Value |
|---|---:|
| Input features | `{config['input_dim']}` |
| Hidden dimension | `{config['hidden_dim']}` |
| Transformer blocks | `{config['num_layers']}` |
| Output classes | `{config['num_classes']}` |
| Total parameters | `{total_params:,}` |
| Trainable parameters | `{trainable_params:,}` |
"""))