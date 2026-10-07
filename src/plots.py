"""Figures generated from measured data and stored model predictions."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hitpredict-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from .data import ROOT, FEATURES

def make_plots(d,comparison,predictions,importance,summary):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":160,"savefig.bbox":"tight"})
    def save(name):
        plt.tight_layout(); plt.savefig(ROOT/f"results/figures/{name}.png"); plt.close()
    train = d[d.split=="train"]
    fig,axs = plt.subplots(1,3,figsize=(12,3.5))
    for ax,feature in zip(axs,["Danceability","Acousticness","Loudness"]):
        for label,color in [(0,"#94a3b8"),(1,"#15803d")]:
            ax.hist(train.loc[train.Label==label,feature],bins=25,density=True,alpha=.6,
                    label="Hit" if label else "Non-hit",color=color)
        ax.set_xlabel(feature + (" (dB)" if feature=="Loudness" else " (0-1)"))
        ax.set_ylabel("Density")
    axs[0].legend(); fig.suptitle("Audio feature distributions in the training data")
    save("feature_distributions")
    fig,ax = plt.subplots(figsize=(9,4.5))
    c=comparison.sort_values("cv_auc_mean")
    ax.barh(c.model,c.cv_auc_mean,xerr=c.cv_auc_std,color="#15803d",capsize=3)
    ax.set_xlim(0,1);ax.set_xlabel("Training group cross-validation ROC-AUC (mean ± SD)")
    save("model_comparison")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    fpr,tpr,_=roc_curve(predictions.Label,predictions.score)
    axs[0].plot(fpr,tpr,color="#15803d",label=f"AUC = {summary['test']['roc_auc']:.3f}")
    axs[0].plot([0,1],[0,1],"--",color="#94a3b8");axs[0].set(xlabel="False positive rate",ylabel="True positive rate",title="Test ROC curve")
    p,r,_=precision_recall_curve(predictions.Label,predictions.score)
    axs[1].plot(r,p,color="#15803d",label=f"AP = {summary['test']['average_precision']:.3f}")
    axs[1].axhline(predictions.Label.mean(),linestyle="--",color="#94a3b8",label="Test hit prevalence")
    axs[1].set(xlabel="Recall",ylabel="Precision",title="Test precision-recall curve",ylim=(0,1.02))
    for ax in axs:ax.legend(loc="lower left")
    save("test_curves")
    fig,ax=plt.subplots(figsize=(4.5,4))
    cm=np.array(summary["test"]["confusion_matrix"])
    ax.imshow(cm,cmap="Greens")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=22,color="white" if cm[i,j]>cm.max()*.6 else "#102a20")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Non-hit","Hit"],yticklabels=["Non-hit","Hit"],xlabel="Predicted label",ylabel="Source label",title="Test confusion matrix")
    save("confusion_matrix")
    fig,ax=plt.subplots(figsize=(8,4))
    imp=importance.sort_values("auc_drop_mean")
    ax.barh(imp.feature,imp.auc_drop_mean,xerr=imp.auc_drop_std,color="#15803d",capsize=3)
    ax.axvline(0,color="#94a3b8");ax.set_xlabel("Validation ROC-AUC decrease after shuffling (mean ± SD)")
    save("feature_importance")
    fig,ax=plt.subplots(figsize=(8,3.7))
    counts=d.groupby(["split","Label"]).size().unstack().reindex(["train","validation","test"])
    counts.rename(columns={0:"Non-hit",1:"Hit"}).plot.bar(stacked=True,ax=ax,color=["#94a3b8","#15803d"],rot=0)
    ax.set(xlabel="Split",ylabel="Tracks",title="Distinct artist/audio groups across splits")
    save("data_splits")
