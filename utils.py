from sklearn import metrics

def evaluate(pred,target):
    return {'f1_score':metrics.f1_score(target,pred,average='weighted'),
            'acc_score':metrics.accuracy_score(target,pred),
            'precision_score':metrics.precision_score(target,pred,average='weighted'),
            'recall_score':metrics.recall_score(target,pred,average='weighted'),
            'roc_auc_score':metrics.roc_auc_score(target,pred,average='weighted')}

def f1_metric(eval_df,_builtin_metrics):
    return metrics.f1_score(eval_df['target'],eval_df['prediction'],average='weighted')
def recall_metric(eval_df,_builtin_metrics):
    return metrics.recall_score(eval_df['target'],eval_df['prediction'],average='weighted')
def precision_metric(eval_df,_builtin_metrics):
    return metrics.precision_score(eval_df['target'],eval_df['prediction'],average='weighted')
def roc_metric(eval_df,_builtin_metrics):
    return metrics.roc_auc_score(eval_df['target'],eval_df['prediction'],average='weighted')
def acc_metric(eval_df,_builtin_metrics):
    return metrics.accuracy_score(eval_df['target'],eval_df['prediction'])

def print_dct(dct):
    for k in dct:
        print(f"{k} - {dct[k]}")