from random import sample
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import torch

def auoc_func(conf_matrix: np.ndarray, n_beta: int = 1001, class_values=None) -> float:
    """
    AUOC robuste aux lignes vides (classes jamais vraies) et aux labels non consécutifs.
    - conf_matrix: matrice carrée (KxK) de comptes >= 0
    - class_values: liste/array de taille K donnant la "valeur" ordinale de chaque classe
                    (dans le même ordre que la matrice). Si None -> distances |i-j|.
    """
    C = np.asarray(conf_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("conf_matrix must be a square 2D array")
    if not np.isfinite(C).all() or (C < 0).any():
        raise ValueError("conf_matrix must have finite, non-negative entries")

    K = C.shape[0]

    # Normalisation par ligne p(ŷ | y) en mettant 0 sur les lignes vides
    row_sums = C.sum(axis=1, keepdims=True)
    p = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums != 0)

    # Bénéfice (uniquement la diagonale)
    benefit = np.zeros_like(p)
    np.fill_diagonal(benefit, np.diag(p))

    # Matrice de distances ordinale
    if class_values is None:
        yy, yhat = np.indices((K, K))
        D = np.abs(yy - yhat).astype(float)
    else:
        v = np.asarray(class_values, dtype=float).reshape(-1)
        if v.shape[0] != K or not np.isfinite(v).all():
            raise ValueError("class_values must be finite and match conf_matrix size")
        D = np.abs(v[:, None] - v[None, :])  # |value_i - value_j|

    penalty = p * D

    neg_benefit_over_K = -benefit / K
    penalty_over_K = penalty / K

    betas = np.linspace(0.0, 1.0, n_beta)
    uoc_vals = np.empty_like(betas)

    for idx, beta in enumerate(betas):
        cell_cost = neg_benefit_over_K + beta * penalty_over_K

        # DP min-path (monotone down/right)
        dp = np.empty((K, K), dtype=float)
        dp[0, 0] = cell_cost[0, 0]
        for j in range(1, K):
            dp[0, j] = dp[0, j-1] + cell_cost[0, j]
        for i in range(1, K):
            dp[i, 0] = dp[i-1, 0] + cell_cost[i, 0]
        for i in range(1, K):
            for j in range(1, K):
                dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + cell_cost[i, j]

        uoc_vals[idx] = 1.0 + dp[-1, -1]  # constante expliquée précédemment

    auoc_value = float(np.trapz(uoc_vals, betas))
    return auoc_value

def iou_score(y_true, y_pred):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    if isinstance(y_pred, DMatrix):
        y_pred = np.copy(y_pred.get_data().toarray())

    if isinstance(y_true, DMatrix):
        y_true = np.copy(y_true.get_label())

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

def under_prediction_score(y_true, y_pred):
    """
    Calcule le score de sous-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est inférieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sous-prédiction.
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sous-prédiction
    under_prediction_area = np.trapz(np.maximum(y_true - y_pred, 0))  # Valeurs positives où y_true > y_pred
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return under_prediction_area / union_area if union_area > 0 else 0

def over_prediction_score(y_true, y_pred):
    """
    Calcule le score de sur-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est supérieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sur-prédiction.
    """
    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sur-prédiction
    over_prediction_area = np.trapz(np.maximum(y_pred - y_true, 0))  # Valeurs positives où y_pred > y_true
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return over_prediction_area / union_area if union_area > 0 else 0

def calculate_ic95(data):
    """
    Function to calculate the 95% confidence interval (IC95) for a given dataset.
    
    Parameters:
    data (array-like): Array of data points (e.g., model performance scores).

    Returns:
    tuple: lower bound and upper bound of the 95% confidence interval.
    """
    # Convert data to numpy array for convenience
    data = np.array(data)
    
    # Calculate the mean and standard error
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    
    # Calculate the 95% confidence interval using 1.96 for a 95% confidence level
    ci_lower = mean - 1.96 * std_err
    ci_upper = mean + 1.96 * std_err
    
    return ci_lower, ci_upper

def calculate_area_under_curve(y_values):
    """
    Calcule l'aire sous la courbe pour une série de valeurs données (méthode de trapèze).

    :param y_values: Valeurs sur l'axe des ordonnées pour calculer l'aire sous la courbe.
    :return: Aire sous la courbe.
    """
    return np.trapz(y_values, dx=1)

def evaluate_metrics(df, y_true_col='target', y_pred=None):
    """
    Calcule l'IoU et le F1-score sur chaque département, puis calcule l'aire sous la courbe normalisée (aire / aire maximale).
    
    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'nbsinister', 'target']
    :param dataset: Nom du dataset à filtrer
    :param y_true_col: Colonne représentant les cibles réelles
    :param y_pred: Liste ou tableau des prédictions
    :param metric: Choix de la métrique ('IoU' ou 'F1')
    :param top: Nombre de départements à afficher (ou 'all' pour tout afficher)
    :return: Dictionnaire contenant l'aire normalisée pour chaque modèle.
    """
    
    # Trier les valeurs par 'nbsinister' décroissant
    #df_sorted = df.sort_values(by='nbsinister', ascending=False)
    df_sorted = df
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]

    y_true = df[y_true_col]
    
    iou = iou_score(y_true, y_pred)
    f1 = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    prec = precision_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    rec = recall_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    
    f1_macro = f1_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    prec_macro = precision_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    rec_macro = recall_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    
    auoc = auoc_func(conf_matrix=confusion_matrix(y_true, y_pred, labels=np.union1d(y_true, y_pred)))

    under = under_prediction_score(y_true, y_pred)
    over = over_prediction_score(y_true, y_pred)

    # Initialiser un dictionnaire pour les résultats
    results = {'iou' : iou, 'f1' : f1, 'under' : under, 'over' : over, 'prec' : prec, 'recall' : rec,
               'auoc' : auoc, 'f1_macro' : f1_macro, 'prec_macro' : prec_macro, 'rec_macro' : rec_macro}

    # Calculer l'IoU et F1 pour chaque département
    IoU_scores = []
    F1_scores = []
    rec_scores = []
    prec_scores = []
    
    for i, department in enumerate(df_sorted['departement'].unique()):
        # Extraire les valeurs pour chaque département
        y_true = df_sorted[df_sorted['departement'] == department][y_true_col].values
        if np.all(y_true == 0):
            continue
        y_pred_department = y_pred[df_sorted['departement'] == department]  # Récupérer les prédictions associées au département
        
        # Calcul des scores IoU et F1
        IoU = iou_score(y_true, y_pred_department)
        F1 = f1_score(y_true > 0, y_pred_department > 0, zero_division=0)
        prec = precision_score(y_true > 0, y_pred_department > 0, zero_division=0)
        rec = recall_score(y_true > 0, y_pred_department > 0, zero_division=0)

        IoU_scores.append(IoU)
        F1_scores.append(F1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        
    df_sorted_test_area = df_sorted[df_sorted[y_true_col] > 0]
    # Calcul de l'aire maximale possible (cas parfait où toutes les prédictions sont correctes)
    max_area = np.trapz(np.ones(len(df_sorted_test_area['departement'].unique())), dx=1)
    
    # Calcul de l'aire sous la courbe pour l'IoU et le F1
    IoU_area = calculate_area_under_curve(IoU_scores)
    F1_area = calculate_area_under_curve(F1_scores)

    prec_area = calculate_area_under_curve(prec_scores)
    rec_area = calculate_area_under_curve(rec_scores)

    # Normalisation par l'aire maximale
    normalized_IoU = IoU_area / max_area if max_area > 0 else 0
    normalized_F1 = F1_area / max_area if max_area > 0 else 0
    normalized_rec = rec_area / max_area if max_area > 0 else 0
    normalized_prec = prec_area / max_area if max_area > 0 else 0
    
    y_true = df[y_true_col]
    
    # Stocker les résultats dans le dictionnaire
    results['normalized_iou'] = normalized_IoU
    results['normalized_f1'] = normalized_F1

    results['normalized_prec'] = normalized_prec
    results['normalized_rec'] = normalized_rec

    for elt in np.unique(y_true):
        if elt == 0:
            continue

        mask = (y_true >= elt) | (y_pred >= elt)

        if not np.any(mask):
            continue

        iou_elt = iou_score(y_true[mask], y_pred[mask])
        f1_elt = f1_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)
        prec_elt = precision_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)
        rec_elt = recall_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)

        f1_macro = f1_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')
        prec_macro = precision_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')
        rec_macro = recall_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')

        auoc_elt = auoc_func(confusion_matrix(y_true[mask], y_pred[mask], labels=np.union1d(y_true[mask], y_pred[mask])))

        results[f'iou_elt_sup_{elt}'] = iou_elt
        results[f'f1_elt_sup_{elt}'] = f1_elt
        results[f'prec_elt_sup_{elt}'] = prec_elt
        results[f'rec_elt_sup_{elt}'] = rec_elt
        
        results[f'f1_macro_elt_sup_{elt}'] = f1_macro
        results[f'prec_macro_elt_sup_{elt}'] = prec_macro
        results[f'rec_macro_elt_sup_{elt}'] = rec_macro

        results[f'auoc_elt_sup_{elt}'] = auoc_elt
    
    return results

def update_metrics_as_arrays(self, tp, metrics_run, set):
    """
    Met à jour self.metrics[tp] en stockant des tableaux NumPy.
    Pour chaque (k, v) dans metrics_run, on alimente la clé f"{k}_val".
    - v peut être un scalaire ou un array/list -> converti en 1D via np.atleast_1d.
    """
    bucket = self.metrics.setdefault(tp, {})
    for k, v in metrics_run.items():
        key = f"{k}_{set}"
        v_arr = np.atleast_1d(v).astype(float)

        if key not in bucket:
            # Première insertion -> tableau directement
            bucket[key] = v_arr.copy()
        else:
            # Concaténation avec l'existant
            bucket[key] = np.concatenate([bucket[key], v_arr])

from typing import Dict, Any, Iterable, Optional
import numpy as np

def add_ic95_to_dict(
    d: Dict[str, Any],
    keys: Optional[Iterable[str]] = None,
    suffix: str = "_ic95",
    dropna: bool = True,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Pour chaque clé 'metric' de d (ou sous-ensemble 'keys'), calcule l'IC95
    via calculate_ic95(d[metric]) et stocke un tuple (lower, upper) sous
    'metric{suffix}' (ex.: 'f1_ic95').

    Hypothèses:
    - d[metric] est une séquence numérique (list/tuple/ndarray) de valeurs (runs, sous-samples, etc.)
    - La fonction calculate_ic95(array_like) existe et renvoie (lower, upper)

    Paramètres
    ----------
    d : dict
        Dictionnaire des métriques => séquences de valeurs.
    keys : itérable de str, optionnel
        Si fourni, ne traite que ces clés. Sinon, toutes les clés sauf celles finissant par `suffix`.
    suffix : str
        Suffixe pour la clé IC95 (par défaut "_ic95").
    dropna : bool
        Si True, ignore les NaN avant le calcul.
    overwrite : bool
        Si False, n’écrase pas une clé '{metric}{suffix}' déjà existante.

    Retour
    ------
    dict (même objet) enrichi de paires '{metric}{suffix}': (lower, upper).
    """
    # Sélection des clés candidates
    if keys is None:
        candidates = [k for k in d.keys() if not k.endswith(suffix)]
    else:
        candidates = list(keys)

    for k in candidates:
        vals = d.get(k, None)
        if vals is None:
            continue

        # Convertir en tableau 1D de floats
        arr = np.asarray(vals, dtype=float).ravel()
        if dropna:
            arr = arr[~np.isnan(arr)]

        # Besoin d'au moins 2 points pour un IC95 basé sur SD
        if arr.size < 2:
            d[f"{k}{suffix}"] = (np.nan, np.nan)
            continue

        # Appel à la fonction externe calculate_ic95
        try:
            lower, upper = calculate_ic95(arr)
            lower = float(lower)
            upper = float(upper)
        except Exception:
            lower, upper = (np.nan, np.nan)

        out_key = f"{k}{suffix}"
        if overwrite or out_key not in d:
            d[out_key] = (lower, upper)

    return d

from typing import Any

def round_floats(obj: Any, ndigits: int = 2, round_keys: bool = False) -> Any:
    """
    Arrondit tous les float rencontrés dans une structure Python (dict, list, tuple, set),
    et renvoie une nouvelle structure du même type.
    
    - obj: structure d'entrée (dict, list, tuple, set, scalaires)
    - ndigits: nombre de décimales (par défaut 2)
    - round_keys: si True, arrondit aussi les *clés* de type float dans les dicts
                  (attention aux collisions possibles de clés après arrondi)
    """
    # float -> on arrondit
    if isinstance(obj, float):
        return round(obj, ndigits)

    # dict -> on traite clés/valeurs
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = round(k, ndigits) if (round_keys and isinstance(k, float)) else k
            new_dict[new_k] = round_floats(v, ndigits, round_keys)
        return new_dict

    # list -> on traite chaque élément
    if isinstance(obj, list):
        return [round_floats(x, ndigits, round_keys) for x in obj]

    # tuple -> on traite chaque élément et on recompose un tuple
    if isinstance(obj, tuple):
        return tuple(round_floats(x, ndigits, round_keys) for x in obj)

    # set -> on traite chaque élément (attention: l'arrondi peut fusionner des éléments)
    if isinstance(obj, set):
        return {round_floats(x, ndigits, round_keys) for x in obj}

    # autre type (int, str, bool, None, etc.) -> inchangé
    return obj
