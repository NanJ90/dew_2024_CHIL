from collections import Counter
from glob import glob
import math
import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def get_divisors(n):
	divisors = []
	for i in range(1, int(math.sqrt(n) + 1)):
		if n % i == 0:
			divisors.append(i)
			divisors.append(int(n / i))
	return sorted(divisors)


def plot_cdf(folder):
	experiment_folder_patterns = [
		'eeg_eye_state', 'myocardial_infarction', 'diabetic_retinopathy_final',
		'wisconsin_bc_diagnosis', 'wisconsin_bc_prognosis', 'diabetes_vcu', 'cleveland_heart_disease'
	]

	all_error_files = []
	for f in experiment_folder_patterns:
		all_error_files += glob(os.path.join(folder, f, '**', '**', '**', 'errors_0.csv'))
	filtered_files = []
	for idx, errors_file in enumerate(all_error_files):
		errors = pd.read_csv(errors_file)
		try:
			dew_col = [c for c in errors.columns if '_top_None' in c][0]
			filtered_files.append(errors_file)
		except:
			print(list(errors.columns))
			continue
	print(filtered_files)
	num_files = len(filtered_files)
	integer_divisors = get_divisors(num_files)
	num_divisors = len(integer_divisors)
	cols = integer_divisors[num_divisors // 2]
	rows = integer_divisors[num_divisors // 2 - 1]
	fig, axes = plt.subplots(nrows=rows, ncols=cols)
	for idx, errors_file in enumerate(filtered_files):
		_name = errors_file.split('/')[-5]
		row, col = idx // cols, idx % cols
		errors = pd.read_csv(errors_file)
		try:
			dew_col = [c for c in errors.columns if '_top_None' in c][0]
		except:
			print(list(errors.columns))
			continue
		comparison_df = errors[['Uniform Model Averaging', dew_col]]
		comparison_df.diff(axis=1)[dew_col].hist(cumulative=True, density=1, bins=400)
		mean_ = comparison_df.diff(axis=1)[dew_col].mean()
		median_ = comparison_df.diff(axis=1)[dew_col].median()
		std_ = comparison_df.diff(axis=1)[dew_col].std()
		pct_better = (comparison_df.diff(axis=1)[dew_col] < 0).astype(int).sum() / len(comparison_df)

		print(_name)
		print(f'mean: {mean_}')
		print(f'median: {median_}')
		print(f'std: {std_}')
		print(f'percent better: {pct_better}')
		print('==============\n')

		plt.axhline(y=0.5, c='orange', linestyle='--')
		plt.title(f'CDF of DEW error improvements, {_name}', fontsize=8)

		fig.savefig(f'relative_errors/relative_errors_plots_{_name}.png', dpi=500)
		plt.clf()


def get_average_precisions(folder):
	experiment_folder_patterns = [
		'eeg_eye_state', 'myocardial_infarction', 'diabetic_retinopathy_final',
		'wisconsin_bc_diagnosis', 'wisconsin_bc_prognosis', 'diabetes_vcu', 'cleveland_heart_disease'
	]

	all_error_files = []
	all_pred_files = []
	for f in experiment_folder_patterns:
		all_error_files += glob(os.path.join(folder, f, '**', '**', '**', 'errors_0.csv'))
		all_pred_files += glob(os.path.join(folder, f, '**', '**', '**', 'predictions_0.csv'))
	
	filtered_error_files = []
	filtered_pred_files = []
	for idx, (errors_file, pred_file) in enumerate(zip(all_error_files, all_pred_files)):
		errors = pd.read_csv(errors_file)
		preds = pd.read_csv(pred_file)
		try:
			dew_col = [c for c in errors.columns if '_top_None' in c][0]
			filtered_error_files.append(errors_file)
			filtered_pred_files.append(pred_file)
		except:
			print(list(errors.columns))
			continue
	
	num_files = len(filtered_error_files)
	integer_divisors = get_divisors(num_files)
	num_divisors = len(integer_divisors)
	cols = integer_divisors[num_divisors // 2]
	rows = integer_divisors[num_divisors // 2 - 1]
	fig, axes = plt.subplots(nrows=rows, ncols=cols)

	rows = []
	names = []

	for idx, (errors_file, pred_file) in enumerate(zip(filtered_error_files, filtered_pred_files)):
		_name = errors_file.split('/')[-5] + ' ' + errors_file.split('/')[-3]
		names.append(_name)
		print(_name)
		errors = pd.read_csv(errors_file)
		preds = pd.read_csv(pred_file)

		uma_error_col = errors['Uniform Model Averaging']
		uma_pred_col = preds['Uniform Model Averaging']

		dew_error_col = errors[dew_col]
		dew_pred_col = preds[dew_col]

		targets = [
			0 if np.abs(uma_error_col[idx] - uma_pred_col[idx]) < 0.001 
			else 1 
			for idx in range(len(uma_error_col)) 
		]

		print(Counter(targets))
		pct_pos = np.round(sum(targets) / len(targets) * 100)
		
		ap_uma = np.round(average_precision_score(targets, uma_pred_col) * 100, 3)
		ap_dew = np.round(average_precision_score(targets, dew_pred_col) * 100, 3)
		auroc_uma = np.round(roc_auc_score(targets, uma_pred_col) * 100, 3)
		auroc_dew = np.round(roc_auc_score(targets, dew_pred_col) * 100, 3)

		print('UMA AP: ', ap_uma)
		print('DEW AP: ', ap_dew)
		print('UMA AUROC: ', auroc_uma)
		print('DEW AUROC: ', auroc_dew)

		dew_colname = [c for c in errors.columns if '_top_None' in c][0]

		comparison_df = errors[['Uniform Model Averaging', dew_col]]
		pct_better = (comparison_df.diff(axis=1)[dew_col] < 0).astype(int).sum() / len(comparison_df)
		pct_better = np.round(pct_better * 100, 3)

		row = [ap_uma, ap_dew, auroc_uma, auroc_dew, pct_pos, pct_better]
		rows.append(row)

	colnames = ['UMA AP', 'DEW AP', 'UMA AUROC', 'DEW AUROC', 'Percent Positive Class', 'Percent Samples: DEW Error < UMA Error']
	out_df = pd.DataFrame(data = np.array(rows), columns=colnames)
	out_df.index = names
	out_df.to_csv('experiment_metrics.csv')


get_average_precisions('../Experiment_Trials')

plot_cdf('../Experiment_Trials')
