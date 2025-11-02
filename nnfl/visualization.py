import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations


def load_and_prepare(file_path):
    df = pd.read_csv(file_path)
    return df


def filter_valid_features(df):
    drop_cols = ['Risk']
    id_like = [col for col in df.columns if 'id' in col.lower() or df[col].dtype == 'O']
    valid_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in drop_cols + id_like]
    return valid_cols


def visualize_frequencies(df, valid_cols):
    os.makedirs('output/frequency', exist_ok=True)
    for col in valid_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Frequency Distribution - {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f'output/frequency/{col}_distribution.png')
        plt.close()


def visualize_single_vs_risk(df, valid_cols):
    os.makedirs('output/compare_vs_risk', exist_ok=True)
    for col in valid_cols:
        plt.figure()
        sns.histplot(data=df, x=col, hue='Risk', kde=True, element='step', palette='Set2')
        plt.title(f'{col} vs Credit Risk')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'output/compare_vs_risk/{col}_vs_risk.png')
        plt.close()


def visualize_combinations_vs_risk(df, valid_cols):
    os.makedirs('output/combo_vs_risk', exist_ok=True)
    for combo in combinations(valid_cols, 2):
        plt.figure()
        sns.scatterplot(data=df, x=combo[0], y=combo[1], hue='Risk', palette='Set1', alpha=0.7)
        plt.title(f'{combo[0]} vs {combo[1]} by Risk')
        plt.tight_layout()
        safe_name = f"{combo[0].replace(' ', '_')}_{combo[1].replace(' ', '_')}_vs_risk.png"
        plt.savefig(f'output/combo_vs_risk/{safe_name}')
        plt.close()


def run_all_visualizations(file_path):
    df = load_and_prepare(file_path)
    valid_cols = filter_valid_features(df)
    visualize_frequencies(df, valid_cols)
    visualize_single_vs_risk(df, valid_cols)
    visualize_combinations_vs_risk(df, valid_cols)


if __name__ == '__main__':
    run_all_visualizations('german_credit_data.csv')