import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def create_scorecard_dataframe(airports_val, y_cls_val, y_reg_val, cls_predictions, reg_predictions):
    print(f"\n[6/6] Creating scorecard and visualizations...")
    
    scorecard_data = []
    unique_airports = np.unique(airports_val)

    for airport in unique_airports:
        mask = airports_val == airport
        if mask.sum() < 10:  
            continue
            
        airport_true_cls = y_cls_val[mask]
        airport_pred_cls = cls_predictions[mask]
        airport_true_reg = y_reg_val[mask]
        airport_pred_reg = reg_predictions[mask]

        airport_pred_cls_binary = (airport_pred_cls > 0.5).astype(int)
        
        tp = np.sum((airport_pred_cls_binary == 1) & (airport_true_cls == 1))
        fn = np.sum((airport_pred_cls_binary == 0) & (airport_true_cls == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fp = np.sum((airport_pred_cls_binary == 1) & (airport_true_cls == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        mae_airport = np.mean(np.abs(airport_true_reg - airport_pred_reg))
        avg_delay = np.mean(airport_true_reg)
        delay_rate = np.mean(airport_true_cls) * 100
        on_time_rate = np.mean(airport_true_reg <= 15) * 100

        mae_score = max(0, 100 - (mae_airport * 2))
        precision_score = precision * 100
        tpr_score = tpr * 100
        on_time_score = on_time_rate
        
        composite_score = (
            0.3 * mae_score +
            0.2 * precision_score +
            0.2 * tpr_score +
            0.3 * on_time_score
        )

        scorecard_data.append({
            'Airport': airport,
            'Score': round(composite_score, 1),
            'Avg_Delay': round(avg_delay, 1),
            'Delay_Rate': round(delay_rate, 1),
            'OnTime_Rate': round(on_time_rate, 1),
            'MAE': round(mae_airport, 1),
            'Precision': round(precision * 100, 1),
            'Recall': round(tpr * 100, 1),
            'Flights': mask.sum()
        })

    scorecard_df = pd.DataFrame(scorecard_data)
    scorecard_df = scorecard_df.sort_values('Score', ascending=False)
    return scorecard_df

def save_visualizations(scorecard_df, output_dir='output'):
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Airport Performance Analysis - Neural Network Model', fontsize=16, fontweight='bold')

    top_airports = scorecard_df.head(15)
    axes[0, 0].barh(top_airports['Airport'], top_airports['Score'])
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_title('Top 15 Airports by Performance Score')
    axes[0, 0].invert_yaxis()

    scatter = axes[0, 1].scatter(
        scorecard_df['Avg_Delay'],
        scorecard_df['Delay_Rate'],
        c=scorecard_df['Score'],
        cmap='viridis',
        alpha=0.6,
        s=scorecard_df['Flights'] / 10
    )
    axes[0, 1].set_xlabel('Average Delay (minutes)')
    axes[0, 1].set_ylabel('Delay Rate (%)')
    axes[0, 1].set_title('Delay Patterns by Airport')
    plt.colorbar(scatter, ax=axes[0, 1], label='Score')

    axes[1, 0].hist(scorecard_df['Score'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(scorecard_df['Score'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {scorecard_df['Score'].mean():.1f}")
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Number of Airports')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()

    axes[1, 1].scatter(
        scorecard_df['MAE'],
        scorecard_df['OnTime_Rate'],
        alpha=0.6,
        c=scorecard_df['Score'],
        cmap='plasma'
    )
    axes[1, 1].set_xlabel('Prediction MAE (minutes)')
    axes[1, 1].set_ylabel('Actual On-Time Rate (%)')
    axes[1, 1].set_title('Model Accuracy vs Actual Performance')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scorecard_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_dir}/scorecard_visualization.png")

def save_summary(scorecard_df, auc, mae, original_count, model_count, output_dir='output'):
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Flight Delay Prediction Project Summary\n")
        f.write(f"Generated: {time.ctime()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total flights analyzed: {original_count:,}\n")
        f.write(f"Flights in model: {model_count:,}\n")
        f.write(f"Airports analyzed: {len(scorecard_df)}\n")
        f.write(f"Model AUC: {auc:.4f}\n")
        f.write(f"Model MAE: {mae:.2f} minutes\n")
        f.write(f"\nTop 5 Airports:\n")
        for i, row in scorecard_df.head(5).iterrows():
            f.write(f"{i+1}. {row['Airport']}: Score={row['Score']}, Delay={row['Avg_Delay']}min\n")
        f.write(f"\nBottom 5 Airports:\n")
        for i, row in scorecard_df.tail(5).iterrows():
            f.write(f"{i+1}. {row['Airport']}: Score={row['Score']}, Delay={row['Avg_Delay']}min\n")
            
    print(f"Saved summary to {output_dir}/summary.txt")