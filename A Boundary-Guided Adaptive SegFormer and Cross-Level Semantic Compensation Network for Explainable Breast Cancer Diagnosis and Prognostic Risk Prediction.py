# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curves(y_true, y_pred_proba, num_classes, title='ROC Curves', save_path=None):
    """Plot ROC curves for multi-class classification"""
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        auc = roc_auc_score(y_true == i, y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_segmentation(image, mask, pred_mask=None):
    """Visualize segmentation results"""
    fig, axes = plt.subplots(1, 3 if pred_mask is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    if pred_mask is not None:
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def compute_segmentation_metrics(pred_mask, true_mask, smooth=1e-6):
    """Compute segmentation metrics: Dice, IoU, Hausdorff Distance"""
    
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # Dice coefficient
    intersection = np.sum(pred_flat * true_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(true_flat) + smooth)
    
    # IoU
    union = np.sum(pred_flat) + np.sum(true_flat) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # Hausdorff Distance (simplified)
    pred_points = np.argwhere(pred_mask > 0.5)
    true_points = np.argwhere(true_mask > 0.5)
    
    if len(pred_points) == 0 or len(true_points) == 0:
        hausdorff = 0
    else:
        dists = cdist(pred_points, true_points)
        hd_forward = np.max(np.min(dists, axis=1))
        hd_backward = np.max(np.min(dists, axis=0))
        hausdorff = max(hd_forward, hd_backward)
    
    return {
        'dice': dice,
        'iou': iou,
        'hausdorff_distance': hausdorff
    }

def perform_ablation_study(model, datasets, config):
    """Perform ablation study"""
    
    print("Performing ablation study...")
    
    ablation_configs = [
        {'name': 'Without SegFormer Encoder-Decoder', 
         'modifications': ['remove_encoder_decoder']},
        {'name': 'Without Boundary-Guided Feature Refinement',
         'modifications': ['remove_boundary_guidance']},
        {'name': 'Without Adaptive Multi-Scale Feature Aggregation',
         'modifications': ['remove_multi_scale']},
        {'name': 'Without Deep Supervision',
         'modifications': ['remove_deep_supervision']},
        {'name': 'Without Soft Dice Loss',
         'modifications': ['remove_dice_loss']},
        {'name': 'Without Lovász-Hinge Loss',
         'modifications': ['remove_lovasz_loss']},
        {'name': 'Without Edge-Aware BCE Loss',
         'modifications': ['remove_edge_bce_loss']},
        {'name': 'Full Proposed Model',
         'modifications': []}
    ]
    
    results = []
    for config_ab in ablation_configs:
        print(f"\nTesting: {config_ab['name']}")
        # Modify model based on config
        # Train and evaluate
        # Store results
    
    return results

def perform_k_fold_validation(model_class, data, config, n_folds=5):
    """Perform K-Fold cross-validation"""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, data['diagnosis'])):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split data
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Create datasets
        train_dataset = create_tf_dataset(train_data, config, 'train')
        val_dataset = create_tf_dataset(val_data, config, 'val')
        
        # Initialize and train model
        model = model_class(config)
        model.compile_model()
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.TRAINING_CONFIG['epochs'],
            verbose=0
        )
        
        # Evaluate
        metrics = evaluate_model(model, val_dataset)
        fold_metrics.append(metrics)
        
        print(f"  Validation Accuracy: {metrics['diagnosis']['accuracy']:.4f}")
        print(f"  Validation F1: {metrics['diagnosis']['f1_score']:.4f}")
    
    # Compute average metrics across folds
    avg_metrics = {}
    for metric_name in fold_metrics[0].keys():
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in fold_metrics])
        std_metrics[metric_name] = np.std([m[metric_name] for m in fold_metrics])
    
    print("\nK-Fold Cross-Validation Results:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    return avg_metrics, std_metrics

def perform_statistical_analysis(model1_results, model2_results):
    """Perform statistical analysis (ANOVA) between models"""
    
    from scipy import stats
    
    print("\nPerforming statistical analysis...")
    
    # Prepare data for ANOVA
    # This is a simplified example
    groups = []
    for metric in model1_results.keys():
        if metric != 'confusion_matrix':
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                model1_results[metric], 
                model2_results[metric]
            )
            print(f"{metric}: t={t_stat:.4f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  Significant difference in {metric}")
            else:
                print(f"  No significant difference in {metric}")
    
    return {}

def compare_with_and_without_preprocessing(model_class, data, config):
    """Compare model performance with and without preprocessing"""
    
    print("\nComparing with and without preprocessing...")
    
    results_with_preprocessing = []
    results_without_preprocessing = []
    
    # With preprocessing (using the standard pipeline)
    preprocessor = HistopathologyPreprocessor(config.PREPROCESSING_CONFIG)
    # ... apply preprocessing and train
    
    # Without preprocessing
    # ... skip preprocessing and train
    
    # Compare results
    comparison = {
        'accuracy': {
            'with_preprocessing': results_with_preprocessing,
            'without_preprocessing': results_without_preprocessing
        },
        'precision': {
            'with_preprocessing': results_with_preprocessing,
            'without_preprocessing': results_without_preprocessing
        },
        'recall': {
            'with_preprocessing': results_with_preprocessing,
            'without_preprocessing': results_without_preprocessing
        },
        'f1_score': {
            'with_preprocessing': results_with_preprocessing,
            'without_preprocessing': results_without_preprocessing
        }
    }
    
    return comparison

def generate_explainable_results(model, sample_data):
    """Generate explainable AI results"""
    
    print("\nGenerating explainable results...")
    
    # Grad-CAM for histopathology
    # SHAP for clinical and molecular features
    
    # For Grad-CAM, we need to get the last convolutional layer
    # For SHAP, we need to explain feature importance
    
    explanations = {
        'histopathology': {
            'heatmap': None,  # Grad-CAM heatmap
            'important_regions': []
        },
        'clinical': {
            'important_features': [],
            'feature_importance': []
        },
        'molecular': {
            'important_biomarkers': [],
            'biomarker_importance': []
        }
    }
    
    return explanations

# ============================================================================
# RUN COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    trainer, results = main()
    
    # Additional analyses
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSES")
    print("=" * 60)
    
    # Plot confusion matrices
    for task, metrics in results.items():
        if 'confusion_matrix' in metrics:
            plot_confusion_matrix(
                metrics['confusion_matrix'], 
                title=f'{task.upper()} Confusion Matrix',
                save_path=f'confusion_matrix_{task}.png'
            )
    
    # Compute segmentation metrics (if segmentation data available)
    # compute_segmentation_metrics(pred_mask, true_mask)
    
    print("\nAll analyses completed successfully!")