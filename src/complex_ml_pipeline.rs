use polars::prelude::*;
use smartcore::tree::decision_tree_classifier::*;
use smartcore::ensemble::random_forest_classifier::*;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::{accuracy, precision, recall, f1};
use smartcore::linalg::basic::matrix::DenseMatrix;
use rayon::prelude::*;
use std::time::Instant;
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
struct ModelMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Debug, Clone)]
struct HyperParameters {
    max_depth: Option<u16>,
    min_samples_split: usize,
    criterion: SplitCriterion,
}

fn main() -> PolarsResult<()> {
    run_complex_pipeline()
}

fn run_complex_pipeline() -> PolarsResult<()> {
    println!("=== Optimized Rust ML Pipeline ===\n");
    let pipeline_start = Instant::now();
    
    // Step 1: Generate complex dataset with missing values
    println!("Step 1: Generating complex dataset...");
    let (mut df, target) = generate_complex_dataset(50000, 20)?;
    println!("Dataset shape: {} rows × {} columns", df.height(), df.width());
    
    // Step 2: Advanced preprocessing with Polars
    println!("\nStep 2: Advanced preprocessing and imputation...");
    let imputation_start = Instant::now();
    df = advanced_preprocessing_polars(df)?;
    println!("Imputation time: {:.4}s", imputation_start.elapsed().as_secs_f64());
    
    // Step 3: Feature engineering with Polars
    println!("\nStep 3: Feature engineering...");
    let feature_start = Instant::now();
    df = feature_engineering_polars(df)?;
    println!("Feature engineering time: {:.4}s", feature_start.elapsed().as_secs_f64());
    
    // Step 4: Convert to SmartCore format
    let (x_matrix, y_vector) = polars_to_smartcore(df, target)?;
    
    // Step 5: Hyperparameter tuning with parallel grid search
    println!("\nStep 4: Hyperparameter tuning (parallel grid search)...");
    let tuning_start = Instant::now();
    let best_params = parallel_hyperparameter_tuning(&x_matrix, &y_vector);
    println!("Best parameters found: {:?}", best_params);
    println!("Tuning time: {:.4}s", tuning_start.elapsed().as_secs_f64());
    
    // Step 6: Train final model with best parameters
    println!("\nStep 5: Training final model with best parameters...");
    let final_start = Instant::now();
    let final_metrics = train_final_model(&x_matrix, &y_vector, &best_params);
    println!("Final model metrics:");
    println!("  Accuracy:  {:.4}", final_metrics.accuracy);
    println!("  Precision: {:.4}", final_metrics.precision);
    println!("  Recall:    {:.4}", final_metrics.recall);
    println!("  F1 Score:  {:.4}", final_metrics.f1);
    println!("Final training time: {:.4}s", final_start.elapsed().as_secs_f64());
    
    // Step 7: Ensemble learning with SmartCore Random Forest
    println!("\nStep 6: Ensemble learning with Random Forest...");
    let ensemble_start = Instant::now();
    let ensemble_metrics = train_ensemble(&x_matrix, &y_vector);
    println!("Ensemble model metrics:");
    println!("  Accuracy:  {:.4}", ensemble_metrics.accuracy);
    println!("  Precision: {:.4}", ensemble_metrics.precision);
    println!("  Recall:    {:.4}", ensemble_metrics.recall);
    println!("  F1 Score:  {:.4}", ensemble_metrics.f1);
    println!("Ensemble training time: {:.4}s", ensemble_start.elapsed().as_secs_f64());
    
    let total_time = pipeline_start.elapsed().as_secs_f64();
    println!("\n=== Total pipeline execution time: {:.4}s ===", total_time);
    
    Ok(())
}

fn generate_complex_dataset(n_rows: usize, n_features: usize) -> PolarsResult<(DataFrame, Vec<f64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut columns = Vec::new();
    
    // Generate numeric features with missing values
    for i in 0..(n_features / 2) {
        let missing_rate = 0.05 + (i as f64 * 0.02);
        let mut values = Vec::with_capacity(n_rows);
        
        for _ in 0..n_rows {
            if rng.gen::<f64>() < missing_rate {
                values.push(None);
            } else {
                values.push(Some(rng.gen_range(-100.0..100.0)));
            }
        }
        
        let series = Series::new(&format!("num_feat_{}", i), values);
        columns.push(series);
    }
    
    // Generate categorical features as strings first, then encode
    let categories = ["A", "B", "C", "D", "E"];
    for i in 0..(n_features / 4) {
        let missing_rate = 0.1;
        let mut values = Vec::with_capacity(n_rows);
        
        for _ in 0..n_rows {
            if rng.gen::<f64>() < missing_rate {
                values.push(None);
            } else {
                values.push(Some(categories[rng.gen_range(0..categories.len())].to_string()));
            }
        }
        
        let series = Series::new(&format!("cat_feat_{}", i), values);
        columns.push(series);
    }
    
    // Generate binary features
    for i in 0..(n_features / 4) {
        let values: Vec<f64> = (0..n_rows)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { 0.0 })
            .collect();
        
        let series = Series::new(&format!("bin_feat_{}", i), values);
        columns.push(series);
    }
    
    // Generate target variable
    let target: Vec<f64> = (0..n_rows)
        .map(|_| if rng.gen::<f64>() < 0.4 { 1.0 } else { 0.0 })
        .collect();
    
    let df = DataFrame::new(columns)?;
    Ok((df, target))
}

fn advanced_preprocessing_polars(df: DataFrame) -> PolarsResult<DataFrame> {
    let mut result_df = df;
    
    // Get all column names
    let column_names: Vec<String> = result_df.get_column_names().iter().map(|s| s.to_string()).collect();
    
    // Process numeric columns with median imputation
    let numeric_cols: Vec<String> = column_names.iter()
        .filter(|name| name.starts_with("num_"))
        .cloned()
        .collect();
    
    for col_name in &numeric_cols {
        result_df = result_df
            .lazy()
            .with_columns([
                col(col_name).fill_null(col(col_name).median())
            ])
            .collect()?;
    }
    
    // Process categorical columns with label encoding
    let categorical_cols: Vec<String> = column_names.iter()
        .filter(|name| name.starts_with("cat_"))
        .cloned()
        .collect();
    
    for col_name in &categorical_cols {
        result_df = result_df
            .lazy()
            .with_columns([
                // Fill nulls with "Unknown"
                col(col_name).fill_null(lit("Unknown"))
            ])
            .with_columns([
                // Manual label encoding
                when(col(col_name).eq(lit("A")))
                    .then(lit(1.0))
                    .when(col(col_name).eq(lit("B")))
                    .then(lit(2.0))
                    .when(col(col_name).eq(lit("C")))
                    .then(lit(3.0))
                    .when(col(col_name).eq(lit("D")))
                    .then(lit(4.0))
                    .when(col(col_name).eq(lit("E")))
                    .then(lit(5.0))
                    .otherwise(lit(0.0))
                    .alias(&format!("{}_encoded", col_name))
            ])
            .drop([col_name])
            .collect()?;
    }
    
    Ok(result_df)
}

fn feature_engineering_polars(df: DataFrame) -> PolarsResult<DataFrame> {
    let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let numeric_cols: Vec<String> = column_names.iter()
        .filter(|name| name.starts_with("num_"))
        .take(5)
        .cloned()
        .collect();
    
    let mut engineered_df = df;
    
    // Add polynomial and interaction features
    if numeric_cols.len() >= 2 {
        engineered_df = engineered_df
            .lazy()
            .with_columns([
                (col(&numeric_cols[0]) * col(&numeric_cols[1])).alias("interaction_0_1"),
                col(&numeric_cols[0]).pow(lit(2)).alias("squared_0"),
            ])
            .collect()?;
    }
    
    // Add normalized features
    for col_name in &numeric_cols {
        engineered_df = engineered_df
            .lazy()
            .with_columns([
                ((col(col_name) - col(col_name).mean()) / col(col_name).std(1))
                    .fill_null(lit(0.0))
                    .alias(&format!("{}_normalized", col_name))
            ])
            .collect()?;
    }
    
    Ok(engineered_df)
}

fn polars_to_smartcore(df: DataFrame, target: Vec<f64>) -> PolarsResult<(DenseMatrix<f64>, Vec<f64>)> {
    let n_rows = df.height();
    let n_cols = df.width();
    
    // Convert to a 2D vector first
    let mut matrix_data: Vec<Vec<f64>> = vec![vec![0.0; n_cols]; n_rows];
    
    for (col_idx, column) in df.get_columns().iter().enumerate() {
        match column.dtype() {
            DataType::Float64 => {
                let values = column.f64().unwrap();
                for (row_idx, value) in values.into_iter().enumerate() {
                    matrix_data[row_idx][col_idx] = value.unwrap_or(0.0);
                }
            }
            DataType::Int32 => {
                let values = column.i32().unwrap();
                for (row_idx, value) in values.into_iter().enumerate() {
                    matrix_data[row_idx][col_idx] = value.unwrap_or(0) as f64;
                }
            }
            DataType::Int64 => {
                let values = column.i64().unwrap();
                for (row_idx, value) in values.into_iter().enumerate() {
                    matrix_data[row_idx][col_idx] = value.unwrap_or(0) as f64;
                }
            }
            _ => {
                // Fill with zeros for unsupported types
                for row_idx in 0..n_rows {
                    matrix_data[row_idx][col_idx] = 0.0;
                }
            }
        }
    }
    
    let matrix = DenseMatrix::from_2d_vec(&matrix_data);
    Ok((matrix, target))
}

fn parallel_hyperparameter_tuning(x: &DenseMatrix<f64>, y: &[f64]) -> HyperParameters {
    let max_depths = vec![Some(5), Some(10), Some(15), None];
    let min_samples_splits = vec![2, 5, 10];
    let criteria = vec![SplitCriterion::Gini, SplitCriterion::Entropy];
    
    let mut param_combinations = Vec::new();
    for &max_depth in &max_depths {
        for &min_samples_split in &min_samples_splits {
            for &criterion in &criteria {
                param_combinations.push(HyperParameters {
                    max_depth,
                    min_samples_split,
                    criterion,
                });
            }
        }
    }
    
    println!("Testing {} parameter combinations...", param_combinations.len());
    
    // Use subset for faster tuning
    let subset_size = std::cmp::min(1000, x.nrows());
    let mut indices: Vec<usize> = (0..x.nrows()).collect();
    indices.shuffle(&mut ChaCha8Rng::seed_from_u64(42));
    let subset_indices = &indices[..subset_size];
    
    let x_subset_data: Vec<Vec<f64>> = subset_indices.iter()
        .map(|&i| x.get_row(i).to_vec())
        .collect();
    let x_subset = DenseMatrix::from_2d_vec(&x_subset_data);
    let y_subset: Vec<f64> = subset_indices.iter().map(|&i| y[i]).collect();
    
    // Parallel evaluation using rayon
    let results: Vec<(HyperParameters, f64)> = param_combinations
        .into_par_iter()
        .map(|params| {
            let score = evaluate_params_smartcore(&x_subset, &y_subset, &params);
            (params, score)
        })
        .collect();
    
    let (best_params, best_score) = results
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    
    println!("Best CV score: {:.4}", best_score);
    best_params
}

fn evaluate_params_smartcore(x: &DenseMatrix<f64>, y: &[f64], params: &HyperParameters) -> f64 {
    // Simple 3-fold cross-validation
    let n_samples = x.nrows();
    let fold_size = n_samples / 3;
    let mut scores = Vec::new();
    
    for fold in 0..3 {
        let test_start = fold * fold_size;
        let test_end = if fold == 2 { n_samples } else { (fold + 1) * fold_size };
        
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        
        for i in 0..n_samples {
            if i >= test_start && i < test_end {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }
        
        let x_train_data: Vec<Vec<f64>> = train_indices.iter()
            .map(|&i| x.get_row(i).to_vec())
            .collect();
        let x_test_data: Vec<Vec<f64>> = test_indices.iter()
            .map(|&i| x.get_row(i).to_vec())
            .collect();
        
        let x_train = DenseMatrix::from_2d_vec(&x_train_data);
        let x_test = DenseMatrix::from_2d_vec(&x_test_data);
        
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y[i]).collect();
        let y_test: Vec<f64> = test_indices.iter().map(|&i| y[i]).collect();
        
        match DecisionTreeClassifier::fit(
            &x_train,
            &y_train,
            DecisionTreeClassifierParameters::default()
                .with_max_depth(params.max_depth)
                .with_min_samples_split(params.min_samples_split)
                .with_criterion(params.criterion)
        ) {
            Ok(model) => {
                match model.predict(&x_test) {
                    Ok(predictions) => {
                        let acc = accuracy(&y_test, &predictions);
                        scores.push(acc);
                    }
                    Err(_) => scores.push(0.0),
                }
            }
            Err(_) => scores.push(0.0),
        }
    }
    
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

fn train_final_model(x: &DenseMatrix<f64>, y: &[f64], params: &HyperParameters) -> ModelMetrics {
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2, true, Some(42));
    
    match DecisionTreeClassifier::fit(
        &x_train,
        &y_train,
        DecisionTreeClassifierParameters::default()
            .with_max_depth(params.max_depth)
            .with_min_samples_split(params.min_samples_split)
            .with_criterion(params.criterion)
    ) {
        Ok(tree) => {
            match tree.predict(&x_test) {
                Ok(predictions) => calculate_metrics_smartcore(&y_test, &predictions),
                Err(_) => ModelMetrics { accuracy: 0.0, precision: 0.0, recall: 0.0, f1: 0.0 },
            }
        }
        Err(_) => ModelMetrics { accuracy: 0.0, precision: 0.0, recall: 0.0, f1: 0.0 },
    }
}

fn train_ensemble(x: &DenseMatrix<f64>, y: &[f64]) -> ModelMetrics {
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2, true, Some(42));
    
    match RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default()
            .with_n_trees(100)
            .with_max_depth(Some(10))
            .with_min_samples_split(2)
    ) {
        Ok(rf) => {
            match rf.predict(&x_test) {
                Ok(predictions) => calculate_metrics_smartcore(&y_test, &predictions),
                Err(_) => ModelMetrics { accuracy: 0.0, precision: 0.0, recall: 0.0, f1: 0.0 },
            }
        }
        Err(_) => ModelMetrics { accuracy: 0.0, precision: 0.0, recall: 0.0, f1: 0.0 },
    }
}

fn calculate_metrics_smartcore(y_true: &[f64], y_pred: &[f64]) -> ModelMetrics {
    let acc = accuracy(y_true, y_pred);
    let prec = precision(y_true, y_pred, 1.0);
    let rec = recall(y_true, y_pred, 1.0);
    let f1_score = f1(y_true, y_pred, 1.0);
    
    ModelMetrics {
        accuracy: acc,
        precision: prec,
        recall: rec,
        f1: f1_score,
    }
}