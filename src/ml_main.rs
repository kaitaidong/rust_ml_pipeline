// src/ml_main.rs
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::metrics::{mean_squared_error, accuracy};
use smartcore::model_selection::train_test_split;

use rand::Rng;
use std::time::Instant;

// Add main function as entry point
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_ml_demo()
}

pub fn run_ml_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Rust Machine Learning with SmartCore ===\n");
    
    // Generate synthetic dataset
    let n_samples = 10000;
    let n_features = 20;
    let mut rng = rand::thread_rng();
    
    let start = Instant::now();
    
    // Create feature matrix
    let mut x_data = Vec::new();
    let mut y_regression = Vec::new();
    let mut y_classification = Vec::new();
    
    for _ in 0..n_samples {
        let mut features = Vec::new();
        let mut weighted_sum = 0.0;
        
        for j in 0..n_features {
            let val = rng.gen_range(-1.0..1.0);
            features.push(val);
            // Create a linear relationship with noise
            weighted_sum += val * (j as f64 + 1.0) * 0.5;
        }
        
        x_data.extend(features);
        
        // Regression target (linear with noise)
        y_regression.push(weighted_sum + rng.gen_range(-5.0..5.0));
        
        // Classification target (binary based on threshold)
        y_classification.push(if weighted_sum > 0.0 { 1 } else { 0 });
    }
    
    // Fix: Use DenseMatrix::new with 4 arguments (including column_major flag)
    let x = DenseMatrix::new(n_samples, n_features, x_data, true);
    
    println!("Dataset created: {} samples, {} features", n_samples, n_features);
    println!("Data generation time: {:?}\n", start.elapsed());
    
    // === Linear Regression ===
    println!("--- Linear Regression ---");
    let regression_start = Instant::now();
    
    let (x_train_reg, x_test_reg, y_train_reg, y_test_reg) = 
        train_test_split(&x, &y_regression, 0.2, true, Some(42));
    
    let lr = LinearRegression::fit(&x_train_reg, &y_train_reg, Default::default())?;
    let predictions_reg = lr.predict(&x_test_reg)?;
    
    let mse = mean_squared_error(&y_test_reg, &predictions_reg);
    let r2_score = calculate_r2(&y_test_reg, &predictions_reg);
    
    println!("Training time: {:?}", regression_start.elapsed());
    println!("Mean Squared Error: {:.4}", mse);
    println!("R² Score: {:.4}\n", r2_score);
    
    // === Decision Tree Classification ===
    println!("--- Decision Tree Classification ---");
    let classification_start = Instant::now();
    
    let (x_train_cls, x_test_cls, y_train_cls, y_test_cls) = 
        train_test_split(&x, &y_classification, 0.2, true, Some(42));
    
    let tree = DecisionTreeClassifier::fit(
        &x_train_cls, 
        &y_train_cls, 
        Default::default()
    )?;
    
    let predictions_cls = tree.predict(&x_test_cls)?;
    let acc = accuracy(&y_test_cls, &predictions_cls);
    
    println!("Training time: {:?}", classification_start.elapsed());
    println!("Accuracy: {:.4}", acc);
    
    // === Parallel Processing Demo ===
    println!("\n--- Parallel Processing with Rayon ---");
    use rayon::prelude::*;
    
    let parallel_start = Instant::now();
    
    // Simulate multiple model training in parallel
    let results: Vec<f64> = (0..10)
        .into_par_iter()
        .map(|i| {
            // Train a model with different random seed
            let (x_train, _, y_train, y_test) = 
                train_test_split(&x, &y_regression, 0.2, true, Some(i as u64));
            
            let model = LinearRegression::fit(&x_train, &y_train, Default::default())
                .expect("Failed to train model");
            
            let preds = model.predict(&x_test_reg).expect("Failed to predict");
            mean_squared_error(&y_test_reg, &preds)
        })
        .collect();
    
    let avg_mse = results.iter().sum::<f64>() / results.len() as f64;
    println!("Trained 10 models in parallel");
    println!("Average MSE: {:.4}", avg_mse);
    println!("Parallel training time: {:?}", parallel_start.elapsed());
    
    println!("\nTotal ML pipeline time: {:?}", start.elapsed());
    
    Ok(())
}

fn calculate_r2(y_true: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    let mean_y = y_true.iter().sum::<f64>() / y_true.len() as f64;
    
    let ss_tot: f64 = y_true.iter()
        .map(|y| (y - mean_y).powi(2))
        .sum();
    
    let ss_res: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(y_t, y_p)| (y_t - y_p).powi(2))
        .sum();
    
    1.0 - (ss_res / ss_tot)
}