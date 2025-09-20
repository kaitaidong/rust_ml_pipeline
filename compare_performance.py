#!/usr/bin/env python3
"""
Complete Performance Comparison Script
Compares Rust vs Python ML pipeline performance with comprehensive reporting.
"""

import subprocess
import time
import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def main():
    """Main function to run the complete comparison"""
    print("🔬 ML Pipeline Performance Comparison: Rust vs Python")
    print("=" * 70)
    
    # Verify prerequisites
    if not check_prerequisites():
        return
    
    # Run both pipelines and collect results
    print("\n📊 Running Performance Tests...")
    rust_result = run_rust_pipeline()
    python_result = run_python_pipeline()
    
    # Generate comparison report
    print("\n📈 Generating Comparison Report...")
    generate_comparison_report(rust_result, python_result)
    
    print("\n✅ Comparison complete! Check the comparison_results/ directory for detailed reports.")

def check_prerequisites() -> bool:
    """Check if all required files and dependencies exist"""
    print("🔍 Checking prerequisites...")
    
    # Check for required files
    required_files = ["Cargo.toml", "complex_ml_pipeline.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ {file} not found. Please ensure it's in the current directory.")
            return False
    
    # Check if Rust project can be built
    print("🦀 Checking Rust build...")
    try:
        build_result = subprocess.run(
            ["cargo", "check", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if build_result.returncode != 0:
            print(f"❌ Rust project check failed: {build_result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Cargo not found or build check timed out")
        return False
    
    # Check Python dependencies
    print("🐍 Checking Python dependencies...")
    try:
        import numpy, pandas, sklearn
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependencies: {e}")
        print("Install with: pip install numpy pandas scikit-learn matplotlib")
        return False

def run_rust_pipeline() -> Dict:
    """Run the Rust ML pipeline and collect metrics"""
    print("\n🦀 Running Rust ML Pipeline...")
    
    start_time = time.time()
    try:
        # Build in release mode
        subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            check=True,
            capture_output=True,
            timeout=120
        )
        
        # Run the pipeline
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "complex_ml_pipeline"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Rust pipeline completed in {total_time:.3f}s")
            metrics = extract_metrics(result.stdout)
            return {
                "success": True,
                "total_time": total_time,
                "output": result.stdout,
                "metrics": metrics
            }
        else:
            print(f"❌ Rust pipeline failed: {result.stderr}")
            return {
                "success": False,
                "total_time": total_time,
                "output": result.stderr,
                "metrics": {}
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Rust pipeline timed out")
        return {"success": False, "total_time": -1, "output": "Timeout", "metrics": {}}
    except Exception as e:
        print(f"❌ Error running Rust pipeline: {e}")
        return {"success": False, "total_time": -1, "output": str(e), "metrics": {}}

def run_python_pipeline() -> Dict:
    """Run the Python ML pipeline and collect metrics"""
    print("\n🐍 Running Python ML Pipeline...")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "complex_ml_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Python pipeline completed in {total_time:.3f}s")
            metrics = extract_metrics(result.stdout)
            return {
                "success": True,
                "total_time": total_time,
                "output": result.stdout,
                "metrics": metrics
            }
        else:
            print(f"❌ Python pipeline failed: {result.stderr}")
            return {
                "success": False,
                "total_time": total_time,
                "output": result.stderr,
                "metrics": {}
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Python pipeline timed out")
        return {"success": False, "total_time": -1, "output": "Timeout", "metrics": {}}
    except Exception as e:
        print(f"❌ Error running Python pipeline: {e}")
        return {"success": False, "total_time": -1, "output": str(e), "metrics": {}}

def extract_metrics(output: str) -> Dict[str, float]:
    """Extract timing and performance metrics from pipeline output"""
    metrics = {}
    
    # Timing patterns
    time_patterns = {
        'imputation_time': r'Imputation time:\s*([\d.]+)',
        'feature_engineering_time': r'Feature engineering time:\s*([\d.]+)',
        'tuning_time': r'Tuning time:\s*([\d.]+)',
        'final_training_time': r'Final training time:\s*([\d.]+)',
        'ensemble_training_time': r'Ensemble training time:\s*([\d.]+)',
        'total_pipeline_time': r'Total pipeline execution time:\s*([\d.]+)'
    }
    
    # Accuracy patterns
    accuracy_patterns = {
        'dt_accuracy': r'Accuracy:\s*([\d.]+)',
        'dt_precision': r'Precision:\s*([\d.]+)',
        'dt_recall': r'Recall:\s*([\d.]+)',
        'dt_f1': r'F1 Score:\s*([\d.]+)'
    }
    
    # Extract timing metrics
    for key, pattern in time_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            try:
                metrics[key] = float(matches[-1])  # Take the last match
            except ValueError:
                pass
    
    # Extract accuracy metrics
    for key, pattern in accuracy_patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            try:
                metrics[key] = float(matches[0])  # Take the first match
            except ValueError:
                pass
    
    return metrics

def generate_comparison_report(rust_result: Dict, python_result: Dict):
    """Generate comprehensive comparison report"""
    
    # Create results directory
    results_dir = Path("comparison_results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate console summary
    print_console_summary(rust_result, python_result)
    
    # Generate detailed reports
    create_csv_report(rust_result, python_result, results_dir)
    create_html_report(rust_result, python_result, results_dir)
    create_json_report(rust_result, python_result, results_dir)
    
    # Try to create plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        create_performance_plots(rust_result, python_result, results_dir)
        print(f"📊 Performance plots saved to {results_dir}")
    except ImportError:
        print("⚠️  Matplotlib not available - skipping plot generation")
        print("   Install with: pip install matplotlib")

def print_console_summary(rust_result: Dict, python_result: Dict):
    """Print summary to console"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    if rust_result["success"] and python_result["success"]:
        rust_time = rust_result["total_time"]
        python_time = python_result["total_time"]
        
        print(f"🦀 Rust Total Time:   {rust_time:.3f}s")
        print(f"🐍 Python Total Time: {python_time:.3f}s")
        print("-" * 40)
        
        if rust_time < python_time:
            speedup = python_time / rust_time
            print(f"🚀 Rust is {speedup:.1f}x FASTER than Python!")
        else:
            slowdown = rust_time / python_time
            print(f"🐌 Python is {slowdown:.1f}x faster than Rust")
        
        # Show key metrics if available
        rust_metrics = rust_result["metrics"]
        python_metrics = python_result["metrics"]
        
        if "dt_accuracy" in rust_metrics and "dt_accuracy" in python_metrics:
            print(f"\n📊 Decision Tree Accuracy:")
            print(f"   Rust:   {rust_metrics['dt_accuracy']:.4f}")
            print(f"   Python: {python_metrics['dt_accuracy']:.4f}")
        
    else:
        print("❌ One or both pipelines failed:")
        if not rust_result["success"]:
            print(f"   Rust: {rust_result['output'][:200]}...")
        if not python_result["success"]:
            print(f"   Python: {python_result['output'][:200]}...")
    
    print("="*70)

def create_csv_report(rust_result: Dict, python_result: Dict, results_dir: Path):
    """Create CSV report"""
    try:
        import pandas as pd
        
        data = []
        
        # Overall timing
        data.append({
            "Metric": "Total Execution Time (s)",
            "Rust": f"{rust_result['total_time']:.3f}" if rust_result["success"] else "Failed",
            "Python": f"{python_result['total_time']:.3f}" if python_result["success"] else "Failed",
            "Advantage": calculate_advantage(rust_result['total_time'], python_result['total_time'], "time")
        })
        
        # Detailed metrics
        if rust_result["success"] and python_result["success"]:
            rust_metrics = rust_result["metrics"]
            python_metrics = python_result["metrics"]
            
            common_metrics = set(rust_metrics.keys()) & set(python_metrics.keys())
            
            for metric in sorted(common_metrics):
                rust_val = rust_metrics[metric]
                python_val = python_metrics[metric]
                
                metric_type = "time" if "time" in metric else "accuracy"
                advantage = calculate_advantage(rust_val, python_val, metric_type)
                
                data.append({
                    "Metric": metric.replace('_', ' ').title(),
                    "Rust": f"{rust_val:.4f}",
                    "Python": f"{python_val:.4f}",
                    "Advantage": advantage
                })
        
        df = pd.DataFrame(data)
        csv_path = results_dir / "performance_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV report saved to {csv_path}")
        
    except ImportError:
        print("⚠️  Pandas not available - skipping CSV report")

def create_html_report(rust_result: Dict, python_result: Dict, results_dir: Path):
    """Create HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🦀 Rust vs 🐍 Python ML Pipeline Comparison</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            }}
            h1 {{ 
                color: #333; 
                text-align: center; 
                border-bottom: 3px solid #4CAF50; 
                padding-bottom: 10px; 
            }}
            .summary {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 20px 0; 
            }}
            .metrics-table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }}
            .metrics-table th, .metrics-table td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            .metrics-table th {{ 
                background-color: #f8f9fa; 
                font-weight: bold; 
            }}
            .metrics-table tr:nth-child(even) {{ 
                background-color: #f8f9fa; 
            }}
            .rust-better {{ background-color: #d4edda; }}
            .python-better {{ background-color: #f8d7da; }}
            .similar {{ background-color: #fff3cd; }}
            .status {{ 
                padding: 5px 10px; 
                border-radius: 20px; 
                font-weight: bold; 
            }}
            .success {{ background-color: #d4edda; color: #155724; }}
            .failure {{ background-color: #f8d7da; color: #721c24; }}
            .footer {{ 
                text-align: center; 
                margin-top: 30px; 
                color: #666; 
                font-size: 0.9em; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦀 Rust vs 🐍 Python ML Pipeline Comparison</h1>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                {generate_html_summary(rust_result, python_result)}
            </div>
            
            <h2>📈 Detailed Performance Metrics</h2>
            {generate_html_table(rust_result, python_result)}
            
            <div class="footer">
                <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Pipeline: Data Generation → Preprocessing → Feature Engineering → Hyperparameter Tuning → Model Training → Ensemble Learning</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    html_path = results_dir / "performance_comparison.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"🌐 HTML report saved to {html_path}")

def generate_html_summary(rust_result: Dict, python_result: Dict) -> str:
    """Generate HTML summary section"""
    if rust_result["success"] and python_result["success"]:
        rust_time = rust_result["total_time"]
        python_time = python_result["total_time"]
        
        if rust_time < python_time:
            speedup = python_time / rust_time
            winner = f"🦀 Rust is <strong>{speedup:.1f}x faster</strong> than Python!"
        else:
            speedup = rust_time / python_time
            winner = f"🐍 Python is <strong>{speedup:.1f}x faster</strong> than Rust!"
        
        return f"""
        <p><strong>🦀 Rust:</strong> {rust_time:.3f}s</p>
        <p><strong>🐍 Python:</strong> {python_time:.3f}s</p>
        <p><strong>Result:</strong> {winner}</p>
        """
    else:
        return "<p><strong>⚠️ One or both pipelines failed to complete successfully.</strong></p>"

def generate_html_table(rust_result: Dict, python_result: Dict) -> str:
    """Generate HTML metrics table"""
    if not (rust_result["success"] and python_result["success"]):
        return "<p>❌ Cannot generate detailed metrics - pipeline failures occurred.</p>"
    
    rows = []
    
    # Add total time row
    rust_time = rust_result["total_time"]
    python_time = python_result["total_time"]
    advantage = calculate_advantage(rust_time, python_time, "time")
    css_class = get_css_class(advantage)
    
    rows.append(f"""
    <tr class="{css_class}">
        <td><strong>Total Execution Time (s)</strong></td>
        <td>{rust_time:.3f}</td>
        <td>{python_time:.3f}</td>
        <td>{advantage}</td>
    </tr>
    """)
    
    # Add detailed metrics
    rust_metrics = rust_result["metrics"]
    python_metrics = python_result["metrics"]
    common_metrics = set(rust_metrics.keys()) & set(python_metrics.keys())
    
    for metric in sorted(common_metrics):
        rust_val = rust_metrics[metric]
        python_val = python_metrics[metric]
        metric_type = "time" if "time" in metric else "accuracy"
        advantage = calculate_advantage(rust_val, python_val, metric_type)
        css_class = get_css_class(advantage)
        
        rows.append(f"""
        <tr class="{css_class}">
            <td>{metric.replace('_', ' ').title()}</td>
            <td>{rust_val:.4f}</td>
            <td>{python_val:.4f}</td>
            <td>{advantage}</td>
        </tr>
        """)
    
    return f"""
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>🦀 Rust</th>
                <th>🐍 Python</th>
                <th>Advantage</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

def create_json_report(rust_result: Dict, python_result: Dict, results_dir: Path):
    """Create JSON report for programmatic access"""
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "rust": rust_result,
        "python": python_result,
        "comparison": {}
    }
    
    if rust_result["success"] and python_result["success"]:
        rust_time = rust_result["total_time"]
        python_time = python_result["total_time"]
        
        report["comparison"] = {
            "rust_faster": rust_time < python_time,
            "speedup": python_time / rust_time if rust_time > 0 else 0,
            "time_difference": abs(rust_time - python_time),
            "winner": "rust" if rust_time < python_time else "python"
        }
    
    json_path = results_dir / "performance_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📋 JSON report saved to {json_path}")

def create_performance_plots(rust_result: Dict, python_result: Dict, results_dir: Path):
    """Create performance visualization plots"""
    import matplotlib.pyplot as plt
    
    if not (rust_result["success"] and python_result["success"]):
        return
    
    # Overall time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Overall execution time
    languages = ['Rust', 'Python']
    times = [rust_result["total_time"], python_result["total_time"]]
    colors = ['#CE422B', '#3776AB']
    
    bars1 = ax1.bar(languages, times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Overall Pipeline Execution Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Detailed timing breakdown
    rust_metrics = rust_result["metrics"]
    python_metrics = python_result["metrics"]
    
    timing_metrics = [k for k in rust_metrics.keys() if 'time' in k and k in python_metrics]
    
    if timing_metrics:
        x_pos = range(len(timing_metrics))
        rust_times = [rust_metrics[m] for m in timing_metrics]
        python_times = [python_metrics[m] for m in timing_metrics]
        
        width = 0.35
        ax2.bar([x - width/2 for x in x_pos], rust_times, width, 
               label='Rust', color='#CE422B', alpha=0.8)
        ax2.bar([x + width/2 for x in x_pos], python_times, width,
               label='Python', color='#3776AB', alpha=0.8)
        
        ax2.set_title('Detailed Timing Breakdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in timing_metrics], rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = results_dir / "performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_advantage(rust_val: float, python_val: float, metric_type: str) -> str:
    """Calculate performance advantage"""
    if rust_val <= 0 or python_val <= 0:
        return "N/A"
    
    if metric_type == "time":
        # For timing metrics, lower is better
        if rust_val < python_val:
            ratio = python_val / rust_val
            return f"Rust {ratio:.1f}x faster"
        else:
            ratio = rust_val / python_val
            return f"Python {ratio:.1f}x faster"
    else:
        # For accuracy metrics, higher is better
        diff = rust_val - python_val
        if abs(diff) < 0.001:
            return "Similar"
        elif diff > 0:
            return f"Rust +{diff:.4f}"
        else:
            return f"Python +{abs(diff):.4f}"

def get_css_class(advantage: str) -> str:
    """Get CSS class based on advantage"""
    if "Rust" in advantage and "faster" in advantage:
        return "rust-better"
    elif "Python" in advantage and "faster" in advantage:
        return "python-better"
    else:
        return "similar"

if __name__ == "__main__":
    main()