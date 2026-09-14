"""Complete automated pipeline for SignalScope: Training, Evaluation, Robustness, and Backend Testing."""

import os
import sys
import unittest
from model.train import train_model
from model.evaluate import run_full_evaluation
from model.robustness import run_robustness_analysis


def main():
    print("=" * 70, flush=True)
    print("      SIGNALSCOPE ML PIPELINE - AUTOMATED EXECUTION & VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Train Model
    print("\n[STEP 1/4] Training ResNet-18 Model on Dataset...", flush=True)
    train_results = train_model(max_samples_per_class=2000, epochs=3)
    print(f" -> Training complete. Best Val AUC: {train_results['best_val_auc']:.4f}", flush=True)

    # Step 2: Full Metric Evaluation
    print("\n[STEP 2/4] Running Full Evaluation on Test & Unseen-Generator Splits...", flush=True)
    eval_results = run_full_evaluation(max_samples_per_class=1000)
    print(f" -> Standard Test ROC-AUC:           {eval_results['summary']['standard_roc_auc']:.4f}", flush=True)
    print(f" -> Unseen-Generator Test ROC-AUC:  {eval_results['summary']['unseen_generator_roc_auc']:.4f}", flush=True)

    # Step 3: Robustness Analysis
    print("\n[STEP 3/4] Benchmarking Robustness to Image Degradations...", flush=True)
    robustness_results = run_robustness_analysis(max_samples=300)
    print(" -> Robustness analysis complete.", flush=True)

    # Step 4: Run Backend Unit Tests
    print("\n[STEP 4/4] Executing Full FastAPI Backend Unit Test Suite...", flush=True)
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    if test_result.wasSuccessful():
        print("\n ALL BACKEND UNIT TESTS PASSED SUCCESSFULLY!", flush=True)
    else:
        print("\n WARNING: Some unit tests failed.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("      SIGNALSCOPE ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
