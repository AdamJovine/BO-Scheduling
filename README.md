# Exam Scheduling Optimization

This project generates optimized exam schedules using integer programming (Gurobi) and Bayesian optimization (Thompson Sampling with Gaussian Processes). The goal is to minimize student conflicts, late exams, and other scheduling penalties.

## Components

### 1. Data Preprocessing
- Loads block assignments, exam sizes, and enrollment data.
- Builds co-enrollment dictionaries for student overlaps across exams.
- Identifies slot patterns for conflict detection (e.g., 3-in-1-day, back-to-back exams).

### 2. Schedule Evaluation
- Computes schedule metrics such as:
  - Student conflicts
  - Triple and quadruple exam clusters
  - Back-to-back exams
  - Weighted lateness

### 3. Scheduling Model (`schedule_ip`)
- Gurobi-based model that assigns exam blocks to slots.
- Supports constraints on large exams, slot timing, and student load.
- Uses warm-starting for faster convergence.

### 4. Optimization (`exam_schedule_objective`)
- Wraps the scheduler and returns 8 metrics for optimization.
- Varies penalty weights and thresholds for large exams and blocks.

### 5. Thompson Sampling Loop
- Runs multi-objective Bayesian optimization using Gaussian Processes.
- Selects new candidates using random scalarization of objectives.
- Saves schedule performance and Pareto-optimal points to CSV.

## Outputs
- Schedules: `/home/asj53/final-scheduling/results/sp25/schedules/`
- Metrics: `/home/asj53/final-scheduling/results/sp25/metrics/`
- Optimization results: `/home/asj53/GaussinSamplingbo_results20.csv`

## Requirements
- Gurobi with a valid WLS license.
- Preloaded data in `/home/asj53/final-scheduling/data/sp25/`
