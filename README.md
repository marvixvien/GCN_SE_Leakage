# GCN_SE_Leakage
The pipeline generates hydraulic simulations via EPANET, trains a physics-informed GCN (enforcing mass balance and head-loss constraints), then feeds pressure-flow estimates into XGBoost/Random Forest leak classifiers. With only five sensors, it achieves 99.92% head correlation and 98% leak detection accuracy at ~7ms inference.
