# Comparación estadística de métodos de explicación

AUC de localización por vídeo promediado sobre semillas; test de Wilcoxon pareado (CAM frente a cada método), corrección de Holm dentro de cada dataset × modelo. Δ = media(CAM − método); 'gana' = % de vídeos en que el CAM tiene mayor AUC. AOPC = fidelidad (caída del logit al borrar los pasos más importantes).


## thumos_i3d — bilstm-const200  (n = 247 vídeo-clase; vídeo mAP 0.943)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.735 ± 0.003 | 0.61 | 2.32 | — | — | — |
| Final memory | 0.553 ± 0.013 | 0.31 | 0.22 | +0.181 * | 84 % | 2.2e-28 |
| Memory writes | 0.547 ± 0.014 | 0.35 | 0.21 | +0.188 * | 85 % | 1.5e-28 |
| Occlusion | 0.619 ± 0.014 | 0.43 | 1.39 | +0.116 * | 86 % | 8.2e-29 |
| Integrated Gradients | 0.718 ± 0.004 | 0.48 | 2.21 | +0.017 * | 70 % | 2.8e-15 |
| Grad x Input | 0.593 ± 0.008 | 0.44 | 1.44 | +0.142 * | 93 % | 4.3e-36 |
| Random | 0.497 ± 0.004 | 0.26 | 0.10 | +0.238 * | 96 % | 1.0e-38 |
| Sampled Shapley (s0) | 0.691 ± 0.000 | 0.48 | 1.63 | +0.041 * | 64 % | 2.2e-09 |

## thumos_i3d — bilstm-standard  (n = 247 vídeo-clase; vídeo mAP 0.411)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.603 ± 0.008 | 0.21 | 1.00 | — | — | — |
| Final memory | 0.448 ± 0.005 | 0.02 | -1.12 | +0.156 * | 87 % | 3.2e-29 |
| Memory writes | 0.233 ± 0.006 | 0.06 | -1.27 | +0.370 * | 96 % | 4.0e-39 |
| Occlusion | 0.528 ± 0.006 | 0.19 | 0.80 | +0.075 * | 83 % | 1.1e-23 |
| Integrated Gradients | 0.549 ± 0.008 | 0.15 | 1.01 | +0.055 * | 69 % | 1.3e-09 |
| Grad x Input | 0.482 ± 0.007 | 0.07 | -0.03 | +0.122 * | 91 % | 4.3e-34 |
| Random | 0.497 ± 0.004 | 0.26 | -0.09 | +0.106 * | 84 % | 1.8e-29 |
| Sampled Shapley (s0) | 0.646 ± 0.000 | 0.36 | 0.80 | -0.043 * | 37 % | 1.1e-05 |

## thumos_i3d — gru-const200  (n = 247 vídeo-clase; vídeo mAP 0.835)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.640 ± 0.012 | 0.54 | 1.08 | — | — | — |
| Final memory | 0.596 ± 0.007 | 0.10 | -0.06 | +0.043 * | 66 % | 5.4e-08 |
| Memory writes | 0.528 ± 0.013 | 0.23 | -0.15 | +0.111 * | 71 % | 1.5e-13 |
| Occlusion | 0.603 ± 0.009 | 0.40 | 1.04 | +0.036 * | 63 % | 5.4e-08 |
| Integrated Gradients | 0.711 ± 0.013 | 0.56 | 1.85 | -0.071 * | 20 % | 3.1e-19 |
| Grad x Input | 0.572 ± 0.012 | 0.38 | 1.26 | +0.067 * | 81 % | 4.9e-25 |
| Random | 0.497 ± 0.004 | 0.26 | -0.04 | +0.143 * | 91 % | 5.1e-36 |
| Sampled Shapley (s0) | 0.691 ± 0.000 | 0.50 | 1.10 | -0.055 * | 28 % | 1.3e-09 |

## thumos_i3d — gru-standard  (n = 247 vídeo-clase; vídeo mAP 0.401)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.545 ± 0.010 | 0.21 | 0.62 | — | — | — |
| Final memory | 0.502 ± 0.006 | 0.03 | -0.89 | +0.043 * | 66 % | 9.0e-07 |
| Memory writes | 0.254 ± 0.004 | 0.07 | -1.33 | +0.291 * | 96 % | 5.3e-38 |
| Occlusion | 0.538 ± 0.009 | 0.19 | 0.86 | +0.007 | 53 % | 4.3e-01 |
| Integrated Gradients | 0.491 ± 0.009 | 0.05 | 0.17 | +0.054 * | 70 % | 4.4e-11 |
| Grad x Input | 0.444 ± 0.014 | 0.06 | -0.40 | +0.101 * | 82 % | 1.2e-25 |
| Random | 0.497 ± 0.004 | 0.26 | -0.02 | +0.048 * | 70 % | 8.1e-13 |
| Sampled Shapley (s0) | 0.641 ± 0.000 | 0.34 | 0.82 | -0.094 * | 28 % | 1.6e-14 |

## thumos_i3d — lstm-const200  (n = 247 vídeo-clase; vídeo mAP 0.913)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.711 ± 0.012 | 0.54 | 1.64 | — | — | — |
| Final memory | 0.576 ± 0.010 | 0.19 | 0.10 | +0.134 * | 82 % | 1.8e-21 |
| Memory writes | 0.547 ± 0.013 | 0.20 | 0.06 | +0.164 * | 80 % | 6.1e-25 |
| Murdoch-Szlam additive | 0.637 ± 0.010 | 0.50 | 1.08 | +0.073 * | 85 % | 1.5e-26 |
| Murdoch-Szlam cell | 0.584 ± 0.007 | 0.50 | 0.90 | +0.127 * | 91 % | 4.2e-32 |
| LRP-LSTM | 0.695 ± 0.011 | 0.54 | 1.54 | +0.015 * | 70 % | 6.0e-11 |
| Contextual Decomposition | 0.747 ± 0.009 | 0.59 | 1.07 | -0.037 * | 28 % | 8.5e-16 |
| Occlusion | 0.576 ± 0.004 | 0.34 | 0.95 | +0.135 * | 89 % | 5.0e-33 |
| Integrated Gradients | 0.679 ± 0.013 | 0.36 | 1.48 | +0.032 * | 78 % | 2.4e-21 |
| Grad x Input | 0.549 ± 0.006 | 0.22 | 0.90 | +0.162 * | 94 % | 2.2e-37 |
| Random | 0.497 ± 0.004 | 0.26 | 0.03 | +0.214 * | 94 % | 1.2e-37 |
| Sampled Shapley (s0) | 0.693 ± 0.000 | 0.50 | 1.26 | +0.036 * | 66 % | 4.9e-08 |

## thumos_i3d — lstm-standard  (n = 247 vídeo-clase; vídeo mAP 0.423)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.585 ± 0.007 | 0.23 | 0.99 | — | — | — |
| Final memory | 0.510 ± 0.004 | 0.03 | -0.48 | +0.075 * | 68 % | 2.5e-10 |
| Memory writes | 0.241 ± 0.009 | 0.08 | -0.81 | +0.344 * | 94 % | 4.9e-38 |
| Murdoch-Szlam additive | 0.581 ± 0.009 | 0.26 | 0.84 | +0.004 | 49 % | 1.0e+00 |
| Murdoch-Szlam cell | 0.583 ± 0.003 | 0.39 | 0.56 | +0.002 | 46 % | 1.0e+00 |
| LRP-LSTM | 0.497 ± 0.019 | 0.16 | 0.70 | +0.088 * | 73 % | 5.0e-17 |
| Contextual Decomposition | 0.517 ± 0.004 | 0.26 | 0.29 | +0.068 * | 65 % | 1.9e-06 |
| Occlusion | 0.518 ± 0.010 | 0.18 | 0.91 | +0.067 * | 70 % | 2.9e-15 |
| Integrated Gradients | 0.507 ± 0.008 | 0.10 | 0.63 | +0.078 * | 77 % | 2.9e-15 |
| Grad x Input | 0.475 ± 0.014 | 0.09 | -0.09 | +0.110 * | 83 % | 2.2e-24 |
| Random | 0.497 ± 0.004 | 0.26 | -0.01 | +0.088 * | 80 % | 1.9e-21 |
| Sampled Shapley (s0) | 0.614 ± 0.000 | 0.32 | 1.06 | -0.019 | 42 % | 5.4e-02 |

## ucf_clip — bilstm-const200  (n = 140 vídeo-clase; vídeo AUC 0.963)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.663 ± 0.005 | 0.40 | 1.21 | — | — | — |
| Final memory | 0.648 ± 0.006 | 0.40 | 0.98 | +0.015 | 59 % | 2.5e-01 |
| Memory writes | 0.663 ± 0.005 | 0.45 | 0.97 | -0.000 | 53 % | 8.4e-01 |
| Occlusion | 0.685 ± 0.012 | 0.26 | 1.15 | -0.022 | 40 % | 2.0e-01 |
| Integrated Gradients | 0.666 ± 0.005 | 0.18 | 1.26 | -0.003 | 54 % | 6.5e-01 |
| Grad x Input | 0.658 ± 0.009 | 0.19 | 1.18 | +0.004 | 56 % | 4.4e-01 |
| Random | 0.506 ± 0.005 | 0.18 | 0.31 | +0.157 * | 76 % | 4.1e-12 |
| Sampled Shapley (s0) | 0.685 ± 0.000 | 0.41 | 1.07 | -0.015 | 46 % | 4.4e-01 |

## ucf_clip — bilstm-standard  (n = 140 vídeo-clase; vídeo AUC 0.965)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.474 ± 0.021 | 0.06 | 0.54 | — | — | — |
| Final memory | 0.400 ± 0.007 | 0.05 | 0.29 | +0.074 * | 68 % | 2.0e-05 |
| Memory writes | 0.658 ± 0.016 | 0.42 | 0.38 | -0.184 * | 25 % | 4.6e-10 |
| Occlusion | 0.573 ± 0.005 | 0.11 | 0.70 | -0.099 * | 39 % | 1.6e-04 |
| Integrated Gradients | 0.509 ± 0.016 | 0.08 | 0.62 | -0.035 * | 39 % | 3.3e-02 |
| Grad x Input | 0.570 ± 0.013 | 0.10 | 0.27 | -0.096 * | 38 % | 1.5e-03 |
| Random | 0.506 ± 0.005 | 0.18 | 0.02 | -0.031 | 47 % | 1.7e-01 |
| Sampled Shapley (s0) | 0.600 ± 0.000 | 0.19 | 0.53 | -0.124 * | 31 % | 4.5e-06 |

## ucf_clip — gru-const200  (n = 140 vídeo-clase; vídeo AUC 0.959)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.549 ± 0.004 | 0.15 | 0.83 | — | — | — |
| Final memory | 0.543 ± 0.006 | 0.09 | 0.66 | +0.006 * | 61 % | 6.0e-03 |
| Memory writes | 0.645 ± 0.010 | 0.41 | 0.99 | -0.096 * | 33 % | 4.5e-06 |
| Occlusion | 0.686 ± 0.009 | 0.34 | 1.12 | -0.137 * | 21 % | 2.0e-09 |
| Integrated Gradients | 0.654 ± 0.005 | 0.39 | 1.26 | -0.105 * | 39 % | 8.7e-05 |
| Grad x Input | 0.638 ± 0.007 | 0.32 | 1.22 | -0.089 * | 35 % | 2.4e-04 |
| Random | 0.506 ± 0.005 | 0.18 | 0.34 | +0.044 * | 59 % | 4.9e-02 |
| Sampled Shapley (s0) | 0.661 ± 0.000 | 0.41 | 1.03 | -0.117 * | 24 % | 2.3e-08 |

## ucf_clip — gru-standard  (n = 140 vídeo-clase; vídeo AUC 0.954)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.498 ± 0.006 | 0.07 | 0.22 | — | — | — |
| Final memory | 0.505 ± 0.005 | 0.05 | 0.05 | -0.007 | 52 % | 1.0e+00 |
| Memory writes | 0.599 ± 0.015 | 0.32 | 0.19 | -0.101 * | 31 % | 1.8e-06 |
| Occlusion | 0.616 ± 0.015 | 0.19 | 0.49 | -0.118 * | 31 % | 1.1e-06 |
| Integrated Gradients | 0.489 ± 0.019 | 0.08 | 0.40 | +0.010 | 54 % | 9.9e-01 |
| Grad x Input | 0.431 ± 0.020 | 0.07 | -0.09 | +0.067 | 54 % | 2.2e-01 |
| Random | 0.506 ± 0.005 | 0.18 | 0.00 | -0.007 | 49 % | 1.0e+00 |
| Sampled Shapley (s0) | 0.607 ± 0.000 | 0.21 | 0.49 | -0.104 * | 36 % | 1.1e-04 |

## ucf_clip — lstm-const200  (n = 140 vídeo-clase; vídeo AUC 0.952)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.613 ± 0.010 | 0.32 | 0.91 | — | — | — |
| Final memory | 0.620 ± 0.004 | 0.28 | 0.73 | -0.007 | 50 % | 1.0e+00 |
| Memory writes | 0.652 ± 0.004 | 0.40 | 0.78 | -0.039 * | 37 % | 1.1e-02 |
| Murdoch-Szlam additive | 0.627 ± 0.010 | 0.28 | 0.89 | -0.013 | 49 % | 1.0e+00 |
| Murdoch-Szlam cell | 0.616 ± 0.008 | 0.25 | 0.85 | -0.003 | 52 % | 1.0e+00 |
| LRP-LSTM | 0.644 ± 0.010 | 0.27 | 0.95 | -0.031 | 49 % | 1.0e+00 |
| Contextual Decomposition | 0.691 ± 0.009 | 0.39 | 0.89 | -0.077 * | 34 % | 1.0e-03 |
| Occlusion | 0.669 ± 0.011 | 0.31 | 0.96 | -0.056 * | 34 % | 5.0e-03 |
| Integrated Gradients | 0.647 ± 0.012 | 0.24 | 1.05 | -0.033 | 48 % | 1.0e+00 |
| Grad x Input | 0.631 ± 0.015 | 0.26 | 0.98 | -0.018 | 49 % | 1.0e+00 |
| Random | 0.506 ± 0.005 | 0.18 | 0.19 | +0.108 * | 68 % | 7.7e-05 |
| Sampled Shapley (s0) | 0.663 ± 0.000 | 0.41 | 1.04 | -0.064 * | 38 % | 4.0e-03 |

## ucf_clip — lstm-standard  (n = 140 vídeo-clase; vídeo AUC 0.956)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.501 ± 0.008 | 0.07 | 0.31 | — | — | — |
| Final memory | 0.502 ± 0.004 | 0.05 | 0.13 | -0.001 | 51 % | 1.0e+00 |
| Memory writes | 0.647 ± 0.017 | 0.40 | 0.30 | -0.145 * | 29 % | 1.9e-09 |
| Murdoch-Szlam additive | 0.528 ± 0.011 | 0.08 | 0.36 | -0.027 | 48 % | 8.7e-01 |
| Murdoch-Szlam cell | 0.571 ± 0.007 | 0.13 | 0.29 | -0.069 | 42 % | 1.3e-01 |
| LRP-LSTM | 0.549 ± 0.012 | 0.10 | 0.31 | -0.048 | 43 % | 1.1e-01 |
| Contextual Decomposition | 0.514 ± 0.007 | 0.11 | 0.33 | -0.012 | 48 % | 1.0e+00 |
| Occlusion | 0.600 ± 0.021 | 0.16 | 0.39 | -0.099 * | 36 % | 1.5e-04 |
| Integrated Gradients | 0.539 ± 0.014 | 0.10 | 0.31 | -0.037 | 46 % | 8.0e-01 |
| Grad x Input | 0.545 ± 0.017 | 0.12 | 0.15 | -0.044 | 39 % | 1.1e-01 |
| Random | 0.506 ± 0.005 | 0.18 | -0.01 | -0.004 | 52 % | 1.0e+00 |
| Sampled Shapley (s0) | 0.601 ± 0.000 | 0.24 | 0.38 | -0.089 * | 37 % | 3.2e-03 |

## ucf_i3d — bilstm-const200  (n = 140 vídeo-clase; vídeo AUC 0.898)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.646 ± 0.015 | 0.37 | 1.63 | — | — | — |
| Final memory | 0.640 ± 0.017 | 0.28 | 0.90 | +0.006 | 57 % | 1.0e+00 |
| Memory writes | 0.650 ± 0.018 | 0.32 | 0.91 | -0.003 | 51 % | 1.0e+00 |
| Occlusion | 0.657 ± 0.016 | 0.30 | 1.31 | -0.010 | 46 % | 8.2e-01 |
| Integrated Gradients | 0.647 ± 0.011 | 0.32 | 1.65 | -0.001 | 49 % | 1.0e+00 |
| Grad x Input | 0.613 ± 0.019 | 0.33 | 1.43 | +0.034 * | 72 % | 5.5e-06 |
| Random | 0.505 ± 0.002 | 0.18 | 0.19 | +0.141 * | 75 % | 2.1e-12 |
| Sampled Shapley (s0) | 0.658 ± 0.000 | 0.37 | 1.26 | -0.004 | 42 % | 4.9e-01 |

## ucf_i3d — bilstm-standard  (n = 140 vídeo-clase; vídeo AUC 0.875)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.483 ± 0.009 | 0.05 | 0.52 | — | — | — |
| Final memory | 0.385 ± 0.008 | 0.04 | -0.15 | +0.099 * | 67 % | 5.3e-07 |
| Memory writes | 0.653 ± 0.009 | 0.32 | 0.17 | -0.169 * | 22 % | 4.4e-11 |
| Occlusion | 0.556 ± 0.009 | 0.13 | 0.57 | -0.073 * | 33 % | 7.5e-05 |
| Integrated Gradients | 0.533 ± 0.003 | 0.08 | 0.61 | -0.050 * | 34 % | 7.5e-04 |
| Grad x Input | 0.526 ± 0.007 | 0.09 | 0.18 | -0.043 | 46 % | 2.0e-01 |
| Random | 0.505 ± 0.002 | 0.18 | -0.05 | -0.022 | 49 % | 3.1e-01 |
| Sampled Shapley (s0) | 0.575 ± 0.000 | 0.22 | 0.57 | -0.095 * | 36 % | 1.1e-05 |

## ucf_i3d — gru-const200  (n = 140 vídeo-clase; vídeo AUC 0.895)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.591 ± 0.002 | 0.24 | 1.10 | — | — | — |
| Final memory | 0.614 ± 0.005 | 0.20 | 0.68 | -0.023 | 48 % | 6.4e-01 |
| Memory writes | 0.649 ± 0.007 | 0.33 | 0.78 | -0.058 * | 38 % | 5.7e-03 |
| Occlusion | 0.668 ± 0.009 | 0.32 | 1.05 | -0.077 * | 36 % | 8.5e-05 |
| Integrated Gradients | 0.636 ± 0.009 | 0.36 | 1.36 | -0.044 * | 42 % | 1.7e-02 |
| Grad x Input | 0.611 ± 0.006 | 0.34 | 1.27 | -0.020 | 48 % | 6.4e-01 |
| Random | 0.505 ± 0.002 | 0.18 | 0.21 | +0.086 * | 66 % | 9.2e-05 |
| Sampled Shapley (s0) | 0.666 ± 0.000 | 0.41 | 1.23 | -0.072 * | 32 % | 6.6e-05 |

## ucf_i3d — gru-standard  (n = 140 vídeo-clase; vídeo AUC 0.860)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.502 ± 0.010 | 0.08 | 0.35 | — | — | — |
| Final memory | 0.507 ± 0.006 | 0.05 | -0.09 | -0.006 | 43 % | 1.0e+00 |
| Memory writes | 0.634 ± 0.010 | 0.31 | 0.05 | -0.133 * | 28 % | 3.7e-07 |
| Occlusion | 0.554 ± 0.009 | 0.15 | 0.51 | -0.052 * | 34 % | 3.6e-03 |
| Integrated Gradients | 0.483 ± 0.032 | 0.07 | 0.41 | +0.019 * | 64 % | 1.5e-02 |
| Grad x Input | 0.453 ± 0.010 | 0.08 | -0.17 | +0.048 | 56 % | 1.3e-01 |
| Random | 0.505 ± 0.002 | 0.18 | -0.03 | -0.004 | 49 % | 1.0e+00 |
| Sampled Shapley (s0) | 0.599 ± 0.000 | 0.24 | 0.64 | -0.112 * | 31 % | 6.3e-07 |

## ucf_i3d — lstm-const200  (n = 140 vídeo-clase; vídeo AUC 0.892)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.628 ± 0.011 | 0.34 | 1.30 | — | — | — |
| Final memory | 0.645 ± 0.008 | 0.28 | 0.89 | -0.017 | 53 % | 1.0e+00 |
| Memory writes | 0.653 ± 0.006 | 0.33 | 0.92 | -0.025 | 46 % | 1.0e+00 |
| Murdoch-Szlam additive | 0.648 ± 0.014 | 0.32 | 1.25 | -0.020 | 48 % | 1.0e+00 |
| Murdoch-Szlam cell | 0.625 ± 0.009 | 0.30 | 1.16 | +0.003 | 54 % | 1.0e+00 |
| LRP-LSTM | 0.643 ± 0.010 | 0.34 | 1.31 | -0.016 | 48 % | 1.0e+00 |
| Contextual Decomposition | 0.681 ± 0.006 | 0.40 | 1.10 | -0.053 * | 36 % | 5.5e-04 |
| Occlusion | 0.657 ± 0.011 | 0.33 | 1.12 | -0.029 | 39 % | 7.4e-02 |
| Integrated Gradients | 0.647 ± 0.006 | 0.30 | 1.41 | -0.019 | 44 % | 1.0e+00 |
| Grad x Input | 0.612 ± 0.010 | 0.29 | 1.22 | +0.015 | 56 % | 1.0e+00 |
| Random | 0.505 ± 0.002 | 0.18 | 0.17 | +0.122 * | 70 % | 2.3e-08 |
| Sampled Shapley (s0) | 0.654 ± 0.000 | 0.42 | 1.08 | -0.012 | 46 % | 1.0e+00 |

## ucf_i3d — lstm-standard  (n = 140 vídeo-clase; vídeo AUC 0.871)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.514 ± 0.015 | 0.08 | 0.53 | — | — | — |
| Final memory | 0.502 ± 0.002 | 0.05 | -0.09 | +0.012 | 58 % | 7.4e-01 |
| Memory writes | 0.649 ± 0.011 | 0.31 | 0.24 | -0.136 * | 24 % | 5.1e-09 |
| Murdoch-Szlam additive | 0.531 ± 0.015 | 0.09 | 0.57 | -0.017 | 45 % | 1.9e-01 |
| Murdoch-Szlam cell | 0.549 ± 0.006 | 0.19 | 0.37 | -0.035 | 46 % | 7.4e-01 |
| LRP-LSTM | 0.558 ± 0.010 | 0.15 | 0.49 | -0.044 * | 39 % | 4.3e-02 |
| Contextual Decomposition | 0.521 ± 0.019 | 0.13 | 0.52 | -0.008 | 47 % | 1.0e+00 |
| Occlusion | 0.553 ± 0.020 | 0.15 | 0.53 | -0.039 | 41 % | 7.1e-02 |
| Integrated Gradients | 0.532 ± 0.014 | 0.14 | 0.43 | -0.019 | 49 % | 1.0e+00 |
| Grad x Input | 0.516 ± 0.017 | 0.14 | 0.09 | -0.003 | 47 % | 1.0e+00 |
| Random | 0.505 ± 0.002 | 0.18 | -0.04 | +0.008 | 51 % | 1.0e+00 |
| Sampled Shapley (s0) | 0.576 ± 0.000 | 0.21 | 0.52 | -0.076 * | 36 % | 2.2e-03 |

## xd_clip — bilstm-const200  (n = 486 vídeo-clase; vídeo AUC 0.995)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.727 ± 0.005 | 0.78 | 4.89 | — | — | — |
| Final memory | 0.700 ± 0.004 | 0.75 | 4.11 | +0.027 * | 63 % | 2.7e-09 |
| Memory writes | 0.705 ± 0.005 | 0.76 | 4.13 | +0.022 * | 61 % | 6.2e-07 |
| Occlusion | 0.705 ± 0.007 | 0.50 | 3.52 | +0.022 | 51 % | 4.3e-01 |
| Integrated Gradients | 0.709 ± 0.005 | 0.53 | 5.01 | +0.018 * | 67 % | 3.7e-15 |
| Grad x Input | 0.678 ± 0.011 | 0.50 | 4.49 | +0.049 * | 75 % | 8.5e-34 |
| Random | 0.502 ± 0.005 | 0.40 | 0.67 | +0.225 * | 90 % | 6.0e-64 |
| Sampled Shapley (s0) | 0.716 ± 0.000 | 0.67 | 3.98 | +0.007 | 49 % | 9.7e-01 |

## xd_clip — bilstm-standard  (n = 486 vídeo-clase; vídeo AUC 0.986)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.538 ± 0.011 | 0.24 | 1.47 | — | — | — |
| Final memory | 0.498 ± 0.010 | 0.22 | 0.90 | +0.041 * | 57 % | 3.6e-06 |
| Memory writes | 0.717 ± 0.010 | 0.76 | 1.35 | -0.179 * | 27 % | 7.5e-31 |
| Occlusion | 0.665 ± 0.010 | 0.41 | 1.49 | -0.126 * | 25 % | 1.3e-27 |
| Integrated Gradients | 0.663 ± 0.014 | 0.38 | 1.40 | -0.124 * | 35 % | 1.4e-15 |
| Grad x Input | 0.609 ± 0.013 | 0.38 | 0.78 | -0.071 * | 46 % | 5.0e-03 |
| Random | 0.502 ± 0.005 | 0.40 | -0.01 | +0.036 * | 59 % | 1.1e-04 |
| Sampled Shapley (s0) | 0.707 ± 0.000 | 0.62 | 1.50 | -0.149 * | 20 % | 1.0e-36 |

## xd_clip — gru-const200  (n = 486 vídeo-clase; vídeo AUC 0.994)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.613 ± 0.011 | 0.41 | 2.82 | — | — | — |
| Final memory | 0.571 ± 0.008 | 0.31 | 2.13 | +0.043 * | 66 % | 2.3e-17 |
| Memory writes | 0.691 ± 0.011 | 0.77 | 3.45 | -0.078 * | 36 % | 9.8e-13 |
| Occlusion | 0.748 ± 0.010 | 0.66 | 3.01 | -0.135 * | 23 % | 1.2e-36 |
| Integrated Gradients | 0.726 ± 0.006 | 0.77 | 4.39 | -0.113 * | 30 % | 3.0e-23 |
| Grad x Input | 0.695 ± 0.011 | 0.74 | 4.15 | -0.082 * | 34 % | 2.9e-15 |
| Random | 0.502 ± 0.005 | 0.40 | 0.63 | +0.111 * | 75 % | 3.9e-25 |
| Sampled Shapley (s0) | 0.746 ± 0.000 | 0.76 | 4.88 | -0.143 * | 23 % | 9.5e-39 |

## xd_clip — gru-standard  (n = 486 vídeo-clase; vídeo AUC 0.980)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.552 ± 0.003 | 0.18 | 0.93 | — | — | — |
| Final memory | 0.525 ± 0.002 | 0.16 | 0.41 | +0.027 * | 63 % | 4.0e-10 |
| Memory writes | 0.592 ± 0.006 | 0.63 | 0.62 | -0.041 | 48 % | 1.0e-01 |
| Occlusion | 0.684 ± 0.002 | 0.50 | 1.36 | -0.133 * | 30 % | 1.7e-24 |
| Integrated Gradients | 0.573 ± 0.011 | 0.36 | 0.86 | -0.022 | 50 % | 4.8e-01 |
| Grad x Input | 0.481 ± 0.008 | 0.41 | 0.13 | +0.071 * | 61 % | 9.0e-06 |
| Random | 0.502 ± 0.005 | 0.40 | 0.04 | +0.050 * | 63 % | 1.7e-07 |
| Sampled Shapley (s0) | 0.680 ± 0.000 | 0.61 | 1.70 | -0.129 * | 29 % | 4.8e-25 |

## xd_clip — lstm-const200  (n = 486 vídeo-clase; vídeo AUC 0.994)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.707 ± 0.004 | 0.72 | 2.01 | — | — | — |
| Final memory | 0.695 ± 0.009 | 0.70 | 1.73 | +0.013 * | 61 % | 7.4e-05 |
| Memory writes | 0.715 ± 0.008 | 0.75 | 1.94 | -0.007 | 53 % | 9.2e-01 |
| Murdoch-Szlam additive | 0.711 ± 0.004 | 0.71 | 1.95 | -0.004 | 44 % | 2.1e-01 |
| Murdoch-Szlam cell | 0.697 ± 0.006 | 0.66 | 1.85 | +0.010 | 48 % | 9.2e-01 |
| LRP-LSTM | 0.725 ± 0.004 | 0.71 | 2.20 | -0.018 * | 42 % | 1.8e-02 |
| Contextual Decomposition | 0.765 ± 0.002 | 0.70 | 1.59 | -0.057 * | 33 % | 1.3e-15 |
| Occlusion | 0.736 ± 0.005 | 0.59 | 1.61 | -0.029 * | 36 % | 3.2e-07 |
| Integrated Gradients | 0.724 ± 0.004 | 0.58 | 2.29 | -0.017 * | 42 % | 5.4e-03 |
| Grad x Input | 0.696 ± 0.006 | 0.55 | 2.00 | +0.011 | 53 % | 2.1e-01 |
| Random | 0.502 ± 0.005 | 0.40 | 0.29 | +0.206 * | 88 % | 2.2e-57 |
| Sampled Shapley (s0) | 0.739 ± 0.000 | 0.70 | 1.40 | -0.028 * | 36 % | 2.6e-09 |

## xd_clip — lstm-standard  (n = 486 vídeo-clase; vídeo AUC 0.981)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.558 ± 0.007 | 0.18 | 0.98 | — | — | — |
| Final memory | 0.542 ± 0.003 | 0.16 | 0.55 | +0.016 * | 51 % | 9.4e-03 |
| Memory writes | 0.704 ± 0.009 | 0.72 | 1.07 | -0.146 * | 27 % | 9.3e-30 |
| Murdoch-Szlam additive | 0.579 ± 0.005 | 0.25 | 1.25 | -0.021 * | 42 % | 6.6e-03 |
| Murdoch-Szlam cell | 0.590 ± 0.004 | 0.49 | 1.17 | -0.031 * | 46 % | 4.5e-02 |
| LRP-LSTM | 0.616 ± 0.010 | 0.25 | 1.13 | -0.058 * | 41 % | 1.4e-04 |
| Contextual Decomposition | 0.639 ± 0.006 | 0.48 | 0.90 | -0.081 * | 35 % | 6.1e-11 |
| Occlusion | 0.663 ± 0.011 | 0.45 | 1.20 | -0.105 * | 31 % | 1.4e-15 |
| Integrated Gradients | 0.598 ± 0.012 | 0.37 | 0.81 | -0.040 * | 46 % | 4.5e-02 |
| Grad x Input | 0.548 ± 0.009 | 0.42 | 0.35 | +0.010 | 52 % | 3.5e-01 |
| Random | 0.502 ± 0.005 | 0.40 | 0.02 | +0.057 * | 64 % | 2.6e-08 |
| Sampled Shapley (s0) | 0.690 ± 0.000 | 0.62 | 1.77 | -0.138 * | 30 % | 1.1e-24 |

## xd_i3d — bilstm-const200  (n = 486 vídeo-clase; vídeo AUC 0.986)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.684 ± 0.011 | 0.69 | 3.72 | — | — | — |
| Final memory | 0.654 ± 0.010 | 0.58 | 2.27 | +0.030 * | 59 % | 7.2e-05 |
| Memory writes | 0.666 ± 0.007 | 0.59 | 2.26 | +0.017 * | 55 % | 4.6e-02 |
| Occlusion | 0.667 ± 0.015 | 0.56 | 2.56 | +0.017 * | 54 % | 4.6e-02 |
| Integrated Gradients | 0.696 ± 0.004 | 0.64 | 3.82 | -0.013 * | 48 % | 4.6e-02 |
| Grad x Input | 0.643 ± 0.015 | 0.61 | 3.04 | +0.041 * | 75 % | 2.3e-26 |
| Random | 0.504 ± 0.003 | 0.41 | 0.25 | +0.179 * | 85 % | 2.3e-53 |
| Sampled Shapley (s0) | 0.697 ± 0.000 | 0.70 | 2.30 | -0.017 * | 37 % | 2.4e-05 |

## xd_i3d — bilstm-standard  (n = 486 vídeo-clase; vídeo AUC 0.964)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.548 ± 0.016 | 0.27 | 1.43 | — | — | — |
| Final memory | 0.465 ± 0.009 | 0.21 | 0.36 | +0.083 * | 71 % | 6.2e-25 |
| Memory writes | 0.699 ± 0.005 | 0.60 | 1.05 | -0.151 * | 27 % | 2.1e-31 |
| Occlusion | 0.640 ± 0.007 | 0.41 | 1.52 | -0.092 * | 30 % | 1.8e-19 |
| Integrated Gradients | 0.658 ± 0.005 | 0.43 | 1.17 | -0.110 * | 35 % | 3.5e-13 |
| Grad x Input | 0.566 ± 0.009 | 0.36 | 0.41 | -0.018 | 53 % | 8.8e-01 |
| Random | 0.504 ± 0.003 | 0.41 | -0.05 | +0.044 * | 60 % | 1.1e-06 |
| Sampled Shapley (s0) | 0.691 ± 0.000 | 0.58 | 1.30 | -0.159 * | 23 % | 1.6e-38 |

## xd_i3d — gru-const200  (n = 486 vídeo-clase; vídeo AUC 0.982)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.607 ± 0.011 | 0.49 | 2.24 | — | — | — |
| Final memory | 0.582 ± 0.007 | 0.41 | 1.42 | +0.025 * | 61 % | 2.5e-08 |
| Memory writes | 0.656 ± 0.005 | 0.60 | 2.10 | -0.049 * | 43 % | 3.2e-05 |
| Occlusion | 0.678 ± 0.010 | 0.63 | 2.34 | -0.071 * | 32 % | 5.2e-15 |
| Integrated Gradients | 0.693 ± 0.006 | 0.70 | 3.60 | -0.086 * | 35 % | 8.6e-17 |
| Grad x Input | 0.640 ± 0.010 | 0.67 | 3.16 | -0.033 * | 45 % | 2.0e-03 |
| Random | 0.504 ± 0.003 | 0.41 | 0.41 | +0.103 * | 72 % | 9.3e-24 |
| Sampled Shapley (s0) | 0.720 ± 0.000 | 0.69 | 2.14 | -0.093 * | 31 % | 5.8e-20 |

## xd_i3d — gru-standard  (n = 486 vídeo-clase; vídeo AUC 0.950)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.535 ± 0.004 | 0.22 | 0.90 | — | — | — |
| Final memory | 0.520 ± 0.004 | 0.16 | 0.16 | +0.015 * | 57 % | 1.4e-03 |
| Memory writes | 0.628 ± 0.021 | 0.56 | 0.57 | -0.093 * | 36 % | 5.5e-12 |
| Occlusion | 0.626 ± 0.006 | 0.44 | 1.52 | -0.091 * | 35 % | 3.1e-16 |
| Integrated Gradients | 0.577 ± 0.010 | 0.41 | 1.16 | -0.042 * | 45 % | 3.0e-02 |
| Grad x Input | 0.512 ± 0.006 | 0.45 | 0.50 | +0.023 * | 55 % | 3.0e-02 |
| Random | 0.504 ± 0.003 | 0.41 | 0.00 | +0.031 * | 59 % | 5.7e-04 |
| Sampled Shapley (s0) | 0.680 ± 0.000 | 0.59 | 1.61 | -0.141 * | 27 % | 1.6e-25 |

## xd_i3d — lstm-const200  (n = 486 vídeo-clase; vídeo AUC 0.984)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.681 ± 0.010 | 0.67 | 2.53 | — | — | — |
| Final memory | 0.668 ± 0.006 | 0.56 | 1.61 | +0.013 | 53 % | 2.1e-01 |
| Memory writes | 0.686 ± 0.006 | 0.62 | 1.82 | -0.004 | 50 % | 9.9e-01 |
| Murdoch-Szlam additive | 0.677 ± 0.009 | 0.66 | 2.53 | +0.004 | 54 % | 9.9e-01 |
| Murdoch-Szlam cell | 0.655 ± 0.008 | 0.63 | 2.29 | +0.027 * | 62 % | 9.2e-06 |
| LRP-LSTM | 0.672 ± 0.004 | 0.62 | 2.59 | +0.010 * | 57 % | 4.3e-02 |
| Contextual Decomposition | 0.721 ± 0.007 | 0.64 | 1.96 | -0.040 * | 40 % | 2.3e-06 |
| Occlusion | 0.668 ± 0.003 | 0.56 | 1.90 | +0.014 | 52 % | 2.1e-01 |
| Integrated Gradients | 0.700 ± 0.005 | 0.65 | 2.91 | -0.018 | 48 % | 2.1e-01 |
| Grad x Input | 0.639 ± 0.003 | 0.62 | 2.24 | +0.043 * | 68 % | 4.6e-16 |
| Random | 0.504 ± 0.003 | 0.41 | 0.23 | +0.177 * | 84 % | 3.0e-51 |
| Sampled Shapley (s0) | 0.697 ± 0.000 | 0.71 | 2.08 | -0.012 | 44 % | 6.4e-02 |

## xd_i3d — lstm-standard  (n = 486 vídeo-clase; vídeo AUC 0.941)

| Método | AUC (media ± sd semillas) | Pointing | AOPC | Δ AUC (CAM − método) | CAM gana | p Holm |
|---|---|---|---|---|---|---|
| **Temporal CAM (ours)** | 0.577 ± 0.007 | 0.27 | 1.09 | — | — | — |
| Final memory | 0.534 ± 0.007 | 0.17 | 0.18 | +0.043 * | 63 % | 2.5e-09 |
| Memory writes | 0.693 ± 0.007 | 0.59 | 0.84 | -0.116 * | 30 % | 6.3e-22 |
| Murdoch-Szlam additive | 0.590 ± 0.006 | 0.31 | 1.26 | -0.013 * | 45 % | 6.6e-03 |
| Murdoch-Szlam cell | 0.584 ± 0.001 | 0.53 | 1.21 | -0.006 | 48 % | 8.5e-01 |
| LRP-LSTM | 0.548 ± 0.012 | 0.29 | 0.76 | +0.030 * | 56 % | 2.0e-02 |
| Contextual Decomposition | 0.627 ± 0.007 | 0.50 | 0.72 | -0.050 * | 42 % | 2.7e-04 |
| Occlusion | 0.631 ± 0.007 | 0.44 | 1.29 | -0.054 * | 38 % | 1.5e-08 |
| Integrated Gradients | 0.584 ± 0.012 | 0.40 | 0.78 | -0.006 | 50 % | 9.1e-01 |
| Grad x Input | 0.508 ± 0.008 | 0.40 | 0.28 | +0.069 * | 60 % | 7.1e-06 |
| Random | 0.504 ± 0.003 | 0.41 | -0.04 | +0.073 * | 69 % | 6.0e-15 |
| Sampled Shapley (s0) | 0.683 ± 0.000 | 0.60 | 1.57 | -0.095 * | 31 % | 8.6e-19 |