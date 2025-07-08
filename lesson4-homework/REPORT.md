# Отчет по домашнему заданию к уроку 4: Сверточные сети

## Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)

### 1.1 Сравнение на MNIST (20 баллов)

*Таблица 1. Сравнение производительности моделей на датасете MNIST:*

| №  |     model     | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|---------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |      FCN      |   109386   |  0.0801   |    0.0204   |    0.9834     |    0.9944      |  237.937  |
| 2  |  Simplie СNN  |   421642   |  0.0326   |    0.0082   |    0.9928     |    0.9971      |  246.039  |     
| 3  | Residual CNN  |   160906   |  0.0205   |    0.0068   |    0.9940     |    0.9977      |  332.674  |                                  

![FCNModelLearningCurveOnMNIST.png](/lesson4-homework/plots/FCNModelLearningCurveOnMNIST.png)

*Рисунок 1. Кривая обучения для FCN*

![SimplieCNNModelLearningCurveOnMNIST.png](/lesson4-homework/plots/SimplieCNNModelLearningCurveOnMNIST.png)

*Рисунок 2. Кривая обучения для Simplie СNN*

![ResidualCNNModelLearningCurveOnMNIST.png](/lesson4-homework/plots/ResidualCNNModelLearningCurveOnMNIST.png)

*Рисунок 3. Кривая обучения для Residual CNN*

![CompareFCNAndSimplieCNNOnTest.png](/lesson4-homework/plots/CompareFCNAndSimplieCNNOnTest.png)

*Рисунок 4. Сравнение кривых обучения для FCN и Simplie CNN*

![CompareFCNAndResidualCNNOnTest.png](/lesson4-homework/plots/CompareFCNAndResidualCNNOnTest.png)

*Рисунок 5. Сравнение кривых обучения для FCN и Residual CNN*


### 1.2 Сравнение на CIFAR-10 (20 баллов)

*Таблица 2. Сравнение производительности моделей на датасете CIFAR-10:*

| №  |          model            | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|---------------------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |           FCN             |   1462538  |  1.7803   |    0.6946   |    0.5340     |    0.7507      |  247.484  |
| 2  |       Residual CNN        |   160906   |  0.6454   |    0.1486   |    0.8193     |    0.9483      |  363.748  |     
| 3  | Regularized Residual CNN  |   678058   |  0.4998   |    0.5074   |    0.8349     |    0.8224      |  407.651  | 

![FCNModelLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/FCNModelLearningCurveOnCIFAR-10.png)

*Рисунок 6. Кривая обучения для FCN*

![ResidualCNNModelLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/ResidualCNNModelLearningCurveOnCIFAR-10.png)

*Рисунок 7. Кривая обучения для Residual CNN*

![/RegularizedResidualCNNModelLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/RegularizedResidualCNNModelLearningCurveOnCIFAR-10.png)

*Рисунок 8. Кривая обучения для Residual CNN*

**Вывод:** Модель с регуляризацией не переобучается, в отличие от FCN и Residual CNN.

![ConfusionMatrixForFCNOnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForFCNOnCIFAR-10.png)

*Рисунок 9. Confusion matrix для FCN*

![ConfusionMatrixForResidualCNNOnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForResidualCNNOnCIFAR-10.png)

*Рисунок 10. Confusion matrix для Residual CNN*

![ConfusionMatrixForRegularizedResidualCNNOnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForRegularizedResidualCNNOnCIFAR-10.png)

*Рисунок 11. Confusion matrix для Regularized Residual CNN*

## Задание 2: Анализ архитектур CNN (30 баллов)

### 2.1 Влияние размера ядра свертки (15 баллов)

*Таблица 3. Сравнение производительности моделей на датасете CIFAR-10:*

| №  |               model                   | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|---------------------------------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |    CNN model with kernel_size=3x3     |   161450   |  0.5521   |    0.2682   |    0.9035     |    0.9035      |  385.420  |
| 2  |    CNN model with kernel_size=5x5     |   161450   |  0.5939   |    0.2791   |    0.8089     |    0.9017      |  395.982  |     
| 3  |    CNN model with kernel_size=7x7     |   202410   |  0.6282   |    0.2776   |    0.8067     |    0.9016      |  436.560  |
| 4  | CNN model with kernel_size=(1x1+3x3)  |   142154   |  0.6413   |    0.2997   |    0.8038     |    0.8925      |  378.796  |

![CNNModelWithKernelSize3x3LearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNModelWithKernelSize3x3LearningCurveOnCIFAR-10.png)

*Рисунок 12. Кривая обучения для CNN model with kernel_size=3x3*

![CNNModelWithKernelSize5x5LearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNModelWithKernelSize5x5LearningCurveOnCIFAR-10.png)

*Рисунок 13. Кривая обучения для CNN model with kernel_size=5x5*

![CNNModelWithKernelSize7x7LearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNModelWithKernelSize7x7LearningCurveOnCIFAR-10.png)

*Рисунок 14. Кривая обучения для CNN model with kernel_size=7x7*

![CNNModelWithKernelSize1x13x3LearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNModelWithKernelSize1x13x3LearningCurveOnCIFAR-10.png)

*Рисунок 15. Кривая обучения для CNN model with kernel_size=(1x1+3x3)*

![ConfusionMatrixForCNNWithKernelSize3x3OnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForCNNWithKernelSize3x3OnCIFAR-10.png)

*Рисунок 16. Confusion matrix для CNN model with kernel_size=3x3*

![ConfusionMatrixForCNNWithKernelSize5x5OnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForCNNWithKernelSize5x5OnCIFAR-10.png)

*Рисунок 17. Confusion matrix для CNN model with kernel_size=5x5*

![ConfusionMatrixForCNNWithKernelSize7x7OnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForCNNWithKernelSize7x7OnCIFAR-10.png)

*Рисунок 18. Confusion matrix для CNN model with kernel_size=7x7*

![ConfusionMatrixForCNNWithKernelSize1x13x3OnCIFAR-10.png](/lesson4-homework/plots/ConfusionMatrixForCNNWithKernelSize1x13x3OnCIFAR-10.png)

*Рисунок 19. Confusion matrix для CNN model with kernel_size=(1x1+3x3)*

![FirstLayerActivationForCNNWithKernelSize3x3.png](/lesson4-homework/plots/FirstLayerActivationForCNNWithKernelSize3x3.png)

*Рисунок 20. Визуализация активации первого слоя для CNN model with kernel_size=3x3*

![FirstLayerActivationForCNNWithKernelSize5x5.png](/lesson4-homework/plots/FirstLayerActivationForCNNWithKernelSize5x5.png)

*Рисунок 21. Визуализация активации первого слоя для CNN model with kernel_size=5x5*

![FirstLayerActivationForCNNWithKernelSize7x7.png](/lesson4-homework/plots/FirstLayerActivationForCNNWithKernelSize7x7.png)

*Рисунок 22. Визуализация активации первого слоя для CNN model with kernel_size=7x7*

![FirstLayerActivationForCNNWithKernelSize1x13x3.png](/lesson4-homework/plots/FirstLayerActivationForCNNWithKernelSize1x13x3.png)

*Рисунок 23. Визуализация активации первого слоя для CNN model with kernel_size=(1x1+3x3)*

### 2.2 Влияние глубины CNN (15 баллов)

*Таблица 4. Сравнение производительности моделей на датасете CIFAR-10:*

| №  |               model            | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|--------------------------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |    CNN model with 2 layers     |   545098   |  1.1359   |    0.2986   |    0.7227     |    0.8905      |  258.607  |
| 2  |    CNN model with 4 layers     |  1439818   |  1.0578   |    0.1069   |    0.7918     |    0.9626      |  299.096  |     
| 3  |    CNN model with 6 layers     |   814122   |  1.1912   |    0.1405   |    0.7576     |    0.9506      |  299.874  |
| 4  |    CNN model with residual     |   327882   |  0.8364   |    0.0504   |    0.8147     |    0.9836      |  370.106  |

![CNNModelWith2LayersLearningCurveOcCIFAR-10.png](/lesson4-homework/plots/CNNModelWith2LayersLearningCurveOcCIFAR-10.png)

*Рисунок 24. Кривая обучения для CNN model with 2 layers*

![CNNModelWith4LayersLearningCurveOcCIFAR-10.png](/lesson4-homework/plots/CNNModelWith4LayersLearningCurveOcCIFAR-10.png)

*Рисунок 25. Кривая обучения для CNN model with 4 layers*

![CNNModelWith6LayersLearningCurveOcCIFAR-10.png](/lesson4-homework/plots/CNNModelWith6LayersLearningCurveOcCIFAR-10.png)

*Рисунок 26. Кривая обучения для CNN model with 6 layers*

![CNNModelWithResidualLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNModelWithResidualLearningCurveOnCIFAR-10.png)

*Рисунок 27. Кривая обучения для CNN model with residual*

![FeatureMapsForCNNModelWith2LayersOnCIFAR-10.png](/lesson4-homework/plots/FeatureMapsForCNNModelWith2LayersOnCIFAR-10.png)

*Рисунок 28. Feature maps for CNN model with 2 layers on CIFAR-10*

![FeatureMapsForCNNModelWith4LayersOnCIFAR-10.png](/lesson4-homework/plots/FeatureMapsForCNNModelWith4LayersOnCIFAR-10.png)

*Рисунок 29. Feature maps for CNN model with 4 layers on CIFAR-10*

![FeatureMapsForCNNModelWith6LayersOnCIFAR-10.png](/lesson4-homework/plots/FeatureMapsForCNNModelWith6LayersOnCIFAR-10.png)

*Рисунок 30. Feature maps for CNN model with 6 layers on CIFAR-10*

![FeatureMapsForCNNModelWithResidualOnCIFAR-10.png](/lesson4-homework/plots/FeatureMapsForCNNModelWithResidualOnCIFAR-10.png)

*Рисунок 31. Feature maps for CNN model with residual on CIFAR-10*

## Задание 3: Кастомные слои и эксперименты (30 баллов)

### 3.1 Реализация кастомных слоев (15 баллов)

*Таблица 5. Сравнение производительности моделей на датасете CIFAR-10:*

| №  |               model            | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|--------------------------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |    CNN with custom layers      |   268650   |  1.5861   |    0.1838   |    0.6730     |    0.9345      |  254.547  |
| 2  |      CNN with attention        |   268810   |  1.2919   |    0.2197   |    0.7069     |    0.9230      |  264.053  |     
| 3  |   CNN with custom activation   |   268810   |  2.1056   |    0.0519   |    0.6774     |    0.9838      |  261.071  |
| 4  |    CNN with custom pooling     |   268650   |  1.6854   |    0.1489   |    0.6878     |    0.9471      |  291.747  |

![CNNWithCustomLayersLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithCustomLayersLearningCurveOnCIFAR-10.png)

*Рисунок 32. Кривая обучения для CNN with custom layers*

![CNNWithAttentionLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithAttentionLearningCurveOnCIFAR-10.png)

*Рисунок 33. Кривая обучения для CNN with attention*

![CNNWithCustomActivationLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithCustomActivationLearningCurveOnCIFAR-10.png)

*Рисунок 34. Кривая обучения для CNN with custom activation*

![CNNWithCustomPoolingLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithCustomPoolingLearningCurveOnCIFAR-10.png)

*Рисунок 35. Кривая обучения для CNN with custom pooling*

### 3.2 Эксперименты с Residual блоками (15 баллов)

*Таблица 6. Сравнение производительности моделей на датасете CIFAR-10:*

| №  |               model            | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
|----|--------------------------------|------------|-----------|-------------|---------------|----------------|-----------|
| 1  |    CNN with base Residual      |   161482   |  0.6969   |    0.1434   |    0.8148     |    0.9500      |  379.705  |
| 2  |  CNN with Bottleneck Residual  |  23520842  |  0.5511   |    0.1952   |    0.8341     |    0.9319      | 1890.913  |     
| 3  |    CNN with Wide Residual      |  2749786   |  0.5300   |    0.1940   |    0.8462     |    0.9321      |  454.309  |

![CNNWithBaseResidualLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithBaseResidualLearningCurveOnCIFAR-10.png)

*Рисунок 36. Кривая обучения для CNN with base Residual*

![CNNWithBottleneckResidualLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithBottleneckResidualLearningCurveOnCIFAR-10.png)

*Рисунок 37. Кривая обучения для CNN with Bottleneck Residual*

![CNNWithWideResidualLearningCurveOnCIFAR-10.png](/lesson4-homework/plots/CNNWithWideResidualLearningCurveOnCIFAR-10.png)

*Рисунок 38. Кривая обучения для CNN with Wide Residual*