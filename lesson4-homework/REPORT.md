# Отчет по домашнему заданию к уроку 4: Сверточные сети

## Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)

### 1.1 Сравнение на MNIST (20 баллов)

*Таблица 1. Сравнение производительности моделей на датасете MNIST:*
```
| №  |     model     | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
| 1  |      FCN      |   109386   |  0.0801   |    0.0204   |    0.9834     |    0.9944      |  237.937  |
| 2  |  Simplie СNN  |   421642   |  0.0326   |    0.0082   |    0.9928     |    0.9971      |  246.039  |     
| 3  | Residual CNN  |   160906   |  0.0205   |    0.0068   |    0.9940     |    0.9977      |  332.674  |                                  
```

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
```
| №  |          model            | parameters | test loss |  train loss | test accuracy | train accuracy |  time (c) |
| 1  |           FCN             |   1462538  |  1.7803   |    0.6946   |    0.5340     |    0.7507      |  247.484  |
| 2  |       Residual CNN        |   160906   |  0.6454   |    0.1486   |    0.8193     |    0.9483      |  363.748  |     
| 3  | Regularized Residual CNN  |   678058   |  0.4998   |    0.5074   |    0.8349     |    0.8224      |  407.651  | 
```

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
