1) The ToeSegmentation1 dataset was chosen as it yielded the largest gap in the test accuracy of 
0.2368. For this dataset, the Greedy Accuracy selector chose 2 classifiers, minirocket and catch22,
and together they significantly outperformed the full-pool ensemble strategy. A core reason for this
is due to the chosen classifiers, and the full-pool strategy averages out the output from all classifiers, 
both the fast and the slow, resulting in a comparatively low performance in comparison to the greedy 
selector which chose 2 high performing classifiers, those mentioned above, resulting in a higher
average. Minirocket uses random convolutional kernels, whereas catch22 uses 22 hand-crafted time-series 
features. The difference in the strategies used by each classifier could mean that any errors made
by either would not likely be done by the other, preventing repeated errors from drastically impacting
performance and potentially covering wherever the other classifier fell short.
2) When comparing the greedy selector across the datasets, the selector using accuracy as a metric appears to 
on average be less accurate as the selector using log_loss as a metric, only exceeding in accuracy in
1 dataset and sharing the same accuracy in 2 datasets. In 5 of the datasets (Computers, DistalPhalanxOutlineAgeGroup,
ECG200, Herring, InsectWingBeatSound) the log loss selector chose more classifiers than the accuracy selector,
the same number as accuracy selector in 1 dataset, and fewer classifiers than the Accuracy selector
in 2 datasets. The dataset with equal number of classifiers chosen and one of the datasets with log 
loss with fewer classifiers (ToeSegmentation) were where they had an equivalent accuracy. The log loss 
selector also seemed to have chosen the minirocket classifier more often than the accuracy selector (out of
11 decisions to choose minirocket, 7 of them were made by the log loss selector)
3) No. Within all of the datasets there is a difference between the oof_accuracy and greedy_accuracy, 
what makes it an unreliable estimate however is that the difference is not consistent and can very from being 
relatively large to being relatively small (gap in distalPhalanxTW was significantly larger than gap in ECG200). 
This variation in their differences makes the oof_accuracy an unreliable estimate of how accurate the 
ensemble selected by the greedy_accuracy selector would perform. This inconsistent gap between the 2 
could be a result of the principles behind them both, as oof_accuracy would aim to provide an unbiased 
score of accuracy, but since the greedy selector actively chooses which classifiers to select to boost 
the accuracy this may make it "unbiased" and result in the inconsistencies observed.