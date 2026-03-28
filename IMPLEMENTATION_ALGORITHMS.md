# Implementation Algorithms for Master's Thesis

This document contains the formalized pseudocode for each machine learning and deep learning architecture implemented in the MedVision-AI framework.

---

### Algorithm 4-1: Implementation KNN (K-Nearest Neighbors)
1: **Preparation**(*train_features, test_features, validation_features*)
2:      **return** *train_normalization, test_normalization, validation_normalization*
3: **for** *k* from 1 to 40 **do**
4:      *KNN* $\leftarrow$ **KNeighborsClassifier**(*n_neighbors* $\leftarrow k$, *metric* $\leftarrow$ 'euclidean')
5:      *Score* $\leftarrow$ **CrossValidate**(*KNN, train_normalization*)
6:      **if** *Score* > *maxScore* **then**
7:          *maxScore* $\leftarrow$ *Score*
8:          *bestK* $\leftarrow k$
9: *Model* $\leftarrow$ **KNeighborsClassifier**(*n_neighbors* $\leftarrow bestK$)
10: *Model.Fit*(*train_normalization, train_labels*)
11: *Model.Evaluate*(*test_normalization*)

---

### Algorithm 4-2: Implementation VGG16
1: **Preparation**(*train_features, test_features, validation_features*)
2:      **return** *train_normalization, test_normalization, validation_normalization*
3: *VGG16* $\leftarrow$ **load**(*weights* $\leftarrow$ *ImageNet*)
4: **for** each *VGG16.Layers* **do**
5:      *layersTrainable* $\leftarrow$ **FALSE**
6: *lastOutput* $\leftarrow$ *VGG16.Layers*[-1].*output*
7: *Output* $\leftarrow$ **Flatten**()(*lastOutput*)
8: *Output* $\leftarrow$ **Dense**(512, *activation* $\leftarrow$ *ReLU*)(*Output*)
9: *Output* $\leftarrow$ **Dropout**(0.3)(*Output*)
10: *Output* $\leftarrow$ **Dense**(2, *activation* $\leftarrow$ *Softmax*)(*Output*)
11: *Model* $\leftarrow$ **Model**(*VGG16.Input, Output*)
12: *Model.Compile*(*Loss, Optimizer, Metrics*)
13: *Model.Fit*(*train_normalization, EPOCH* $\leftarrow$ 25, *validation_normalization, CALLBACKS*)
14: *Model.Evaluate*(*test_normalization*)

---

### Algorithm 4-3: Implementation VGG19
1: **Preparation**(*train_features, test_features, validation_features*)
2:      **return** *train_normalization, test_normalization, validation_normalization*
3: *VGG19* $\leftarrow$ **load**(*weights* $\leftarrow$ *ImageNet*)
4: **for** each *VGG19.Layers* **do**
5:      *layersTrainable* $\leftarrow$ **FALSE**
6: *lastOutput* $\leftarrow$ *VGG19.Layers*[-1].*output*
7: *Output* $\leftarrow$ **Flatten**()(*lastOutput*)
8: *Output* $\leftarrow$ **Dense**(128, *activation* $\leftarrow$ *ReLU*)(*Output*)
9: *Output* $\leftarrow$ **Dense**(2, *activation* $\leftarrow$ *Softmax*)(*Output*)
10: *Model* $\leftarrow$ **Model**(*VGG19.Input, Output*)
11: *Model.Compile*(*Loss, Optimizer, Metrics*)
12: *Model.Fit*(*train_normalization, EPOCH* $\leftarrow$ 25, *validation_normalization, CALLBACKS*)
13: *Model.Evaluate*(*test_normalization*)

---

### Algorithm 4-4: Implementation ResNet50
1: **Preparation**(*train_features, test_features, validation_features*)
2:      **return** *train_normalization, test_normalization, validation_normalization*
3: *ResNet50* $\leftarrow$ **load**(*weights* $\leftarrow$ *ImageNet*)
4: **for** each *ResNet50.Layers* **do**
5:      *layersTrainable* $\leftarrow$ **FALSE**
6: *lastOutput* $\leftarrow$ *ResNet50.Layers*[-1].*output*
7: *Output* $\leftarrow$ **GlobalAveragePooling2D**()(*lastOutput*)
8: *Output* $\leftarrow$ **Dense**(256, *activation* $\leftarrow$ *ReLU*)(*Output*)
9: *Output* $\leftarrow$ **Dense**(2, *activation* $\leftarrow$ *Softmax*)(*Output*)
10: *Model* $\leftarrow$ **Model**(*ResNet50.Input, Output*)
11: *Model.Compile*(*Loss, Optimizer, Metrics*)
12: *Model.Fit*(*train_normalization, EPOCH* $\leftarrow$ 25, *validation_normalization, CALLBACKS*)
13: *Model.Evaluate*(*test_normalization*)

---

### Algorithm 4-5: Implementation CNN + CLAHE (Proposed Hybrid Pipeline)
1: **Input**: *Raw Mammogram Data*
2: **Apply CLAHE**(*Input, TileGridSize* $\leftarrow$ (8,8), *ClipLimit* $\leftarrow$ 2.0)
3: **Scale** images [0, 1]
4: **return** *X_enhanced*
5: *Model* $\leftarrow$ **Sequential**()
6: *Model.Add*(**Conv2D**(32, *kernel* $\leftarrow$ (3,3), *activation* $\leftarrow$ *ReLU*))
7: *Model.Add*(**MaxPooling2D**(*pool* $\leftarrow$ (2,2)))
8: *Model.Add*(**Conv2D**(64, *kernel* $\leftarrow$ (3,3), *activation* $\leftarrow$ *ReLU*))
9: *Model.Add*(**Flatten**())
10: *Model.Add*(**Dense**(128, *activation* $\leftarrow$ *ReLU*))
11: *Model.Add*(**Dense**(2, *activation* $\leftarrow$ *Softmax*))
12: *Model.Compile*(*Loss* $\leftarrow$ *CategoricalCrossEntropy*, *Optimizer* $\leftarrow$ *Adam*)
13: *Model.Fit*(*X_enhanced, epochs* $\leftarrow$ 25)
14: *Model.Evaluate*(*test_enhanced*)
