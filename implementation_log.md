# FloraDex Implementation Log

This log records key decisions, learning, implementation issues, fixes, experiments, and AI-assisted work throughout the FloraDex project.

## 19 August 2026 — Project setup

### What I did
Created the FloraDex GitHub repository and initial project structure.

### Why
I wanted a clear separation between the machine learning notebook, project documentation, dataset notes, and saved results.

### Tools
- GitHub
- VS Code
- Git

### AI use
Used ChatGPT to help plan the workspace structure and explain the roles of GitHub, VS Code, Google Colab and Google Drive.

### Verification
I manually created the repository and project structure, then checked that the local repository was connected to GitHub and could be committed and pushed successfully.

## 19 August 2026 — Problem definition

### Decision
Defined FloraDex as a supervised multiclass image classification problem where the input is a flower photograph and the output is one flower class from a predefined set.

### Reasoning
I chose a constrained classification problem rather than attempting to identify any possible flower species. This keeps the machine learning task feasible while still representing the core FloraDex concept.

### Current assumption
The model will only be able to predict flower classes represented in its training dataset. Images of unknown flower species may therefore still be incorrectly assigned to one of the known classes.

### AI use
Used ChatGPT to help translate the FloraDex concept into a formal machine learning task and clarify the distinction between the practical application goal and the model's training objective.

## 20 August 2026 — Dataset selection

### Decision
Selected the Oxford 17 Category Flower Dataset for the FloraDex proof-of-concept.

### Alternatives considered
I compared the Oxford 17 dataset with TensorFlow Flowers. TensorFlow Flowers would have been simpler because it has only five classes and more images overall, but it would provide a more limited representation of the FloraDex concept.

### Reasoning
Oxford 17 provides a better balance between challenge and feasibility. It contains 17 flower classes, which makes the classification task more meaningful for FloraDex, while still being manageable for A2.

The dataset also contains only 80 images per class, which creates a useful machine-learning problem because the model will need to learn from a relatively small labelled dataset. This provides a stronger justification for investigating transfer learning rather than training a large CNN entirely from scratch.

### Trade-offs identified
The smaller number of images per class may increase the risk of overfitting and make validation results more sensitive to the train/validation/test split.

This means I will need to pay particular attention to:
- validation design
- data augmentation
- generalisation to unseen images
- class-level errors and confusion

### Current scope
The model will only classify the 17 flower categories represented in the dataset, so FloraDex will be treated as a proof-of-concept rather than a complete flower-identification system.

### AI use
Used ChatGPT to compare suitable flower datasets, explain the trade-offs between TensorFlow Flowers and Oxford 17, and help assess which dataset better aligned with the A2 learning objectives and FloraDex concept.

### Verification
Dataset characteristics and class counts were checked against the Oxford Visual Geometry Group dataset information before finalising the choice.

## 21 August 2026 — Dataset exploration and validation split

### What I did
Loaded the Oxford 17 dataset into Colab, extracted the image files and verified that all 1,360 expected images were present.

I then explored the dataset structure and image characteristics before beginning any modelling.

### Dataset observations
The dataset contains 17 balanced flower classes with 80 images per class.

Visual inspection showed that:
- images are generally close-up photographs
- some flowers are centred while others are slightly off-centre
- some images contain multiple flowers
- backgrounds vary from simple to cluttered
- image clarity varies
- some classes, such as iris and fritillary, appear more visually diverse than others
- some different classes share similar colours or shapes, which may make classification more difficult

### Image properties
All 1,360 images were confirmed to use RGB colour mode.

The images are not uniform in size. I found 486 unique image dimensions, with widths ranging from 499 to 1057 pixels and heights ranging from 499 to 1093 pixels.

This means the images will need to be resized to a consistent input shape before training.

### Data organisation
Created a labelled dataframe linking each image to:
- filename
- image path
- class index
- class name

This confirmed that all 17 classes contain exactly 80 images.

### Validation strategy
Created a reproducible stratified 70/15/15 train, validation and test split using `random_state=42`.

The final distribution is:
- 952 training images — 56 per class
- 204 validation images — 12 per class
- 204 test images — 12 per class

Stratification was used to preserve the balanced class distribution across all three subsets.

### Data leakage check
Checked for duplicated filenames across the three subsets.

Results:
- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0

This confirms that each image appears in only one subset.

### Learning
This stage reinforced that model development should not begin before understanding the data and evaluation setup.

The dataset also highlighted an important generalisation challenge: only 56 original training images are available per flower class after splitting, which may increase the risk of overfitting when using a high-capacity neural network.

### Next steps
Before training, I still need to:
- choose the model architecture
- select the required image input size
- decide on preprocessing and augmentation
- build a baseline model

### AI use
Used ChatGPT to explain the dataset exploration process, help interpret image dimensions and colour modes, explain the purpose of stratified train/validation/test splitting, and review the split for potential data leakage.

### Verification
I verified the AI-assisted steps by running the code in Colab and checking the actual outputs, including:
- 1,360 total images
- 17 classes with 80 images each
- RGB mode for all images
- image-dimension ranges
- 56/12/12 images per class across train/validation/test
- zero overlap between subsets

## 31 August 2026 — Baseline CNN and transfer learning comparison

### What I did
Designed, implemented and trained two image-classification models on the Oxford 17 dataset:

1. A relatively simple CNN trained from scratch
2. A MobileNetV2 transfer-learning model using pretrained ImageNet weights

Both models were trained using the same train/validation split so that their performance could be compared fairly.

### Hypothesis-family design
The baseline hypothesis family was defined as a small CNN that learns all visual features directly from the Oxford 17 training data.

The main hypothesis family was defined as a pretrained CNN using MobileNetV2, where the pretrained feature-extraction layers remain frozen and only a new FloraDex classification head is trained.

This creates the main project investigation:

> Can transfer learning provide better flower-image classification than a model learning visual features from scratch, given FloraDex's limited training data?

### Model architecture decisions
The baseline CNN used:
- three convolutional layers
- increasing filter sizes: 32, 64 and 128
- max pooling after each convolution
- global average pooling
- a dense layer
- dropout
- a final 17-class softmax output layer

The baseline contained 111,953 trainable parameters.

MobileNetV2 was selected as the transfer-learning architecture because it is lightweight, computationally efficient, compatible with 224 × 224 RGB images, and well suited to transfer learning on a relatively small dataset.

The transfer-learning model contained:
- 2,279,761 total parameters
- 21,777 trainable parameters
- 2,257,984 frozen pretrained parameters

### Preprocessing and augmentation
All images were resized to 224 × 224 × 3.

Training data augmentation included:
- horizontal flipping
- small rotations
- small zoom changes
- small translations

Augmentation was applied only to the training data.

The baseline model used pixel rescaling to the 0–1 range, while MobileNetV2 used its model-specific `preprocess_input` function.

### Training configuration
Both models used:
- sparse categorical cross-entropy loss
- Adam optimiser
- learning rate of 0.001
- batch size of 32
- maximum of 30 epochs
- early stopping based on validation loss

### Baseline results
The baseline CNN learned gradually across 30 epochs.

Validation accuracy increased to approximately 60%, while validation loss decreased substantially.

Training and validation curves remained relatively close, with no strong evidence of severe overfitting.

This showed that the baseline was able to learn meaningful flower features from scratch, but its overall classification performance remained moderate.

### Transfer-learning results
The MobileNetV2 transfer-learning model learned much more rapidly.

Validation accuracy reached approximately 89–92%, substantially higher than the baseline.

Training accuracy approached 99%, while validation accuracy stabilised around 90%, creating a visible generalisation gap.

Validation loss flattened while training loss continued to decrease, indicating some overfitting. However, validation performance remained strong.

### Initial comparison
The first comparison strongly favours transfer learning.

The baseline CNN achieved approximately 60% validation accuracy, while MobileNetV2 achieved approximately 89–92%.

This suggests that pretrained visual features transfer effectively to the Oxford 17 flower-classification task and provide stronger generalisation than learning all visual features from scratch using the limited training data.

### Issue encountered
The MobileNetV2 training cell was accidentally run more than once while the internet connection was unstable.

Because rerunning `.fit()` continues training from the model's existing weights, the repeated run was not treated as the official result.

To correct this, I rebuilt MobileNetV2 from the original ImageNet weights, recreated the 17-class classification head, recompiled the model, created a fresh early-stopping callback, and completed one clean training run.

### Learning
This session helped connect several course concepts to the implementation:

- the model architecture defines the hypothesis family
- training searches for suitable parameter values within that family
- convolutional layers build increasingly complex visual features by stacking transformations
- cross-entropy measures prediction error
- Adam uses gradients to update trainable parameters
- validation performance provides evidence about generalisation
- pretrained representations can substantially reduce the amount of task-specific learning required

### Next steps
The next stage is to evaluate the selected MobileNetV2 model on the untouched test set using:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- misclassification analysis
- prediction confidence

### AI use
Used ChatGPT to:
- compare MobileNetV2 and EfficientNetB0
- refine the hypothesis-family design
- explain the CNN architecture and parameter counts
- design the TensorFlow data pipeline
- explain the role of cross-entropy, Adam, epochs and early stopping
- interpret training and validation curves
- identify the repeated-training issue and rebuild a clean transfer-learning run

### Verification
AI-assisted recommendations were verified by:
- inspecting TensorFlow model summaries
- checking trainable and non-trainable parameter counts
- running both models in Colab
- comparing actual training and validation curves
- confirming the clean MobileNetV2 run started from pretrained ImageNet weights
- keeping the test set untouched for later final evaluation

## 10 September 2026 – Final Evaluation and Error Analysis

### Work completed
- Restored the saved MobileNetV2 model without retraining.
- Recreated the original stratified train/validation/test split using the same `random_state=42`.
- Confirmed the held-out test set contained 204 images, with 12 images per class.
- Evaluated the final MobileNetV2 model on the held-out test set.
- Achieved 97.06% test accuracy and 0.1658 test loss, with 198 of 204 images classified correctly.
- Generated a classification report containing precision, recall and F1-score for all 17 classes.
- Generated a confusion matrix to inspect class-specific errors.
- Identified and visualised all six misclassified images.
- Compared model confidence between correct and incorrect predictions.

### Key findings
- MobileNetV2 generalised strongly to the held-out Oxford 17 test set.
- Macro precision, recall and F1-score were all approximately 0.97, suggesting consistently strong class-level performance.
- Crocus was the weakest class, with recall of 0.75 and three misclassified examples.
- Errors were concentrated in visually similar or ambiguous examples rather than being spread broadly across all classes.
- Correct predictions had a mean confidence of approximately 92.9%, compared with 74.8% for incorrect predictions.
- One incorrect crocus → iris prediction had 98.8% confidence, showing that softmax confidence should not be treated as certainty.

### Challenges / decisions
- The test accuracy of 97.06% was higher than the validation accuracy of approximately 89.22%, so the original split procedure was checked to ensure the same held-out test set had been reconstructed correctly.
- The original split used the same stratified `train_test_split` process and `random_state=42`, confirming consistency.
- No additional model tuning or fine-tuning was performed after examining the test set, because the test set was reserved for final evaluation.

### AI assistance and verification
AI assistance was used to help interpret evaluation metrics, structure the error analysis, and explain the relationship between model confidence and correctness. The generated explanations were checked against the actual classification report, confusion matrix, misclassified-image outputs and saved confidence values before being incorporated into the project.

### Work completed
- Completed the practical evaluation of the final MobileNetV2 classifier.
- Compared the optimisation objective, sparse categorical cross-entropy, with the broader practical goal of providing accurate and trustworthy flower identification.
- Identified cases where strong statistical performance may still lead to poor practical outcomes, particularly highly confident incorrect predictions.
- Documented the main limitations of the current FloraDex classifier, including limited dataset size, same-distribution evaluation, closed-set classification and uncertainty in softmax confidence.
- Proposed future improvements including more diverse real-world data, unknown-class handling, confidence calibration, alternative predictions and further transfer-learning experiments.
- Completed the overall project conclusion and answered the main investigation question.

### Key findings
- MobileNetV2 substantially outperformed the baseline CNN under the experimental setup, supporting transfer learning for this limited-data classification task.
- High test accuracy and macro F1-score indicate strong in-dataset generalisation, but do not guarantee equivalent performance on real smartphone photographs.
- The practical objective of FloraDex is broader than minimising cross-entropy loss because users also need predictions to be reliable and uncertainty to be communicated appropriately.
- The 98.8% confident crocus → iris error demonstrates that softmax confidence should not be interpreted as certainty.
- The current 17-class closed-set formulation remains a significant deployment limitation because unsupported flower species would still be forced into one of the known classes.

### Challenges / decisions
- The final analysis focused on interpreting existing evidence rather than adding additional models or tuning after test evaluation.
- Further model comparisons were treated as future work because the project question was already answered by the controlled baseline-versus-transfer-learning comparison.
- Limitations and future improvements were separated so that each identified weakness was linked to a practical next step.

### AI assistance and verification
AI assistance was used to help structure the practical evaluation, limitations, future-work discussion and conclusion. Suggestions were checked against the implemented model, actual evaluation outputs, the project scope and the assessment criteria before being incorporated into the final analysis.