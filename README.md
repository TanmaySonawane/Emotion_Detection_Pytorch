**Multimodal Emotion Detection from Speech**

This project implements a multimodal emotion recognition system that classifies human emotions from speech audio using deep learning. The system processes raw audio files, extracts Mel-based representations, and trains a PyTorch model to predict emotional states. The focus of this project is learning and experimenting with CNN-based audio representations, model training behavior, and practical challenges in emotion classification.

_Project Motivation_

Human speech contains rich emotional information beyond spoken words. Automatically detecting emotions from audio has applications in areas such as:

Human–computer interaction

Mental health monitoring

Adaptive gaming systems

Call-center analytics

This project explores how audio representations and convolutional neural networks can be used to model emotional patterns in speech, and how architectural and data-related decisions impact model performance.




_Dataset_

The project primarily uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset.

Clean, labeled emotional speech samples

Multiple speakers (actors)

Consistent recording conditions

Emotion labels encoded per audio file

To ensure valid evaluation, actor-wise train/test splits are used so that the model does not see the same speaker during training and testing.


Emotion Classes

Original emotion labels were consolidated into five emotion categories to reduce ambiguity and improve learning:

Final Class	Combined Emotions
Neutral	Neutral + Calm
Happy	Happy + Surprise
Sad	Sad + Fear
Angry	Angry
Disgust	Disgust

This consolidation reflects similarities in vocal intensity and pitch patterns across emotions.

Audio Preprocessing
Why Mel Spectrogram Arrays (.npy)?

Instead of saving spectrograms as images, audio files are converted into Mel spectrogram NumPy arrays:

Faster loading during training

Avoids image compression artifacts

Preserves numerical precision

Easier integration with PyTorch tensors

Preprocessing Steps

Load audio using librosa

Convert waveform to Mel spectrogram

Apply log scaling

Normalize values

Save as .npy arrays

This step is handled in a dedicated preprocessing script.

Model Architecture

The model is implemented in PyTorch and uses a pretrained ResNet-18 as the backbone.

Why ResNet-18?

Lightweight and fast to train

Proven performance on structured visual representations

Skip connections help stabilize training

Easy to adapt for spectrogram-like inputs

Multimodal Design

Audio features are extracted from Mel spectrogram arrays

Features are passed through CNN layers

Final fully connected layers output emotion probabilities

Training Strategy
Epochs

An epoch represents one full pass over the training dataset.

Early experiments used too few epochs, resulting in underfitting

Training was extended to multiple epochs to allow proper convergence

Model Selection

The model is trained multiple times, and the best-performing model is selected based on validation performance. This approach:

Reduces randomness from weight initialization

Prevents saving poorly converged models

Produces more stable results

Challenges and Solutions
Challenge	Solution
Very low initial accuracy (~10%)	Increased epochs and improved loss setup
Model predicting a single emotion	Combined similar emotion categories
Speaker leakage	Implemented actor-wise splits
Slow image-based training	Switched from PNG images to .npy arrays
Unstable training	Used pretrained ResNet-18 backbone
Results

Final model achieved approximately 72% accuracy

Precision, recall, and F1-score were also approximately 0.72

Significant improvement over early baseline models

These results demonstrate the effectiveness of:

Class consolidation

Proper data splitting

Improved preprocessing

Deeper CNN architectures

Future Work

Potential extensions of this project include:

Training CNN models using TensorFlow for comparison

Evaluating performance on the TESS dataset

Combining RAVDESS and TESS for larger-scale training

Applying emotion recognition to:

Emergency call analysis

Adaptive gaming systems (difficulty based on user emotion)

Human–computer interaction research
