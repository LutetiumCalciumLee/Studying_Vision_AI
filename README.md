# Computer Vision
***
## Chapter 1. Introduction to Computer Vision
### Vision Fundamentals
- **Computer Vision vs. Image Processing**: Image Processing takes an image as input and outputs a modified image, whereas Computer Vision takes an image and outputs knowledge or interpretation of the scene.
- **Biological Inspiration**: The field is heavily inspired by the human visual system, as shown by the Nobel Prize-winning research of Hubel and Wiesel on the visual cortex.
- **Core Goal**: The primary objective is to bridge the semantic gap between the raw pixel data a computer sees and the meaningful interpretation a human perceives.
- **Image Representation**: Images are treated as functions, represented digitally as a matrix of pixel values. Grayscale images have one value (intensity) per pixel, while color images typically have three (e.g., RGB).

### Applications
- **Wide-Ranging Impact**: Computer vision is a critical technology with applications in surveillance, medical imaging, special effects (motion capture), 3D urban modeling (Google Street View, Microsoft Photosynth), and more.
- **Consumer Technology**: It is integrated into everyday devices for features like face and smile detection in digital cameras, biometrics (iris and fingerprint scanning), and optical character recognition (OCR).
- **Modern Systems**: Advanced applications include mobile visual search (Google Goggles), automotive safety systems (lane detection, pedestrian warning), and vision-based interaction for gaming (Microsoft Kinect, Sony EyeToy).
***
## Chapter 2. Color Image Processing
### Image Representation and Processing
- **Bitmap vs. Vector**: Images can be represented as bitmaps (grids of pixels), which are good for complex scenes but have fixed resolution, or as vectors (geometric objects), which are efficient and scalable but limited to simpler graphics.
- **Image Processing Pipeline**: A typical workflow involves image acquisition, preprocessing (enhancement, restoration), and high-level analysis for recognition and interpretation.
- **Color's Role**: Color is a powerful descriptor for object identification. While humans can distinguish thousands of colors, many traditional algorithms work on grayscale images. However, modern deep learning methods actively leverage color information.

### Color Models
- **Light and Color**: Color is the perception of light reflected from an object within the visible electromagnetic spectrum (400–700 nm).
- **RGB Model**: An additive color model where Red, Green, and Blue are combined to produce other colors. It is standard for displays and digital images, often using 24 bits per pixel (8 bits for each channel).
- **CMY(K) Model**: A subtractive model used for printing, where Cyan, Magenta, and Yellow pigments are used. Black (K) is added for practical printing purposes.
- **HSI Model**: Represents color using Hue (the pure color), Saturation (the purity/richness of the color), and Intensity (brightness). This model is often more intuitive for human perception and certain processing tasks.
***
## Chapter 3. Edge and Line Detection
### Edge Detection
- **Purpose**: To identify significant and abrupt changes (discontinuities) in image intensity, which correspond to the boundaries of objects. Edges contain most of the semantic information in an image.
- **Process**: The fundamental method involves finding the peaks in the first derivative of the image's intensity function. To combat noise, the image is typically smoothed (e.g., with a Gaussian filter) before differentiation.
- **Canny Edge Detector**: A widely used, multi-stage algorithm for optimal edge detection.
    - 1. **Noise Reduction**: Smooth the image with a Gaussian filter.
    - 2. **Gradient Calculation**: Find the intensity gradient's magnitude and direction.
    - 3. **Non-Maximum Suppression**: Thin edges by keeping only the sharpest local points in the gradient direction.
    - 4. **Hysteresis Thresholding**: Use two thresholds (high and low) to track edges, starting strong edges and continuing them through weaker sections to reduce noise.

### Line Detection
- **Hough Transform**: A feature extraction technique used to find instances of objects with a certain shape (e.g., lines, circles) by a voting procedure in a parameter space.
- **How it Works**: Each edge point in the image "votes" for all possible lines that could pass through it. The lines that receive the most votes in the accumulator array are identified as the most prominent ones in the image.
- **Application**: It is robust to noise and gaps in lines, making it useful for tasks like identifying roads in aerial images or detecting circular objects like traffic signs or pupils.
***
## Chapter 4. Face Detection and Recognition
### Face Detection
- **Goal**: To determine whether human faces exist in an image and, if so, to identify their location and size.
- **Approaches**: Methods range from knowledge-based rules and skin color segmentation to template matching and modern feature-based learning approaches.
- **Viola-Jones Algorithm**: A landmark real-time face detector that uses three key ideas :
    - **Haar-like Features**: Simple rectangular features that are efficient to calculate.
    - **Integral Image**: A representation that allows for rapid feature calculation at any scale.
    - **AdaBoost and Cascade Classifier**: A machine learning method trains and combines many simple classifiers into a "cascade" that quickly rejects non-face regions, focusing computation on promising areas.

### Face Recognition
- **Challenges**: Recognition is difficult due to variations in pose, lighting, facial expression, occlusion (e.g., sunglasses), and aging.
- **Principal Component Analysis (PCA)**: A classic approach, also known as "Eigenfaces." It reduces the high dimensionality of image data by identifying the principal components (the directions of greatest variance) of a training set of faces. Each face can then be represented as a weighted combination of these "eigenfaces," allowing for efficient comparison.
- **State-of-the-Art**: Modern systems like those evaluated in the Face Recognition Vendor Test (FRVT) use more advanced techniques (e.g., LBP, LDA) and deep learning to achieve very high accuracy, even surpassing human performance in some controlled conditions.
***
## Chapter 5. Advanced 3D and Aging Models
### 3D Morphable Models
- **Concept**: A powerful method that uses a database of 3D face scans to create a flexible 3D face model. Any new face can be represented as a linear combination of the shapes and textures from this database.
- **Application**: By fitting this 3D model to a 2D input image, the system can estimate the face's 3D shape and pose. This is extremely useful for correcting pose variations, a major challenge in face recognition.

### Aging Modeling for Face Recognition
- **Problem**: A person's facial appearance changes significantly over time, making recognition across large age gaps difficult.
- **Solution**: Researchers develop models to simulate the aging process. By using a database containing multiple images of individuals at different ages, the system can learn the patterns of facial changes and apply them to either "age" or "de-age" a face image, improving recognition accuracy across a person's lifespan.
