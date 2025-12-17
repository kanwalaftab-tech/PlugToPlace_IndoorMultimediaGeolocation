(PLACEHOLDER - CHANGE/UPLOAD DUE TO SIZE LIMITS?  - Graham A.)
# Dataset: Plug and Socket Detection (Roboflow)

The dataset used in this work consists of annotated images of electrical plugs and sockets
for object detection using YOLO-based models.

## Source
- **Roboflow Universe Project:**  
  https://universe.roboflow.com/objsocket/my-first-project-s4zpp
- Dataset Version: v16
- Export Date: September 29, 2025
- License: CC BY 4.0

## Dataset Details
- Total images: 4,754
- Annotation format: YOLOv11
- Image size: 640 × 640 (after preprocessing)

### Preprocessing
- Auto-orientation (EXIF stripped)
- Resize to 640×640 (stretch)

### Augmentation
- Random crop (0–20%)
- Random rotation (−15° to +15°)
- Random brightness adjustment (−19% to +19%)

## Directory Structure
When exported from Roboflow, the dataset has the following structure

train/
images/
labels/
test/
images/
labels/
data.yaml

## Usage
The dataset is referenced by the training and evaluation scripts in this repository
via the `data.yaml` file.
