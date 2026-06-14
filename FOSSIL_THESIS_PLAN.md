# Fossil Tile Recognition Thesis Plan

Created: 2026-05-13

Current situation: the project has about 108 Fujifilm `.RAF` photos in `../FuckingFossils`. The core goal is to recognize larger visible fossils in limestone floor tiles from one building, starting with two classes: corals and shells. The stretch goals are to add more fossil types, estimate the angle at which a fossil was cut, and show matching 3D models.

## Short Summaries

1. **First thesis MVP:** build a tool that detects/recognizes large visible fossils as `coral`, `shell`, or `unknown/background` in tile photos. This is realistic in 1.5 months if we use transfer learning and careful labels instead of training a model from scratch.

2. **First technical experiment:** convert the `.RAF` files into consistent JPEG/TIFF images, crop around obvious fossils, label those crops, and train a tiny transfer-learning classifier. This answers the urgent question: "Are the photos good enough?"

3. **Best likely recognition path:** use object detection or segmentation-assisted detection. Start with bounding boxes for large fossils; move to masks only if boxes are too crude for fossils blending into limestone.

4. **Small-data survival plan:** do not chase a giant neural network. Use pretrained models, strict train/validation splits, aggressive but sensible augmentation, and collect more photos only where the model is confused.

5. **Angle estimation is a research extension, not the first promise:** make it work first on a narrow fossil type where the shape has measurable geometry. It can be framed as an exploratory geometric estimation module rather than a guaranteed universal feature.

6. **3D models are a delightful final layer:** use simple curated 3D models as explanatory visualizations. Do not make 3D reconstruction from the floor photos a requirement unless everything else is already working.

7. **Keep the project fun on purpose:** turn data collection into fossil hunts, keep a "model got confused by this one" gallery, name experiments clearly, and make weekly visual demos. The thesis will feel much less impossible if it produces visible artifacts every few days.

8. **Brainstorm option: modern minimum-data classifier study:** instead of only presenting the fossil website as a product, frame the thesis around how modern transfer learning, pretrained vision models, simple augmentation, phone-camera data collection, and lightweight tooling change the amount of labelled data needed to make a useful domain-specific classifier. The fossil task would become the concrete case study.

## Brainstorm Option: How Much Data Does A Classifier Need Now?

### Core Idea

The original ML-course intuition is that a reliable classifier needs a very large dataset, often thousands of examples per class. A possible thesis direction is to investigate whether that expectation changes for a narrow real-world task in 2026, where students can use pretrained models, transfer learning, phone cameras, annotation tools, and AI-assisted coding workflows.

### Possible Research Question

How much labelled data is required to obtain a useful fossil classifier for limestone tile images when using modern transfer learning and data augmentation?

Alternative versions:

- How does classifier performance change as the number of labelled fossil examples increases?
- Can a small, carefully labelled image dataset support a usable domain-specific classifier when combined with pretrained computer vision models?
- What are the practical minimum data requirements for building a two-class fossil classifier under small-data constraints?

### What The Experiment Could Look Like

Use the fossil dataset as a case study. Keep a fixed test set aside, then train several classifier versions with increasing amounts of training data:

- very small set, for example 5 examples per class
- small set, for example 10 examples per class
- medium set, for example 20 examples per class
- all available labelled training examples

For each training size, compare performance with and without simple augmentation. Report accuracy, confusion matrix, per-class precision/recall if possible, and a small gallery of mistakes.

### Why This Could Be Thesis-Friendly

This turns the project into a clear empirical study rather than only a product demo. Even imperfect results become useful because the thesis can discuss data requirements, overfitting, class ambiguity, lighting, background noise, and the limits of transfer learning.

### Risks

The question must stay narrow. "How have classifier requirements changed in 2026?" is too broad for a bachelor thesis. A safer version is: "For this specific fossil-recognition task, how far can modern small-data methods get, and where do they fail?"

The thesis would also need a proper related-work section on transfer learning, small-data learning, data augmentation, and possibly few-shot learning. The fossil website could still be included as a prototype, but the main scientific contribution would be the controlled data-size experiment.

## Recommended First 10 Days

Day 1-2: organize the data, convert RAW images, make contact sheets, and decide which photos are usable.

Day 3-4: label a tiny starter set: 20-40 clear coral crops, 20-40 clear shell crops, plus background/noise crops.

Day 5: train a simple transfer-learning classifier on crops. Use this only as a feasibility test, not as the final product.

Day 6-7: annotate 30-50 full images with bounding boxes around large visible fossils.

Day 8-9: train a small pretrained detector and inspect false positives/false negatives visually.

Day 10: choose the thesis direction based on evidence:

- If crops classify well and detector finds fossils: continue toward the detection product.
- If crops classify well but detector struggles: use segmentation-assisted or manual crop proposal.
- If crops classify poorly: improve capture protocol, labels, and class definitions before adding complexity.

## Path 1: Data Audit And Capture Protocol

### Goal

Before modelling, answer: are the images sharp, exposed, consistent, and representative enough for the classes we want?

### Why It Matters

With only about 100 photos, bad splits and inconsistent data can lie to us. The model may learn lighting, tile location, or camera angle instead of fossil shape. A clean data audit is also very thesis-friendly because it gives you a serious methodology section.

### What To Do

Create a spreadsheet or CSV with one row per image:

- filename
- tile/location if known
- fossil classes visible
- approximate fossil count
- photo quality: sharp / slightly blurry / unusable
- lighting: even / shadow / glare
- distance: standing-height / close-up / mixed
- notes

Make contact sheets from converted images so you can scan the dataset quickly. Mark images that have big clear fossils versus mostly fossil fragments.

### Capture Rules For New Photos

Use the same camera settings and distance as much as possible. Take one standing-height context photo and one closer detail photo for each interesting fossil. Add a ruler, coin, or printed scale marker when possible. Try to avoid glare by changing your angle slightly. Take several photos of confusing cases because those will be the most valuable training examples later.

### Fun Version

Make a "fossil quest log": every new photo gets a tiny note like "great coral spiral", "shell but cursed by glare", or "probably background, model bait". This sounds silly, but memorable labels help when you review mistakes.

## Path 2: RAW Conversion And Preprocessing

### Goal

Turn `.RAF` files into consistent training images while keeping enough texture detail for fossils.

### Recommended Output

Keep two derived image sets:

- `processed/full_jpg/`: normal JPEGs for annotation and app display.
- `processed/full_tiff_or_png/`: higher-quality images for experiments if disk space allows.

Do not overwrite the RAW files.

### What To Test

Start with minimal preprocessing:

- RAW conversion with fixed white balance/exposure settings if possible.
- Resize long side to a consistent size for modelling, for example 1280 or 1600 px.
- Optional contrast enhancement only as an experiment, not as the default.

Then compare:

1. original-looking converted image
2. lightly contrast-enhanced image
3. grayscale or local-contrast version

For each version, ask: do fossils become more visible to humans? If not, it probably will not help the model either.

### Preprocessing To Avoid At First

Avoid heavy edge filters, harsh thresholding, or turning everything into black-and-white too early. Fossils in limestone may be defined by subtle texture and color differences, so heavy preprocessing can destroy the signal.

### Useful Tools

`rawpy` can read RAW files and postprocess them through LibRaw. `scikit-image` and OpenCV can handle contrast, thresholding, morphology, and blur checks. Keep preprocessing scripts deterministic so the thesis can describe exactly what happened.

## Path 3: Fast Neural Network Sanity Check

### Goal

Quickly answer whether a neural network can separate coral from shell at all.

### Scope

This is not yet full fossil detection. It is a crop-level classifier:

- input: a crop that already contains one visible fossil
- output: coral / shell / background or uncertain

### Model Choice

Use transfer learning. Good first choices:

- ResNet18 or MobileNet from PyTorch/torchvision
- EfficientNet small variant
- CLIP-style image embeddings plus a simple classifier, if you want a no-training or low-training comparison

Do not train a CNN from scratch. The dataset is too small.

### Dataset

Create crops manually from the best images:

- 20-40 coral crops
- 20-40 shell crops
- 20-40 background/noise crops

If there are not enough clear examples, duplicate viewpoints are less valuable than new fossils. Similar crops from the same original photo must stay in the same split, otherwise validation accuracy will be fake.

### Augmentation

Use augmentations that match the real world:

- rotation
- small random crop/zoom
- brightness and contrast jitter
- mild blur
- horizontal/vertical flips if fossil orientation does not define the class

Avoid wild color changes or distortions that make fossils biologically/geometrically impossible.

### Success Criteria

This first model is promising if:

- validation accuracy is clearly above chance
- mistakes make sense to a human
- the model is not just learning image brightness or crop size

If it fails, that is still useful: it means the thesis should focus on data collection, segmentation, or narrower class definitions before promising full automation.

### Fun Version

Make a "tiny fossil exam" notebook: show 20 random crops, the model's guess, confidence, and the true label. The first time the model confidently calls a shell a coral, put it in a blooper gallery and use that to decide what data to collect next.

## Path 4: Object Detection For The Main Product

### Goal

Detect large visible fossils in full tile photos and classify each detection.

### Why This Is The Most Product-Like Path

The user experience should be: point camera at the floor, see boxes or outlines around fossils, get labels. Object detection directly matches that.

### Model Choice

Use a small pretrained YOLO-style detector. It is practical because it supports custom datasets, pretrained weights, data augmentation, and export options. Start with bounding boxes because they are faster to label than masks.

### Labels

Annotate only fossils visible from standing height. Ignore tiny fragments. Label:

- `coral`
- `shell`
- later: `sea_lily`, `other_shell`, `unknown_large_fossil`

Use a clear rule: if a fossil is too small or ambiguous for a human from standing height, it is background/noise for this thesis.

### Dataset Split

Split by photo group or tile region, not randomly by annotation. If the same fossil appears in several photos, keep all of those photos in the same split.

Suggested split:

- 70% train
- 15% validation
- 15% test

If there are too few examples, use cross-validation or repeated train/validation splits, but report the uncertainty honestly.

### Metrics

Use standard detection metrics, but also include human-friendly numbers:

- precision: how many detected fossils were real?
- recall: how many visible fossils did the model find?
- confusion matrix between coral and shell
- false positive gallery
- false negative gallery

For a thesis demo, visual error galleries are gold.

### Success Criteria

A realistic successful version:

- finds most large fossils in favorable lighting
- handles the first two classes better than chance
- refuses or marks uncertain cases instead of pretending every fossil is obvious

### Risk

The biggest risk is not architecture; it is labels and data diversity. If all coral examples come from one area and all shell examples from another, the model may learn the tile background. The fix is more mixed data and careful splitting.

## Path 5: Segmentation-Assisted Recognition

### Goal

Use segmentation to isolate fossil shapes before classification.

### When To Use It

Try this if bounding boxes are too rough because fossils blend into the limestone, or if cut-angle estimation needs precise outlines.

### Practical Workflow

1. Use a tool like CVAT for annotation.
2. Use SAM/SAM 2 or another interactive segmentation model to help create masks.
3. Clean masks manually for a small high-quality subset.
4. Train either:
   - an instance segmentation model, or
   - a detector plus a crop classifier using the segmented region.

### Why This May Help

Masks can remove background limestone and let the classifier focus on fossil texture/shape. They also provide geometry for angle estimation, such as major axis, circularity, rib direction, and contour shape.

### Why This May Be Too Much

Mask annotation takes longer than boxes. For 1.5 months, segmentation should be a targeted experiment, not the default for every photo.

### Best Compromise

Label boxes for the full dataset and masks for only the best 20-40 examples. Use masks to support the thesis argument and the angle prototype.

## Path 6: Cut-Angle Estimation Prototype

### Goal

Estimate how the fossil was cut by the stone surface, but only for fossil types where the visible cross-section has interpretable geometry.

### Realistic Framing

Do not promise "the system estimates the cut angle for all fossils." Instead:

Research direction: given a detected fossil and a simplified geometric model, can image-derived shape descriptors provide a plausible estimate or classification of cut orientation?

### Possible Methods

For shells:

- Extract outline or rib pattern.
- Measure ellipse axes, curvature, rib spacing, and symmetry.
- Map shape descriptors to rough orientation categories: cross-section / oblique / longitudinal.

For corals:

- Detect circular or radial structures.
- Estimate whether the cut is perpendicular, oblique, or tangential based on circularity and visible internal pattern.

For sea lilies later:

- Look for star-like or circular stem segments.
- Use geometry if the fossil type has a known cross-section.

### Math Direction

A reasonable first model is not magical; it is a simplified projection model:

- Assume the fossil has an idealized 3D shape.
- A planar cut through it creates a 2D section.
- Different cut angles produce different ellipse ratios, contour shapes, or visible pattern spacing.
- Estimate the angle category by comparing the observed 2D features to simulated or hand-drawn templates.

### Output

Use angle categories before exact degrees:

- likely cross-section
- likely oblique cut
- likely longitudinal cut
- unknown

Exact degrees can be a stretch goal.

### Validation

Ask a geology/paleontology source, professor, or reference images to judge a small set. If expert labels are impossible, present this module as exploratory and evaluate consistency rather than absolute truth.

### Fun Version

Make a "slice simulator": a tiny visual tool where you rotate an idealized shell/coral form and see the expected 2D cut shape. Even if the math is approximate, it makes the thesis more understandable and much more fun to demo.

## Path 7: 3D Model Display

### Goal

When the product recognizes a fossil, show a matching 3D model or simplified reconstruction.

### Realistic Version

Use curated or manually made generic models:

- coral-like model
- shell-like model
- crinoid/sea lily-like model later

The app does not need to reconstruct the exact fossil from the photo. It only needs to show an educational matching model.

### Implementation Options

- Web app with Three.js for interactive 3D display.
- Static 3D viewer in a desktop app.
- Pre-rendered rotating GIF/video if time gets tight.

### Thesis Value

This makes the project feel like a product, not only a notebook. It can also help explain the cut-angle idea: the app can show "this photo may correspond to this kind of slice through the fossil."

## Path 8: Thesis Research Questions

The research question is not set yet, which is actually good. Let the first experiments choose the wording.

### Strong Realistic Options

1. How effective is transfer learning for recognizing visible fossil types in limestone floor tiles under low-data conditions?

2. Does segmentation-assisted preprocessing improve fossil classification compared with direct classification/detection on full tile images?

3. Can a practical computer vision pipeline detect and classify large visible fossils in architectural limestone using a small custom dataset?

4. Can geometric descriptors extracted from segmented fossil cross-sections support approximate cut-orientation classification?

### Recommended Thesis Shape

Main research question:

Can a low-data computer vision pipeline identify large visible fossil types in limestone floor tiles from a specific building?

Subquestions:

- What preprocessing and annotation strategy gives the most reliable results?
- How well do pretrained models adapt to this narrow fossil-recognition task?
- Can segmentation-derived geometry provide an initial estimate of fossil cut orientation?
- How can recognition results be presented in an educational interactive product?

## Path 9: Six-Week Battle Plan

### Week 1: Data Triage And Baseline

- Convert RAW images.
- Make contact sheets.
- Create the metadata CSV.
- Crop clear examples.
- Train the first crop classifier.
- Decide whether the photos are usable.

### Week 2: Annotation And First Detector

- Annotate bounding boxes for 30-50 images.
- Train first object detector.
- Create visual prediction gallery.
- Write down failure categories.

### Week 3: Improve Data And Labels

- Collect targeted new photos based on model failures.
- Add background/noise examples.
- Fix inconsistent labels.
- Retrain and compare results.

### Week 4: Segmentation Or Geometry Experiment

- Use SAM/CVAT to mask a small high-quality subset.
- Extract simple geometric features.
- Test whether masks improve classification or support cut-angle categories.

### Week 5: Product Prototype

- Build a minimal app or notebook demo:
  - upload/select photo
  - show fossil detections
  - show class labels and confidence
  - show related 3D model or educational panel

### Week 6: Evaluation And Thesis Writing

- Freeze the test set.
- Run final evaluation.
- Make confusion matrices and error galleries.
- Document limitations honestly.
- Polish demo and final presentation.

## Decision Tree

If the crop classifier works, continue to object detection.

If crop classification works but object detection does not, use segmentation or manual region proposals.

If crop classification fails, simplify the labels, improve photos, or reframe the project around dataset construction and feasibility.

If detection works for two classes, add a third class only if there are enough examples.

If detection works and segmentation masks are decent, try cut-angle categories.

If time is low, keep angle estimation and 3D as demonstrable prototypes rather than fully evaluated modules.

## What To Build First In Code

Recommended project structure:

```text
FuckingRocks/
  data/
    metadata.csv
    processed/
      full_jpg/
      crops/
      labels_yolo/
  notebooks/
    01_data_audit.ipynb
    02_crop_classifier.ipynb
    03_detector_experiments.ipynb
    04_angle_geometry.ipynb
  src/
    convert_raw.py
    make_contact_sheet.py
    train_crop_classifier.py
    train_detector.py
    predict.py
  reports/
    figures/
    error_galleries/
```

First scripts:

1. `convert_raw.py`: convert `.RAF` to JPEG/TIFF.
2. `make_contact_sheet.py`: create overview sheets for fast human review.
3. `crop_dataset.py`: save manually selected or annotation-derived fossil crops.
4. `train_crop_classifier.py`: run the first neural sanity check.

## Preprocessing Experiments To Compare

Keep the comparison simple:

1. Raw conversion only.
2. Raw conversion plus mild contrast normalization.
3. Raw conversion plus CLAHE/local contrast.
4. Grayscale version.

Evaluate each on the same validation split. Keep the default as the simplest version that works.

## Important Thesis Honesty

A strong thesis does not need to solve every fossil forever. It needs a clear problem, a method, evidence, and honest limits.

Good limitation statements:

- The model is specialized to one building and one limestone material.
- It focuses on large fossils visible from standing height.
- Small fossil fragments are treated as background/noise.
- Cut-angle estimation is exploratory and fossil-type dependent.
- Results may not generalize to other stone types without new data.

These are not failures. They define the scope.

## Reference Links

- Ultralytics object detection task and custom dataset documentation: https://docs.ultralytics.com/tasks/detect/ and https://docs.ultralytics.com/datasets/detect/
- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- CVAT annotation tool overview: https://docs.cvat.ai/docs/getting_started/overview/
- Segment Anything / SAM repository: https://github.com/facebookresearch/segment-anything
- SAM 2 repository: https://github.com/facebookresearch/sam2
- `rawpy` RAW processing package: https://pypi.org/project/rawpy/
- LibRaw supported cameras: https://www.libraw.org/supported-cameras
- scikit-image exposure/CLAHE documentation: https://scikit-image.org/docs/stable/api/skimage.exposure.html
