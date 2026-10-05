# BlastoGrade AI

**Multi task deep learning for Gardner blastocyst grading, with heatmaps that show where each grade comes from**

<p align="center">
  <img src="docs/images/gradcam_gallery.png" alt="Grad CAM heatmaps for expansion, inner cell mass and trophectoderm" width="820">
</p>

## Project brief

On the fifth day after fertilisation an embryologist looks down a microscope at each blastocyst and writes a three part code such as 4AB. The number describes how far the embryo has expanded, from an early cavity (1) to a blastocyst that has hatched out of its shell (6). The first letter grades the inner cell mass, the tight cluster of cells that will become the fetus. The second letter grades the trophectoderm, the outer layer that will become the placenta. This is the Gardner system, and in most laboratories it decides which embryo is transferred first, which are frozen and which are discarded.

The system has a known weakness: it is a human judgement made in seconds. Two experienced embryologists looking at the same embryo disagree on at least one of the three components surprisingly often, and the same embryologist can disagree with their own grade from a week earlier. The consequences are concrete. A borderline trophectoderm graded B on Monday and C on Thursday can be the difference between an embryo being frozen and being discarded. Laboratories invest in training and double reading to control this, but consistency is expensive, and the grade that reaches the clinical record rarely carries any trace of how confident the grader was.

BlastoGrade AI is a complete computer vision product aimed at that consistency problem. A single convolutional network with a shared backbone and three output heads reads a blastocyst image and predicts expansion, inner cell mass grade and trophectoderm grade together, in the way the three judgements share anatomy. For every prediction it produces a separate Grad CAM heatmap per head, so a reviewer can see that the inner cell mass grade came from the cell cluster and not from a shadow on the dish. Low confidence cases are routed to a human review queue. The repository covers data generation, augmentation, masked multi task training, ordinal evaluation, a localisation audit of the heatmaps, an inference API and a Docker image, with tests and continuous integration.

Annotated blastocyst image sets are small and licence restricted, so the model here is trained on 6,400 procedurally rendered blastocysts. The renderer draws each embryo from its Gardner grades (cavity size, zona thickness, herniation, cell cluster size and compactness, trophectoderm cell count) under randomised illumination, blur and noise, and the training labels carry simulated observer disagreement. Rendered images are far cleaner than real micrographs, so the accuracies below show that the pipeline works and how it behaves under label noise. They are not an estimate of performance on clinical images.

## What the system does

<table>
  <tr><th align="left">Capability</th><th align="left">How it is delivered</th></tr>
  <tr><td>Multi task grading</td><td>ResNet18 backbone (EfficientNet B0 and deeper ResNets selectable) with heads for expansion (6 classes), inner cell mass and trophectoderm (3 classes each)</td></tr>
  <tr><td>Clinically correct masking</td><td>Inner cell mass and trophectoderm are not gradable before expansion 3, so those losses and metrics are masked for early blastocysts</td></tr>
  <tr><td>Explainability</td><td>Grad CAM implemented from hooks, one heatmap per head, attached to the deepest stage that still has useful spatial resolution</td></tr>
  <tr><td>Heatmap audit</td><td>Pointing game against the known location of the inner cell mass, with a computed chance rate</td></tr>
  <tr><td>Review queue</td><td>Cases whose least confident head falls under 60 percent are flagged for a second reader</td></tr>
  <tr><td>Serving</td><td>FastAPI service for single images and ranked cohorts, returning grades, probabilities and heatmap overlays; Dockerised</td></tr>
</table>

<p align="center"><img src="docs/images/dataset_gallery.png" alt="Rendered blastocysts across the Gardner scale" width="820"></p>

## Key findings

### Trained on noisy labels, the model agrees with the reference standard better than a single annotator does

The training labels disagree with the true rendered grade at realistic rates: about 6 percent of expansion labels, 8 percent of inner cell mass labels and 11 percent of trophectoderm labels are off by one grade. The network never sees the true grades. On 1,196 held out images it nevertheless matches the reference standard more often than the annotator whose labels it learned from.

<table>
  <tr><th align="left">Component</th><th>Single annotator agreement with reference</th><th>BlastoGrade AI agreement with reference</th><th>Quadratic weighted kappa (model)</th></tr>
  <tr><td>Expansion (1 to 6)</td><td align="center">93.9%</td><td align="center">99.7%</td><td align="center">0.997</td></tr>
  <tr><td>Inner cell mass (A to C)</td><td align="center">92.4%</td><td align="center">99.5%</td><td align="center">0.995</td></tr>
  <tr><td>Trophectoderm (A to C)</td><td align="center">89.4%</td><td align="center">96.1%</td><td align="center">0.968</td></tr>
  <tr><td><b>Complete Gardner grade correct</b></td><td align="center"><b>81.2%</b></td><td align="center"><b>96.2%</b></td><td align="center"></td></tr>
</table>

<p align="center"><img src="docs/images/agreement_with_reference.png" alt="Agreement with reference grades: annotator against model" width="560"></p>

The mechanism is the reason a consistency tool can work at all. Individual annotation errors are scattered in both directions, so across thousands of examples they largely cancel and the network converges on the consensus. The business reading is that a model trained on a laboratory's own imperfect historical grades can still become a steadier grader than any one of the people who produced them. The complete grade matters most: because three components each have to be right, a single annotator gets the whole code right only 81 percent of the time, and that is the figure the model moves to 96 percent.

It also explains a subtlety in evaluation. Scored against the noisy annotator labels, the same model appears to reach only 86.6 percent on trophectoderm. That number is capped by the noise in the labels, not by the model. Any real world evaluation of a grading model against a single reader's labels will understate it in the same way, which is why consensus panels are the appropriate reference.

### Trophectoderm is the hardest component, and every error is to a neighbouring grade

<p align="center"><img src="docs/images/confusion_matrices.png" alt="Confusion matrices for the three heads" width="900"></p>

Expansion and inner cell mass are close to solved on this data. Trophectoderm accounts for almost all remaining errors (38 of 983 gradable test embryos), concentrated between B and C. This mirrors the clinical literature, where trophectoderm is consistently the component with the lowest agreement between observers, and it has a physical explanation: the grade depends on counting small cells around a thin ring, the first detail lost to blur and low resolution. No trophectoderm prediction was more than one grade from the reference. For a laboratory this identifies where a second reader adds the most value.

### The heatmaps point at the right anatomy, and that can be measured

Explanations are usually judged by eye. Because the renderer records where it drew the inner cell mass, this project can score them. Across 400 test embryos, the peak of the inner cell mass head's Grad CAM map fell on the inner cell mass in 78.8 percent of cases. The chance rate for the same test, the share of the 8 by 8 feature map that would count as a hit, is 10.0 percent. The gallery at the top of this page shows the qualitative pattern: the inner cell mass head attends to the cell cluster, the trophectoderm head to the cavity lining, and the expansion head to the zona and to the herniating lobe of a hatching blastocyst.

The figure is not 100 percent, and the fourth row of the gallery shows why: on some images a head also responds to structures outside the embryo, such as an empty zona shell. Grad CAM is a coarse instrument. It is good evidence that a head is looking in the right region, and it should not be read as a segmentation.

### Confidence is informative enough to drive a review queue

Flagging every embryo whose least confident head is under 60 percent sends 18 percent of cases to review. Among flagged embryos the complete grade is correct 87.0 percent of the time; among the rest it is correct 98.2 percent of the time. The flagged fifth therefore contains about 61 percent of all the model's errors. A laboratory that double reads only the flagged cases would catch most mistakes with a fraction of the double reading workload.

### The transfer decision is more robust than the grade

Many laboratories use a simple cut: a blastocyst is good quality if it is expansion 3 or more with both letter grades B or better. On that binary decision the model is 98.5 percent accurate, with precision of 99.4 percent and recall of 97.9 percent. Errors between A and B do not change the decision, so the operational impact of the remaining trophectoderm confusion is smaller than the raw accuracy suggests.

<p align="center"><img src="docs/images/training_curves.png" alt="Training loss and validation agreement by task" width="760"></p>

## Architecture

```
blastocyst image (any size, grayscale or colour)
        |
centre crop, resize, normalise                training adds rotation, flips, small scale and exposure changes
        |
shared convolutional backbone (ResNet18 by default)
        |
        +==> expansion head (6 classes)
        +==> inner cell mass head (3 classes)       loss masked when expansion is 1 or 2
        +==> trophectoderm head (3 classes)         loss masked when expansion is 1 or 2
        |
Grad CAM per head  ==>  heatmap overlays
        |
Gardner grade, per head probabilities, good quality flag, review flag
        |
FastAPI  /grade  /grade/batch  /model  /health        packaged as a Docker image
```

## Repository layout

```
blastograde_ai
    configs/config.yaml               data, backbone, training and artifact settings
    models_store/blastograde.pt       trained checkpoint (half precision weights)
    docs/images                       figures in this document
    docs/real_data.md                 how to train on real annotated images
    reports/metrics.json              full evaluation output
    src/blastograde
        data/synth.py                 procedural blastocyst renderer and dataset writer
        data/dataset.py               dataset, preprocessing, augmentation, label masking
        models/network.py             shared backbone with three heads
        engine/train.py               masked multi task training loop
        engine/metrics.py             ordinal metrics and Gardner helpers
        engine/evaluate.py            test evaluation, heatmap audit, figures
        explain/gradcam.py            Grad CAM, overlays, peak location
        inference/grader.py           serving facade
        api/main.py                   FastAPI service
    tests                             20 tests covering rendering, masking, network, Grad CAM, serving and training
    Dockerfile, compose.yaml          CPU inference image
```

## Getting started

Python 3.10 or later is required.

```
make install
make all
make test
```

`make all` renders the dataset, trains for seven epochs, evaluates and rebuilds the figures. Training takes 20 to 40 minutes on a single CPU core and far less on a GPU, which is detected automatically. A trained checkpoint is included, so grading and the API work straight after installation.

Start the API and open the interactive documentation at `http://localhost:8000/docs`:

```
make api
```

Or build and run the container:

```
docker compose up
```

### Grading an image

```python
import base64
import httpx

with open("embryo.png", "rb") as handle:
    response = httpx.post("http://localhost:8000/grade", params={"heatmaps": "true"},
                          files={"file": ("embryo.png", handle, "image/png")})
result = response.json()
print(result["gardner_grade"], result["good_quality"], result["needs_review"])
print(result["trophectoderm"]["probabilities"])

with open("icm_heatmap.png", "wb") as handle:
    handle.write(base64.b64decode(result["heatmaps"]["icm"]))
```

The batch endpoint accepts up to 16 images for one patient and returns them ranked, best grade first, with unreadable files reported individually instead of failing the request.

### Training on real images

`docs/real_data.md` describes the folder layout and label file the pipeline expects, lists public sources with Gardner annotations and explains the settings to change (pretrained ImageNet weights, 224 pixel inputs). The evaluation step works on real data and skips the parts that need rendered ground truth.

## Limitations and responsible use

* The images are rendered, not photographed. Real micrographs have focal plane variation, debris, overlapping cells and laboratory specific optics, and accuracy on them will be lower.
* Label noise is simulated as independent adjacent grade errors. Real disagreement is partly systematic, and a model will learn a systematic bias.
* The network sees one focal plane at 128 pixels. Clinical systems typically use several focal planes at higher resolution.
* Grad CAM indicates the region a head relies on. It is not proof of correct reasoning.
* BlastoGrade AI is a research and portfolio prototype. It is not a medical device and must not be used to select embryos.

## Licence

Released under the MIT licence.
