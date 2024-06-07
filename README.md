# CNN-based tracker for coronary artery centerline extraction
Code for the paper _Coronary Artery Centerline Extraction in Cardiac CT Angiography Using a CNN-Based Orientation Classifier_, Wolterink et al. 2019, Medical Image Analysis  ([link to the paper](https://www.sciencedirect.com/science/article/pii/S1361841518308491), [arXiv](https://arxiv.org/abs/1810.03143)).

This method extracts coronary artery centerlines in CCTA using a convolutional neural network (CNN). A 3D dilated CNN is trained to predict the most likely direction and radius of an artery at any given point in a CCTA image based on a local image patch. Starting from a single seed point placed manually or automatically anywhere in a coronary artery, a tracker follows the vessel centerline in two directions using the predictions of the CNN. Tracking is terminated when no direction can be identified with high certainty.


![image](https://github.com/iolag/orientation-classifier/assets/63402398/edbc9d4d-f2f1-493d-bf79-900051a177a8)

# Usage
For training, SITK readable images and centerline points in txt form are required.
The centerline files should look like this:

```
x1 y1 z1 radius1
x2 y2 z2 radius2
...
xN yN zN radiusN
``` 
The folder structure assumed is the following:

```
args.datadir
│
├── images
│   └── image1.mhd
│
└── centerlines
    └── image1
        ├── centerline1.txt
        ├── centerline2.txt
        ├── centerline3.txt
        └── centerline4.txt
```

To change any of the above, modify the getData function accordingly. 

To train the tracker, you can simply run:
```
python train.py --datadir path/to/data/
```

For tracking, the algorithm requires 2 text files, one including the coordinates of the 2 ostia, and the other including 1 or more seed points for initialization. Structure is the same as the centerline files above. The paths to these files should be provided using the --ostia and --seeds flags respectively. 

Example command:
```
python track.py --seeds seeds.txt --ostia ostia.txt --impath path/to/image/image.mhd
```

# Reference
If you use this code, please cite the accompanying paper:

```
@article{WOLTERINK201946,
  title = {Coronary Artery Centerline Extraction in Cardiac {{CT}} Angiography Using a {{CNN-based}} Orientation Classifier},
  author = {Wolterink, Jelmer M. and {van Hamersvelt}, Robbert W. and Viergever, Max A. and Leiner, Tim and I{\v s}gum, Ivana},
  year = {2019},
  journal = {Medical Image Analysis},
  volume = {51},
  pages = {46--60},
  issn = {1361-8415},
  doi = {10.1016/j.media.2018.10.005}
}
```
#
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>
