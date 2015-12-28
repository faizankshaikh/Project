### Modules

* #### Detection
    * ##### Preprocessing
    * ##### Segmentation
* #### Recognition
    * ##### Character
    * ##### Word

### Description of modules

* **Detection** aka Localization : This contains breaking down the image into image patches, pushing it through a text / no-text classifier and spewing out word bounding box.
* **Recognition** aka Classification : This contains recognizing individual words from the bounding boxes and give a letter stream. This is transferred to a spell corrector which finds out the intended word.

### Algorithms in use and their importance

* Image normalization by MuSigma : (1.1)
    * Before putting images through pipeline, normalization reduces work by decreasing the range of variance of image, and thus updating feature maps is easier.
* Grayscale conversion : (1.1)
    * For character recognition, color does not matter. So instead of confusing the classifiers with color, put them out of misery.
* Patch extraction : (1.2)
    * Divide image into patches of different sizes, so that when detector CNN does its work, it has more options.
* Detector CNN : (1.2)
    * Take the patches and tell if contain character or not
* ??  (1.2)
    * Create a word BBox
* ?? (2.1)
    * Divide word BBox into character BBox
* Recognizer CNN (2.1)
    * Do a character sensitive search and give a letter stream
* Spell Checker (2.2)
    * Take in the letter streams and give RESULT!
