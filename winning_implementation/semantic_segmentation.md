# semantic segmentation

## loss fn

Because crossentropy only calculates pixelwise loss. Looking at the whole can cause some imbalances

- weighted loss function
  - reduces the imbalances by weighing right border more
  - reduces the importance of prevailing classes (e.g. in multiclass env the largest class does not become trivial)
- weight sceme for the borders of segmentations
- dice coefficent
  - checks the overlap of two samples
  - created for binary data

### Dice Loss

