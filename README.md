## Prerequisites

* Python 3.3+
* Tensorflow 1.6+
* pillow (PIL)
* (Optional) [Monet-Photo Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)

## Usage

To train the model:
```
> python train_S-cycleGAN.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type vgg --batch_size 8 --epoch 100 --num 4500
```

* data_path: Path to directory having trainA and trainB folders (Folders with these specific names (trainA, trainB) having domainA and domainB training images respectively)
* input_fname_pattern: Glob pattern of training images (file type of images such as .jpg or .png)
* model_dir: Directory name to save checkpoints
* loss_type: Loss type with which cycleGAN model is trained. (Available Options -- l1, l2, ssim, ssim_l1, ssim_l2_a, ssim_l2_b, l1_l2, ssim_l1l2_a, ssim_l1l2_b, vgg)
* batch_size: The number of samples selected for one training session.
* epoch: The epoch of S-cycleGAN.
* num: Total number of training images.


To test the model:
```
> python test_cycleGAN_loss.py --testA_image testA --testB_image testB --model_dir cycleGAN_model --loss_type ssim_l1 --num 10 --width 64 --length 256

```

```
* width: Test the width of the image
* length: Test the length of the image
* num: Total number of test images
* testA_image: TestA Image Path
* testB_image: TestB Image Path 
* model_dir: Path to directory having checkpoint file(Folders with these specific names (testA_image, testB_image) having domainA and domainB testing images respectively)
* loss_type: Loss type with which cycleGAN model is tested.
* num: Total number of images.
* width: Test the width of the image.
* length: Test the length of the image.

##Introduction

We propose a novel supervised learning-driven modelbased on cycle-consistence generative adversarial networks (CycleGAN), called S-CycleGAN. Significantly distinguished from the original unsupervised CycleGAN, S-CycleGAN can achieve supervised learning while preserving the attributes and loss function of the previous model. Moreover, a new loss combination strategy including the perceptual loss is developed to improve the segmentation performance. This strategy can highlight and segment GPR target by comparing the perceptual features of a segmented output against those of the ground truth in same established feature space. Therefore, the proposed method transfers our visual perception knowledge to the target instance segmentation task and is able to preserve key information.

## Results 
Trained CycleGAN model on Monet-Photo Database.

### Structural diagram

<img src="https://github.com/xiaoweiovo1/S-CycleGAN/Fig/FrameWork.png" alt="S-CycleGAN structure">

## Author

Boxuanqiao / (https://github.com/xiaoweiovo1/S-CycleGAN)
