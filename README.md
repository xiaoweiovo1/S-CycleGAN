# CycleGAN_ssim

This project is an extension of the project [Image Editing using GAN](https://github.com/Tandon-A/Image-Editing-using-GAN). 

Implemented and trained Cycle Consistent Generative Adversarial Network (CycleGAN) as described in the [paper](https://arxiv.org/abs/1703.10593) with different [loss functions](https://arxiv.org/abs/1511.08861), specifically SSIM loss, L1 loss, L2 loss and their combinations, to produce images of better visual quality. 


<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/CycleGAN_working.png" width="600" alt="CycleGAN model">
<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/img_proj.png" width="400" alt="Project Working">

###### Fig 1: CycleGAN working

For the CycleGAN implementation with L1 Loss refer to [here](https://github.com/Tandon-A/Image-Editing-using-GAN/tree/master/CycleGAN). For the official CycleGAN implementation read [here](https://github.com/junyanz/CycleGAN). 

## Prerequisites

* Python 3.3+
* Tensorflow 1.6+
* pillow (PIL)
* (Optional) [Monet-Photo Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)

## Usage

To train the model:
```
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type l1
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type l2
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l1
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l2_a
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l2_b
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type l1_l2
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l1l2_a
> python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l1l2_b --batch_size 32
python train_cycleGAN_loss.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type vgg --batch_size 8
```


cd autodl-tmp/cg/result
python train_cycleGAN.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l1l2_a --batch_size 32 --epoch 100 
python train_cycleGAN.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type vgg --batch_size 32 --epoch 100 
python train.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type ssim_l1l2_a --batch_size 32 --epoch 100 
python train.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type vgg --batch_size 32 --epoch 100 


python train.py --data_path Dataset --input_fname_pattern .png --model_dir cycleGAN_model --loss_type vgg --batch_size 2 --epoch 2 


* data_path: Path to directory having trainA and trainB folders (Folders with these specific names (trainA, trainB) having domainA and domainB training images respectively)
* input_fname_pattern: Glob pattern of training images (file type of images such as .jpg or .png)
* model_dir: Directory name to save checkpoints
* loss_type: Loss type with which cycleGAN model is trained. (Available Options -- l1, l2, ssim, ssim_l1, ssim_l2_a, ssim_l2_b, l1_l2, ssim_l1l2_a, ssim_l1l2_b)


To test the model:
```
> python test_cycleGAN_loss.py --testA_image A01.jpg --testB_image B01.jpg --model_dir cycleGAN_model --loss_type ssim_l1

```


python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_modelc --loss_type ssim_l1l2_a
python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_modelcv --loss_type vgg
python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model --loss_type ssim_l1l2_a
python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_modelscv --loss_type vgg
zip -r ssim_l1l2_a.zip ssim_l1l2_a
zip -r vgg.zip vgg

> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model --loss_type ssim_l1l2_a
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model2 --loss_type l2
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model3 --loss_type ssim
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model4 --loss_type ssim_l1
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model5 --loss_type ssim_l2_a
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model6 --loss_type ssim_l2_b
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model7 --loss_type l1_l2
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model8 --loss_type ssim_l1l2_a
> python test_cycleGAN_loss.py --testA_image 0.png --testB_image 1.png --model_dir cycleGAN_model9 --loss_type ssim_l1l2_b
```
* testA_image: TestA Image Path
* testB_image: TestB Image Path 
* model_dir: Path to directory having checkpoint file
* loss_type: Loss type with which cycleGAN model is tested.


## Results 
Trained CycleGAN model on Monet-Photo Database.

### Comparison

<img src="https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compare.gif" alt="CycleGAN SSIM Compare">

###### Fig 2: Sample video showing comparison between CycleGAN with different SSIM loss settings


#### Photo to Monet Paintings

|Input Image |L1 Image |SSIM Image |SSIM + L1 |SSIM + L2(a) |SSIM + L2(b) |SSIM + L1 + L2(b)|
|:----------:|:-------:|:---------:|:--------:|:-----------:|:-----------:|:---------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB2.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/p2m/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/p2m/ex1.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/p2m/org/orgB5.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/p2m/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/p2m/ex2.png) |


#### Monet to Photo Paintings

|Input Image |L1 Image |SSIM Image |SSIM + L1 |SSIM + L2(a) |SSIM + L2(b) |SSIM + L1 + L2(b)|
|:----------:|:-------:|:---------:|:--------:|:-----------:|:-----------:|:---------------:|
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA2.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/m2p/ex1.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/m2p/ex1.png) |
![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/compar/m2p/org/orgA9.png)  | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/l1/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_a/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim_l2_b/m2p/ex2.png) | ![](https://raw.githubusercontent.com/Tandon-A/CycleGAN_ssim/master/assets/ssim%20_l1_l2_b/m2p/ex2.png) |



## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
