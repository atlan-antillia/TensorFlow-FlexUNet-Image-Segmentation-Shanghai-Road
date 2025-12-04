<h2>TensorFlow-FlexUNet-Image-Segmentation-Aerial-Imagery-Shanghai-Road (2025/12/05)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Shanghai-Road</b> (Singleclass) based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels PNG
<a href="https://drive.google.com/file/d/1bmHTLIUPxYbaBIiQs38hGgAXO8r3CDv-/view?usp=sharing">
<b>Augmented-Shanghai-Road-ImageMask-Dataset.zip</b></a>
which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/harshilsanganiii/r-shanghai">
<b>shanghai satellite images for CNN model training</b>
</a> on the kaggle web site.
<br><br>

<hr>
<b>Actual Image Segmentation for the Shanghai-Road Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10325.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10325.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10325.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10422.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10422.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10422.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10808.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10808.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10808.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The datase used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/harshilsanganiii/r-shanghai">
<b>shanghai satellite images for CNN model training</b>
</a> on the kaggle web site.
<br>
<b>road network detection dataset from spacenet 3 challenge</b>
<br><br>
<b>Authors</b><br>
spacenet<br>
<br>
<b>Collaborators</b><br>
HARSHIL SANGANIii
<br><br>
<b>About Dataset</b><br>
Shanghai satellite images for CNN model training<br>
Road network detection dataset from the SpaceNet 3 Challenge, including Kaggle datasets with high-resolution satellite 
imagery and annotated road networks for deep learning-based segmentation and mapping tasks.
<br><br>
<b>Citation</b><br>
<pre>
@misc{spacenet_2024,
	title={shanghai satellite images  for CNN model training},
	url={https://www.kaggle.com/dsv/10276551},
	DOI={10.34740/KAGGLE/DSV/10276551},
	publisher={Kaggle},
	author={spacenet},
	year={2024}
}
</pre>
<br>
<b>License</b><br>
<a href="https://www.mit.edu/~amini/LICENSE.md">
MIT
</a>
<br>
<br>
<h3>
2 Shanghai-Road ImageMask Dataset
</h3>
<h4>2.1 Download Shanghai-Road dataset</h4>
 If you would like to train this Shanghai-Road Segmentation model by yourself,
 please download the augmented <a href="https://drive.google.com/file/d/1bmHTLIUPxYbaBIiQs38hGgAXO8r3CDv-/view?usp=sharing">
 <b>Augmented-Shanghai-Road-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Shanghai-Road
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Shanghai-Road Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Shanghai-Road/Shanghai-Road_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Shanghai-Road TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Shanghai-Road/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Shanghai-Road and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Shanghai-Road 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                     Road: yellow
rgb_map = {(0,0,0):0, (255,255,0):1,}
</pre>


<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = False
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 37,38,39)</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 39 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/train_console_output_at_epoch39.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Shanghai-Road/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Shanghai-Road/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Shanghai-Road</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Shanghai-Road.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/evaluate_console_output_at_epoch39.png" width="880" height="auto">
<br><br>Shanghai-Road
<a href="./projects/TensorFlowFlexUNet/Shanghai-Road/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Shanghai-Road/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1121
dice_coef_multiclass,0.9451
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Shanghai-Road</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Shanghai-Road.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Shanghai-Road/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the Shanghai-Road Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
augmented dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10310.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10310.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10310.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10422.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10422.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10422.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10467.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10467.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10467.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10554.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10554.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10554.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10777.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10777.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10777.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/images/10867.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test/masks/10867.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Shanghai-Road/mini_test_output/10867.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Satellite Imagery Aerial-Imagery Segmentation</b><br>
Nithish<br>
<a href="https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812">
https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
</a>
<br>
<br>
<b>2. Deep Learning-based Aerial-Imagery Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>3. A Comparative Study of Deep Learning Methods for Automated Aerial-Imagery Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>
<b>4. Machine Learning for Aerial Image Labeling</b><br>
Volodymyr Mnih<br>
<a href="https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf">
https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
</a>
<br>
<br>
<b>5.City-Scale Road Extraction from Satellite Imagery v2:
Road Speeds and Travel Times</b><br>
Adam Van Etten<br>
<a href="https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf">
https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf
</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road
</a>
<br>
<br>
<b>8. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Vegas-City-Road</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Vegas-City-Road">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Vegas-City-Road
</a>
