


# **Using RBM to detect weed**

## **ABSTRACT**

This report presents a small project to classify the weed and crop in the field. we randomly select the 181 images from dataset Stuttgart which is a notable weed segmentation dataset and contain 2563 valid images. Our project following two steps, firstly, we use conventional image processing method to segment the vegetation patch from whole image and generate the correspond label from ground truth, then we apply Restricted Boltzmann Machine to extract the feature map and feed into the logistic regression classifier to recognize the weed.

## **Data preparation**

The experiments are conducted on sugar beet ﬁelds located near Stuttgart in Germany. dataset has been recorded with different variants of the BOSCH DeepField Robotics BoniRob platform. All robots use the 4-channel RGB+NIR camera JAI AD-130 GE mounted in nadir view, but with different lighting setups, the datasets differ from each other in terms of the weed and soil types. For Stuttgart dataset, the crop, weed pixels occupy around 1.5%, 0.7% of whole images. For efficiently detect the weed from solid, we preprocess the dataset as following approaches:

1. We randomly select 181 images in our experiment from totally 2563 valid segmentation images. Each image includes four band information: RGB+NIR and has its corresponding ground truth. The structure for each chunk of data are shown below.
```
data
----gt
  ----gt.<index>.png
----nir
  ----nir.<index>.png
----rgb
  ----rgb.<index>.png
```

1. Some vegetation indices are often used for vegetation segmentation, in terms of our experiments, we apply Excess Green (ExG) to split the vegetation from soil. Where the $I_G = \frac{g}{(r+g+b)}, I_R = \frac{r}{r+g+b}, I_B= \frac{b}{r+g+b}$, $r,g,b$is the channel band from rgb image. Figure 2
2. The segmented image includes a lot of noise and the boundaries is not clear enough, we exploit the morphology opening and closing operation to refine the boundary as well as remove the noise. Figure 3
  1. Morphology opening is the dilation of the erosion of a set A by a structuring element B. , where denote erosion and dilation, respectively. Which helps to remove small object from foreground of the image.
  2. Morphology closing, , it uses to remove the small holes.
  3. In our experiments, the structuring element for morphology opening and closing is 3x3 block and repeat twice for each operation.
3. The different between solid and vegetable is clear enough, and we able to use simple thresholding method to split them, if the pixel value is greater than 13, we assume to the vegetation pixels, otherwise is solid.
4. In this step, we remove the connected components objects small than 200 pixels for filter the unnecessary noise object in image. Figure 6
5. Then, we covert the segmentation mask to bounding box annotation and save to xml file. Figure 7.
6. We generate the patch image according xml files and find it ground truth label. The example of final crop image and label file shows in Figure 8.


Figure 3 ExG image

Figure 2 NIR image

 ![](RackMultipart20210330-4-1yxzy61_html_840d70827d009e21.gif) ![](RackMultipart20210330-4-1yxzy61_html_79fd828992ff3f3d.jpg) ![](RackMultipart20210330-4-1yxzy61_html_3717d424120d2a98.png) ![](RackMultipart20210330-4-1yxzy61_html_76c894d868d985bb.png)

Figure 5 Binary Threshold to separate the vegetable from background

Figure 6 Remove small object

F ![](RackMultipart20210330-4-1yxzy61_html_202605ccd706a99.png) ![](RackMultipart20210330-4-1yxzy61_html_86e48a73026fd785.jpg) igure 1 RGB image

Figure 4 Morphology opening closing

![](RackMultipart20210330-4-1yxzy61_html_c85d465e4958c1b4.jpg)

Figure 7 Covert the segmentation mask to bounding box annotation and convert to xml file

![](RackMultipart20210330-4-1yxzy61_html_cfc70ec149d7dac8.jpg)

Figure 8 left column shows the crop RGB, NIR Image, right column is the crop image description with ground truth label.

**FEATURE EXTRACTION**

In our experiments, we apply the restricted boltzmann machine model to learn the good latent representations from the dataset, which can perform effective non-linear feature extraction to learn more discrimination features.

**Restricted**  **Boltzmann Machine**

1. **The structure of RBM**

The RBM is a two layers neural network, which consist of hidden layer and visible layer. The hidden layer and visible layer are fully connected and have undirected, but the neuron in hidden layer and visible layer is independent, so we can quickly get the unbiased sample from the posterior distribution when give data. The figure 9 shows the structure of RBM, where v is the visible layer, and h is the hidden layer, W is the weight between hidden layer and visible layer.

![](RackMultipart20210330-4-1yxzy61_html_37bee70d1cfc4b58.jpg)

Figure 9 The structure of RBM

1. **The probability distribution of RBM**

Given state h and w, we use **energies to define probabilities** :

Where the visible layer v has bias a, and hidden layer has bias b, the W is the weight distribution.

**The probability of the join configuration** between hidden and visible layer depends on the energy of that join configuration compared with the energy of all other join configures.

And Z is the normalized term denotes as

**Conditional distribution** , n is the neurons number in hidden layer. We can derive the condition distribution as follow:

**Activation function:**

Visible layer to hidden layer

Hidden layer to visible layer:

1. **Training RBM**

We maximum the likelihood to learn the RBM, m is the neurons number in visible layer.

Gradient descend to optimized L

Finally, the derivative of bias a, b, w:

**CLASSIFIER**

After extracting the features, we use logistic regression model to classify the weed and crop. The logistic regression is based on the linear regression, we first make a definition of **linear regression**.

And t **he logistic regression function** defines as following:

Which use the sigmoid function to map the linear regression to the value [0, 1], and we set threshold 0.5, when , it belongs to A category, otherwise, belongs to B categories.

**Optimized the Logistic Regression Function:**

1. The definition of loss function

1. Maximum likelihood to solve loss function, we assume there are m independent samples, the join probability is:

Log likelihood:

1. Gradient descent to learn

Where the second equality follows from

Partial derivatives

**EXPERIMENT AND RESULT**

In our experiments, we obtain 1705 weed/crop patches with its corresponding labels, in training phase, we randomly select 1364 patches to train the RBM with following hyper-parameters and get perdition result in Figure 10.

**Hyper-parameters:**

**RBM**

Learning rate: 0.06

Iteration: 10

The number of hidden layers: 100

**Result:**

Our method perform 0.82% accuracy to classify the weed and crop, we compute the average precision(AP) from the prediction to alleviate the different data distribution of crop and weed, where the AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

where and  are the precision and recall at the nth threshold.

The Figure 10 illustrate the classification report for crop and weed, where the performance of detect crop is lower than the weed, we hypothesis this is caused by the number of crop object is far less than weed, and hard to learn enough discrimination features. Otherwise, optimize the hyper-parameters f or RBM can also help to improve the performance, which we will do in the future.

![](RackMultipart20210330-4-1yxzy61_html_970ddd3aa5042590.jpg)

Figure 10 The result of classify weed and crop
