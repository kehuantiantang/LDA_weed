


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

**APPENDIX**

1. **Code for**  **Data preparation**

| # coding=utf-8 **from**** pathlib ****import** Path **import**** os ****from**** skimage ****import** morphology **from**** lxml ****import** etree **import**** cv2 ****from**** tqdm ****import** tqdm **import**** numpy ****as**** np ****from**** skimage.measure ****import** label, regionprops **import**** matplotlib.pyplot ****as**** plt**

**def**** normalize**(x):&#39;&#39;&#39;normalize the image from 0~255 to 0 ~ 1:param x::return:&#39;&#39;&#39;**return **x /** 255.0**

**def**** ndvi**(rgb, nir):r, g, b = rgb[:, :,**0**], rgb[:, :,**1**], rgb[:, :,**2**]r, g, b = r / (r + b + g), g / (r + b + g), b / (r + b + g)
exg = np.clip( **2** \* g - r - b, **0** , **1** ) \* **255.0**** return** exg

**def**** read\_image\_path**(rgb\_path, nir\_path):&#39;&#39;&#39;read rgb, nir image path:param rgb\_path::param nir\_path::return:&#39;&#39;&#39;rgb\_paths = set([os.path.split(str(p))[-**1**]**for **p** in**sorted(Path(rgb\_path).glob(&#39;\*.png&#39;))])nir\_paths = set([os.path.split(str(p))[-**1**]**for **p** in**sorted(Path(nir\_path).glob(&#39;\*.png&#39;))])intersection = sorted(rgb\_paths.intersection(nir\_paths))
paths = [(os.path.join(rgb\_path, i), os.path.join(nir\_path, i)) **for** i **in** intersection] **return** paths

**def**** write2ndvi**(paths):&#39;&#39;&#39;read the channel rgb, nir and convert to ndvi image:param paths::return:&#39;&#39;&#39;
ndvi\_path = &#39;./data/ndvi&#39;os.makedirs(ndvi\_path, exist\_ok=True) **for** rgb, nir **in** tqdm(paths):rgb\_image = normalize(cv2.imread(rgb)[..., :: - **1**])nir\_image = normalize(cv2.imread(nir, cv2.IMREAD\_GRAYSCALE))
ndvi\_image = ndvi(rgb\_image, nir\_image)name = os.path.split(rgb)[- **1**]
# using morphology transformation to remove noisekernel = cv2.getStructuringElement(cv2.MORPH\_RECT, ( **3** , **3** ))ndvi\_image = cv2.morphologyEx(ndvi\_image, cv2.MORPH\_OPEN, kernel,iterations= **2** )ndvi\_image = cv2.morphologyEx(ndvi\_image, cv2.MORPH\_CLOSE, kernel,iterations= **2** )
ndvi\_image = ndvi\_image.astype(np.uint8)
# 0 black, 255 white\_, thre = cv2.threshold(ndvi\_image, **13** , **255** , cv2.THRESH\_BINARY)thre = thre \&gt; thre.mean()
# remove small objectthre = morphology.remove\_small\_objects(thre, min\_size= **200** ,connectivity= **1** )
# thre = cv2.subtract(255, thre)# thre = cv2.adaptiveThreshold(ndvi\_image, 255,# cv2.ADAPTIVE\_THRESH\_MEAN\_C,# cv2.THRESH\_BINARY, 11, 2)
cv2.imwrite(os.path.join(ndvi\_path, name), thre \* **255** )

**def**** segment2boundingbox**(mask\_path, gt\_path):&#39;&#39;&#39;read the ground truth file and locate the object which it correspondinglabel:param mask\_path::param gt\_path::return:&#39;&#39;&#39;
filename = os.path.split(mask\_path)[- **1**]xml\_path = &#39;./data/xml&#39;os.makedirs(xml\_path, exist\_ok=True)
annotation = etree.Element(&#39;annotation&#39;)etree.SubElement(annotation, &#39;filename&#39;).text = str(filename)
mask = cv2.imread(mask\_path, cv2.IMREAD\_GRAYSCALE) / **255** mask = cv2.dilate(mask, np.ones(( **3** , **3** ), np.uint8), iterations= **2** )
size = etree.SubElement(annotation, &#39;size&#39;)etree.SubElement(size, &#39;width&#39;).text = str(mask.shape[**0**])etree.SubElement(size, &#39;height&#39;).text = str(mask.shape[**1**])
# annotation the different object with different colormask\_label = label(mask)
# bounding boxprops = regionprops(mask\_label)
# ground truth imagegt\_image = cv2.cvtColor(cv2.imread(gt\_path), cv2.COLOR\_BGR2GRAY)
label\_name = {&#39;crop&#39;: &#39;0&#39;, &#39;weed&#39;: &#39;1&#39;}
# different color correspond to different label# print(gt\_image, np.unique(gt\_image))
**for** prop **in** props:# print(&#39;Found bbox&#39;, prop.bbox)xmin, ymin, xmax, ymax = prop.bbox
# corresponding to different labelgt\_crop = gt\_image[xmin:xmax, ymin:ymax]# print(np.unique(gt\_crop))weed\_num = gt\_crop.reshape(- **1** , ).tolist().count( **76** )crop\_num = gt\_crop.reshape(- **1** , ).tolist().count( **150** )gt\_label = &#39;weed&#39; **if** weed\_num \&gt; crop\_num **else**&#39;crop&#39;
# write to xml objectobject = etree.SubElement(annotation, &#39;object&#39;)etree.SubElement(object, &#39;name&#39;).text = gt\_labeletree.SubElement(object, &#39;label&#39;).text = label\_name[gt\_label]bndbox = etree.SubElement(object, &#39;bndbox&#39;)etree.SubElement(bndbox, &#39;xmin&#39;).text = str(xmin)etree.SubElement(bndbox, &#39;xmax&#39;).text = str(xmax)etree.SubElement(bndbox, &#39;ymin&#39;).text = str(ymin)etree.SubElement(bndbox, &#39;ymax&#39;).text = str(ymax)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5)) # ax1.imshow(mask\_label) # ax2.imshow(gt\_image) # ax3.imshow(gt\_crop) # plt.show()
tree = etree.ElementTree(annotation)tree.write(os.path.join(xml\_path, filename.split(&#39;.&#39;)[**0**] + &#39;.xml&#39;),pretty\_print=True, xml\_declaration=True, encoding=&#39;utf-8&#39;)

**def**** object2patch**(xml\_path, img\_shape):patch = &#39;./data/patch/&#39;os.makedirs(patch, exist\_ok=True)record\_file = &#39;./data/patch/label.txt&#39;
records = []counter = **0**** for **path** in** tqdm(sorted(Path(xml\_path).glob(&#39;\*.xml&#39;)), desc=&#39;object2patch&#39;):
xml = etree.parse(str(path))root = xml.getroot()
filename = root.find(&#39;filename&#39;).textrgb\_image = cv2.imread(&#39;./data/rgb/&#39; + filename)nir\_image = cv2.imread(&#39;./data/nir/&#39; + filename, cv2.COLOR\_BGR2GRAY)mask\_image = cv2.imread(&#39;./data/ndvi/&#39; + filename,cv2.COLOR\_BGR2GRAY) / **255**
**for** boxes **in** root.iter(&#39;object&#39;):ymin, xmin, ymax, xmax = None, None, None, None
**for** box **in** boxes.findall(&quot;bndbox&quot;):ymin = int(box.find(&quot;ymin&quot;).text)xmin = int(box.find(&quot;xmin&quot;).text)ymax = int(box.find(&quot;ymax&quot;).text)xmax = int(box.find(&quot;xmax&quot;).text)
label = boxes.find(&#39;label&#39;).textname = boxes.find(&#39;name&#39;).text
# rgb, nir, masksave\_filename = patch + &#39;%08d\_%s.png&#39;counter += **1**
rgb\_crop = cv2.resize(rgb\_image[xmin:xmax, ymin:ymax, :] \* np.stack((mask\_image[xmin:xmax, ymin:ymax],) \* **3** , axis=- **1** ), img\_shape)
nir\_crop = cv2.resize(nir\_image[xmin:xmax, ymin:ymax] \* mask\_image[xmin:xmax,ymin:ymax], img\_shape)
img\_crop = np.concatenate([rgb\_crop, np.expand\_dims(nir\_crop,axis=- **1** )],axis = - **1** ) / **255.0**
# save file to rgb, nir, npy fileimg\_filename = save\_filename.replace(&#39;png&#39;, &#39;npy&#39;) % (counter,&#39;all&#39;)rgb\_filename = save\_filename % (counter, &#39;rgb&#39;)nir\_filename = save\_filename %(counter, &#39;nir&#39;)
np.save(img\_filename, img\_crop)cv2.imwrite(rgb\_filename, rgb\_crop)cv2.imwrite(nir\_filename, nir\_crop)line = (filename, img\_filename, rgb\_filename, nir\_filename, name,label)
records.append(line)
# write to label file **with** open(record\_file, &#39;w&#39;) **as** f: **for** line **in** records:line\_string = &#39;,&#39;.join(line) + &#39; **\n**&#39;f.write(line\_string) **if** \_\_name\_\_ == &#39;\_\_main\_\_&#39;:paths = read\_image\_path(&#39;./data/rgb&#39;, &#39;./data/nir&#39;)write2ndvi(paths)# **for** path **in** tqdm(Path(&#39;./data/ndvi&#39;).glob(&#39;\*.png&#39;)):segment2boundingbox(str(path), str(path).replace(&#39;ndvi&#39;, &#39;gt&#39;))
object2patch(&#39;./data/xml&#39;, ( **224** , **224** )) |
| --- |

2. **Training RBM and predict in test data**

| **import**** numpy ****as**** np ****from**** sklearn ****import** linear\_model, clone, metrics **from**** sklearn.discriminant\_analysis ****import** LinearDiscriminantAnalysis **from**** sklearn.discriminant\_analysis ****import** QuadraticDiscriminantAnalysis **from**** sklearn.metrics ****import** accuracy\_score, average\_precision\_score, \classification\_report **from**** sklearn.neural\_network ****import** BernoulliRBM **from**** sklearn.pipeline ****import** Pipeline **from**** sklearn.preprocessing ****import** scale
**from**** feature\_extractor ****import** create\_lbp\_texture\_feature, create\_color\_feature, \create\_hog\_feature

**def**** prepare\_data**(x):lbp = create\_lbp\_texture\_feature(x)color = create\_color\_feature(x)hog = create\_hog\_feature(x)
x\_feature = np.concatenate([lbp, color, hog], axis=- **1** )x\_feature = scale(x\_feature)
**return** x\_feature
**def**** train\_test\_data**(is\_feature = True):# data = np.load(&quot;data224.npz&quot;)# x\_train, y\_train, x\_test, y\_test = data[&#39;x\_train&#39;], data[&#39;y\_train&#39;], data[# &#39;x\_test&#39;], data[&#39;y\_test&#39;]# print(&#39;Preparing the data finish!&#39;)## X\_train\_feature = prepare\_data(x\_train)# X\_test\_feature = prepare\_data(x\_test)
**if** is\_feature:data = np.load(&#39;feature.npz&#39;)X\_train\_feature, X\_test\_feature, y\_train, y\_test = data[&#39;x\_train&#39;], data[&#39;x\_test&#39;], data[&#39;y\_train&#39;], data[&#39;y\_test&#39;] **print** (&#39;Preparing the data finish!&#39;) **return** X\_train\_feature, y\_train, X\_test\_feature, y\_test **else** :data = np.load(&quot;data.npz&quot;) **print** (&#39;Preparing the data finish!&#39;) **return** data[&#39;x\_train&#39;], data[&#39;y\_train&#39;], data[&#39;x\_test&#39;], data[&#39;y\_test&#39;]
**def**** result\_analysis**(y\_pred, y\_truth, description = &#39;&#39;):**print**(&#39;-----------------%s------------------&#39; %description)**print**(&#39;Test accuracy:&#39;, accuracy\_score(y\_true=y\_truth, y\_pred=y\_pred))**print**(&#39;Average Test accuracy:&#39;,average\_precision\_score(y\_true=y\_truth, y\_score=y\_pred))**print**(classification\_report(y\_truth, y\_pred, target\_names=[&#39;crop&#39;, &#39;weed&#39;]))
**def**** rbm**():X\_train, Y\_train, X\_test, Y\_test = train\_test\_data(is\_feature = False)
rbm = BernoulliRBM(random\_state= **0** , verbose=True)logistic = linear\_model.LogisticRegression(solver=&#39;newton-cg&#39;, tol= **1** )rbm\_features\_classifier = Pipeline(steps=[(&#39;rbm&#39;, rbm), (&#39;logistic&#39;, logistic)])
rbm.learning\_rate = **0.06** rbm.n\_iter = **10** # More components tend to give better prediction performance, but larger# fitting timerbm.n\_components = **100** logistic.C = **50**
X\_train = X\_train.reshape(X\_train.shape[**0**], - **1** )# Training RBM-Logistic Pipelinerbm\_features\_classifier.fit(X\_train, Y\_train)

X\_test = X\_test.reshape(X\_test.shape[**0**], - **1** )Y\_pred = rbm\_features\_classifier.predict(X\_test)
result\_analysis(Y\_pred, Y\_test, &#39;BernoulliRBM&#39;)
**if** \_\_name\_\_ == &#39;\_\_main\_\_&#39;:rbm() |
| --- |

10
