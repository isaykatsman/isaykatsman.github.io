---
layout: post
title: "CVPR 2018 Paper Review: Analytic Expressions for Probabilistic Moments of PLDNNs"
---

This past Summer I had the amazing opportunity to head to CVPR with my lab and present my work [Generative Adversarial Perturbations](http://openaccess.thecvf.com/content_cvpr_2018/papers/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.pdf). Of course I also wasn't going to pass up this awesome opportunity to see what was going on in the field and talk to authors directly about their work.

One such paper that stood out to me was one that studied moments of the output distribution of Piecewise-linear Deep Neural Networks (PL-DNNs) conditioned on their intput distribution. I thought that the applications shown in the paper of using this method to better analytically study adversarial examples was reasonably interesting. I'll do my best to summarize this paper in this blog post and highlight the contributions I think are interesting. 

## Arbitrary Style Transfer ##

Because this work was put on hold after it did not produce publishable results after a month of efforts, I will attempt to keep this section brief. 

Inspired by [Gatys et al](https://arxiv.org/pdf/1508.06576.pdf) (original style transfer paper) and [Johnson et al](https://arxiv.org/pdf/1603.08155.pdf) (recent work with fast style transfer), we attempted to design a network that would be able to do fast arbitrary style transfer. Fundamentally, the network would take two inputs: a content target image and a style target image and would output an image with the content of the content target except in the style of the style target. Instead of training a single network to handle just one style image like in Johnson's work, the idea was to extend the style image space and make it as arbitrary as the content target space. 

The first part of the network transfers the desired content image and style image pair into a latent space. In order to do this, we forward the style image through a CNN (like VGG16) to obtain weights for the convolutional layers after pooling (pool1, pool2, pool3, pool4) and we make a stack of the gram matrices of these pooled layer weights in order to obtain a representation for the style of the image. Each gram matrix is scaled to size 256x256, and 32 of each gram matrix is added for redundancy, giving a tensor size of 256x256x128. To obtain a representation for the content, we forward the content image through a CNN (like VGG16) and obtain the weights for the layer relu3 after pooling, and scale it spatially so that the final matrix has size 256x256x256. To obtain a combined style-content latent representation, we simply concatenate these outputs. The inspiration for stacking gram matrices together with content features in this way comes from [Zhang et al](https://arxiv.org/pdf/1612.03242.pdf), which introduced StackGANs. This concatenated tensor is then forwarded through a high capacity Deconvolution Network (large number of filters) to obtain an output image of size 256x256x3. This entire first part of the network is summarized in the following diagram:

{: style="text-align:center"}
![Arbitrary Style Transfer Network 1]({{ site.baseurl }}/images/blogpost2/StyleTransferArbitraryGram.png){: style="max-width:800px; height: auto;"}

From this output image, a similar loss as what is mentioned in Johnson et al is used. A style loss is computed by forwarding the output through VGG16 and computing the $$l_2$$-norm between pool1, pool2, pool3, and pool4 of this output and the original pool layers of the style image, and then adding these four values together. A content loss is computed by forwarding the output through VGG16 and computing the $$l_2$$-norm between relu3 of this output and the original relu3 of the content image. These losses are added together to compute a final joint loss, and this is what the network optimizes. The below image summarizes this portion of the network:

{: style="text-align:center"}
![Arbitrary Style Transfer Network 2]({{ site.baseurl }}/images/blogpost2/StyleTransferArbitraryGram2.png){: style="max-width:800px; height: auto;"}

Eventually, due to lack of competitive results with recent works released regarding arbitrary style transfer, we put this project on hold and proceeded to work on a different research area. 

## Generating Adversarial Examples ##

Previous work on generating adversarial examples has shown that we can compute small per image perturbations to fool CNN classifiers (Dezfooli et al, [DeepFool](https://arxiv.org/pdf/1511.04599.pdf)) and that we can even generate universal perturbations that can be globally added to all images of a data set to fool a CNN (Dezfooli et al, [Univesal Adversarial Perturbations](https://arxiv.org/pdf/1610.08401.pdf)). Just to clarify completely, a picture of the per image perturbation computed by DeepFool for an image of a whale is shown below:

{: style="text-align:center"}
![Deep Foold]({{ site.baseurl }}/images/blogpost2/deepfool.png){: style="max-width:500px; height: auto;"}

The above image of a whale is originally classified as "whale", but after the perturbation is added in, it is classified as "turtle". 

A picture of how classification labels change following the addition of a universal perturbation generated with Dezfooli et al's method is also shown below:

{: style="text-align:center"}
![UAP]({{ site.baseurl }}/images/blogpost2/universalperturbation.png){: style="max-width:500px; height: auto;"}

Instead of generating per image perturbations and universal perturbations via the procedural methods given in the papers above, we wanted to see if we could generate them with deep autoencoder-based networks. The following subsections "Autoencoding Per-Image Perturbations" and "Autoencoding Universal Perturbations" deal with our approaches for these tasks individually. What I will mention prior is that an early result of our work was that the U-Net autoencoder architecture produces much better results than the traditional conv/deconv autoencoder architecture. The main change is that a U-Net concatenates encoder weights to decoder weights during each convolution stage of the deconv. A visual summary of the U-Net architecture is given below:

{: style="text-align:center"}
![UNet]({{ site.baseurl }}/images/blogpost2/unet.png){: style="max-width:600px; height: auto;"}

### Autoencoding Per-Image Perturbations ###

While speaking of per image perturbation autoencoding we will refer to each original image as $$x$$, the perturbation of this image as $$\Delta x$$, and the augmented image with the perturbation applied as $$\widehat{x}=x+\Delta x$$. We attempted two approaches of obtaining the perturbed images. First we tried generating $$\widehat{x}$$ directly from $$x$$ and using a joint loss based on both the magnitude of the perturbation $$\|x-\widehat{x}\|$$ and also the classification loss (of $$\widehat{x}$$ forwarded through the pretrained CNN). This is summarized with the following architecture:

{: style="text-align:center"}
![Per Image Perturbations 1]({{ site.baseurl }}/images/blogpost2/perimage1.png){: style="max-width:1000px; height: auto;"}

We also tried generating $$\Delta x$$ from $$x$$, thresholding this perturbation to some magnitude, adding it to $$x$$ and then using just the classification loss to train. The architecture we used was the following:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimage2.png){: style="max-width:1000px; height: auto;"}

For the first approach where we obtain $$\widehat{x}$$ immediately out of the U-Net without thresholding the noise, we obtained good results for MNIST. There were some slight modifications made (like adding a batchnorm layer) but otherwise the architecture is relatively unchanged. The originals are:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimorig_mnist.png){: style="max-width:1000px; height: auto;"}

And the perturbed images are:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimrecons_mnist.png){: style="max-width:1000px; height: auto;"}

The accuracy of the classifier on the test set of images is only 0.72% (down from 99.1%), and the per image perturbation magnitude is ~2.7. We also obtained reasonably good results for CIFAR-10. A sample of CIFAR-10 originals is:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimorig_cifar10.png){: style="max-width:1000px; height: auto;"}

And the perturbed images are:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimrecons_cifar10.png){: style="max-width:1000px; height: auto;"}

The accuracy of the pretrained CNN on the test set of images is 3.7% (down from 98.5%). There are some ImageNet results as well. The originals are:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimorig_imagenet.png){: style="max-width:800px; height: auto;"}

And the perturbed images:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perimrecons_imagenet.png){: style="max-width:800px; height: auto;"}

The accuracy on these images is reasonable high (40% down from 72%) but the perturbation magnitude is reasonably small. We tried to obtain some better results with this approach, but few yielded significant improvement. One of the unsuccessful approaches was the act of replacing the magnitude loss component of our joint loss with loss from a GAN discriminator trained on MNIST. The intuition is that this could potentially work because a GAN discriminator tells whether our images are real or fake, and as a result can potentially replace the role of the magnitude loss as it provides a "reasonableness" component to the loss that keeps the perturbed images looking similar to the originals. However, the decision boundary of the discriminator of the GAN was likely very complex/rough because it did not contribute meaningfully to the loss and produced bad results (in comparison to the more straightforward magnitude loss). It would be interesting to further investigate how this discirimator boundary can be improved to align better with human intuition. 

For the second approach where we threshold the $$\Delta x$$, we were able to obtain slightly better result for MNIST. The approach of thresholding the magnitude of the perturbation and optimize only the adversarial accuracy is inspired by Dezfooli's Universal Adversarial Perturbation paper. A sample of the original MNIST images is shown below:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perim2_orig.png){: style="max-width:800px; height: auto;"}

The perturbed images are:

{: style="text-align:center"}
![Per Image Perturbations 2]({{ site.baseurl }}/images/blogpost2/perim2_recons.png){: style="max-width:800px; height: auto;"}

The accuracy of the CNN is 0.8% (down from 99.1%) and the magnitude of the perturbation is ~2.4. This is a better result than what we were formerly able to achieve with the first method (though it did take a fair amount of fine tuning), and it is a somewhat more convenient method because you can choose the magnitude threshold beforehand. A critical factor that helped us obtain better results with this method was the fact that we used tanh activations for generating $$\Delta x$$ as opposed to sigmoid activations. In effect, this allowed a subtractive perturbation as the range of tanh is $$-1$$ to $$1$$ and helped the model lower the classification loss significantly. We wanted to obtain results with this thresholding method for universal perturbations next, so we have not yet gotten around to getting more results for this method on other datasets though doing so is definitely on the to-do list once more urgent results with universal perturbations are obtained.

### Autoencoding Universal Perturbations ###

While speaking about universal perturbations we will maintain the terminology that we defined in the per-image perturbation section, and will also add $$U$$ to refer to the universal perturbation. We attempted generating these univesral perturbations by using the U-Net architecture defined previously to go from a fixed uniform noise image to the perturbation. This perturbation was then added to the input image $$x$$ to obtain $$\widehat{x}$$ that was then forwarded through the pretrained CNN to obtain a trainable loss. After several loss updates, the weights of the generating network would change, but the input would be fixed, thus giving us a way to control the universal perturbation based on a fixed input. The architecture for this network is shown below:

{: style="text-align:center"}
![Universal Perturbations]({{ site.baseurl }}/images/blogpost2/univ1.png){: style="max-width:1000px; height: auto;"}

So far we have only generated MNIST results for this network. Here are the original images:

{: style="text-align:center"}
![Universal Perturbations]({{ site.baseurl }}/images/blogpost2/univoriginal.png){: style="max-width:1000px; height: auto;"}

And here are the results perturbed only by a single universal perturbation:

{: style="text-align:center"}
![Universal Perturbations]({{ site.baseurl }}/images/blogpost2/univrecons.png){: style="max-width:800px; height: auto;"}

We obtained these results with a thresholded U magnitude of ~2.5 and the accuracy of the pretrained CNN on these images was ~25% (down from 99.1%). It is important to note that for universal perturbations, typically the best we can do is reduce the network to random guessing (if the universal perturbation is of a small magnitude). This is due to the fact that unlike with per-image perturbations, we can no longer customize our perturbations by image so that for example all 4's are misclassified as 9's (as is possible with adversarial per-image perturbations). This is because now we add the universal perturbation to all images, there is no inherent per-image bias that allows a distinct remapping of 4 $$\rightarrow$$ 9 for example, as all adversarial elements are equally included in the single universal perturbation. Thus even though we were able to get as low as ~0.7% accuracy on MNIST for per image perturbations, the best we can reasonably expect for universal ones is ~10% (1/num_of_classes, as it is random guessing). We are still in the process of obtaining more results for universal perturbations, and this is most urgent on the priority list.

## Future Work ##

Currently, we are still working on expanding our main ideas for adversarial examples, and are in the process of obtaining new results/trying additional models. There are several CNN adversarial defense approaches we want to try, one of which involves **joint** training of the CNN and the adversarial generator in order to try to obtain a better version of brute force defense described by OpenAI ([Goodfellow et al](https://blog.openai.com/adversarial-example-research/)).  In the near future, we will continue pursuing this work to obtain more results and implement/develop our current ideas.

---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

