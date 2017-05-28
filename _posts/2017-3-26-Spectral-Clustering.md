---
layout: post
title: Spectral Clustering
---

We must first ask, what is clustering? In our case, clustering is the problem of unsupervised grouping of related points in an $$n$$-dimensional feature space. Below we see an example of original data points together with their ideal clustering (based on spatial locality):

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/idealclustering.png){: style="max-width:500px; height: auto;"}

Unsupervised clustering of data is quite an old problem in data science, and numerous algorithms exist for the automatic grouping of data points embedded in a feature space. 

## K-means Clustering ##

One of these old algorithms for unsupervised clustering that is widely used today is K-means. This is a simple algorithm for euclidean distance based clustering that utilizes the Euclidean metric:

$$
||\vec{v}||_2 = \sqrt{|v_1|^2 + |v_2|^2 + \cdots + |v_n|^2}
$$

More concretely, the K-means algorithm is described as follows:

We are given $$x^1...x^m \in \mathbb{R}^n$$ as the vector training set. Our goal is to predict $$k$$ centroids and a cluster label $$c^i$$ for each point.

1. Initialize **cluster centroids** $$\mu_1,\mu_2,...,\mu_k \in \mathbb{R}^n$$ randomly.

2. Repeat until convergence: \{

* For every i, set
 
$$c^i := \text{arg} \min_j ||x^i - \mu_j||^2$$

* For every j, set 

$$\mu_j := \frac{\sum^{m}_{i=1} 1\{c^i = j\}x^i}{\sum^{m}_{i=1} 1\{c^i = j\}}$$

\}

This clustering algorithm can work well, and given a reasonable $$k$$, often finds good clusters for convex data. However, it isn't perfect and there are cases where it fails miserably.

## Motivation for More Advanced Clustering ##

It turns out that K-means is faulty and only works well for convex clusters. Here is an example of K-means run with $$k=2$$ on this data set of 2 clusters:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/kmeansgood.png){: style="max-width:500px; height: auto;"}

As you can see, it separates these two very convex clusters with ease. However, let us take a look at its performance on a trickier two cluster example:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/kmeansbad.png){: style="max-width:500px; height: auto;"}

In this case, K-means does rather poorly. Though a more reasonable clustering would be to cluster the inner points together due to the greater local density of their connections, K-means blindly uses it's Euclidean metric and gives us the garbage above. In order to obtain reasonable clustering for harder cases where we put more value on overall local connectivity than global minimization of average euclidean distance, we are going to need another clustering method. 

## Spectral Clustering ##

You guessed it, Spectral clustering is the unsupervised learning algorithm we are looking for. It is called "Spectral" because we will be using important properties about the spectrum of a particular type of matrix (the spectrum of a matrix is its set of eigenvalues) - the Laplacian matrix - in order to get our algorithm to work. In the following, I hope to give you a thorough enough explanation so you are not only able to apply spectral clustering, but are also able to understand the theory behind why it works. First, we establish some background.

### Necessary Theory ###

* An eigenvector of a matrix $$A$$ is a vector $$\mathbf{v}$$ such that for some $$\lambda \in \mathbb{R}$$ we have:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

* $$\lambda$$ is called the eigenvalue corresponding to the eigenvector $$\mathbf{v}$$

* The quadratic form of an $$n \times n$$ matrix $A$ given a vector $\mathbf{x} \in \mathbb{R}^n$ is:

$$
q_A (x_1,...,x_n) = \sum^{n}_{i=i} \sum^{n}_{j=1} a_{ij} x_i x_j = \mathbf{x}^T A \mathbf{x}
$$

* We call an $$n \times n$$ matrix $A$ positive semi-definite if $$\mathbf{x}^T A \mathbf{x} \geq 0$$

* Given a graph $$G$$, if we call its adjacency matrix $$A$$ and its degree matrix $$D$$ (diagonal matrix that describes number of edges coming out of each node), then its Laplacian matrix $$L$$ can be computed as $$L = D - A$$. For example, for the graph below:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/graphsimple.png){: style="max-width:300px; height: auto;"}

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We would have:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/graphsimpleL.png){: style="max-width:500px; height: auto;"}

### Main Algorithm ###

Okay, now that we have preliminary background out of the way, we should establish the overall structure of the algorithm (so far no intuition):

1. Build similarity graph out of the data (with something like k-nearest neighbors, initially)

2. Compute eigendecomposition of Laplacian matrix representing the similarity graph. Once again recall that the Laplacian matrix is defined as $$L = D - W$$, where $$D$$ is the degree matrix of the similarity graph and $$W$$ is the adjacency matrix of the graph.

3. Do k-means clustering on feature vectors made out of components of the eigenvectors we obtained from the eigendecomposition. 

Alright... so we have a procedure. And it happens that this procedure works well for non-convex clustering. But why??

### Overview: Why the Algorithm Works ###

A rough description of why the algorithm works (which we will later during this post make more rigorous) is given below, with no proof (these are just claims at this point):

1. We have $$L$$, the Laplacian matrix of the similarity graph. Then, the multiplicity of eigenvalue $$0$$ of $$L$$ equals the number of connected components in the graph.

2. The eigenvectors corresponding to this eigenvalue are disjoint, and each represents a connected component of the similarity graph.

3. Thus finding these eigenvectors corresponds to finding the connected components of the similarity graph, which are the clusters we desire.

That's all I will say about the claim for now. Time for some examples. (If you are thinking that this is useless because you might as well just run DFS through the entire graph to find its connected components, I asure you things will get more interesting. We will demonstrate that this method works more generally for "loosely" connected components, which can no longer be easily found via DFS but can still be easily found with spectral clustering).

### General Idea with Example ###

Here is the procedure for an example graph - it is fully connected. The eigenvector of the Laplacian corresponding to the eigenvalue 0 is simply the $$\mathbf{1}$$ vector. This shows that vertices $$1,2,$$and $$3$$ all belong to the same connected component.

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/Lex1.png){: style="max-width:800px; height: auto;"}

Here is another example graph - it is not fully connected. The two eigenvectors of the Laplacian corresponding to the $$0$$ eigenvalue show the two connected components. One shows that vertices $$1,2,$$ and $$3$$ belong to the same connected component, and the other shows that vertices $$4,5,$$ and $$6$$ belong to the same connected component. 

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/Lex2.png){: style="max-width:900px; height: auto;"}

### Establishing Properties to Prove Claims ###

We will go through and establish some mathematical properties regarding our setup in order to prove the claims made in the above section to justify why the algorithm works to find connected components (which we call our clusters) in our similarity graph. Once again, for the below claims, recall that the unnormalized graph Laplacian is defined as $$L = D - W$$:

1. $$L$$ is symmetric because $$D$$ and $$W$$ are symmetric (trivial)

2. $$L$$ is positive semi-definite

3. Because $$L$$ is positive semi-definite, all eigenvalue are greater than or equal to 0

4. The smallest eigenvalue of $$L$$ is 0 and the corresponding eigenvector is $$\mathbf{1}$$

We now go through and prove these properties.

#### 1) L is symmetric ####

$$L = D - W$$, and we know that $$D$$ only has elements along its diagonal and that $$W$$ is symmetric by definition. Thus the difference will still be symmetric. 

#### 2) L is positive semi-definite ####

To prove that $$L$$ is positive semi-definite we need the quadratic form of the matrix $$L$$ to be greater than or equal to 0 for all $$x$$ nonzero. That is, we need:

$$
\forall \mathbf{x} \in \mathbb{R}^n, \mathbf{x} \neq \mathbf{0}: \mathbf{x}'L\mathbf{x} \geq 0
$$

To start, notice that:

$$
\mathbf{x}'L\mathbf{x} = \mathbf{x}'D\mathbf{x}-\mathbf{x}'W\mathbf{x} = \sum^{n}_{i=1} d_i x_i^2 - \sum^{n}_{i,j=1} x_ix_j\omega_{ij}
$$

Now we rewrite slightly:

$$
\sum^{n}_{i=1} d_i x_i^2 -\sum^{n}_{i,j=1} x_i x_j \omega_{ij} = \frac{1}{2}\left(\sum^{n}_{i=1}d_i x_i^2 - 2\sum^{n}_{i,j=1} x_i x_j \omega_{ij} + \sum^{n}_{j=1} d_j x_j^2\right) 
$$

Notice that for a fixed $$i$$ summing along $$j$$ will give you the the sum of the $\omega_{ij}$. This must be $d_i$ due to the fact that in each row the sum of the adjacencies is equal to the degree of the vertex, which is true by definition. We can thus write:

$$
\sum^{n}_{i=1} d_i x_i^2 = \sum^{n}_{i,j=1} \omega_{ij} x_i^2
$$

Using this, we can finally obtain:

$$
\frac{1}{2}\left(\sum^{n}_{i=1}d_i x_i^2 - 2\sum^{n}_{i,j=1} x_i x_j \omega_{ij} + \sum^{n}_{j=1} d_j x_j^2\right) = \frac{1}{2}\left(\sum^{n}_{i,j=1} \omega_{ij} x_i^2 - 2\sum^{n}_{i,j=1} x_i x_j \omega_{ij} + \sum^{n}_{i,j=1} \omega_{ij} x_j^2\right) 
$$

Note that it doesn't matter whether we have $$x_i$$ or $$x_j$$ in our first and third terms - our adjacency matrix is symmetric so the summation will produce the same value. Now that our indexing is consistent, so we can combine the sums into one large sum to obtain:

$$
\frac{1}{2} \left(\sum^{n}_{i,j=1} \omega_{ij}x_i^2 -\omega{ij}2x_i x_j + \omega_{ij}x_j^2 \right) = \frac{1}{2} \sum^{n}_{i,j=1} \omega_{ij} (x_i-x_j)^2
$$

Because $$\omega_{ij}$$ are nonnegative by definition, and because the square term must be nonnegative, we obtain what we desired:

$$
\mathbf{x}'L\mathbf{x} = \frac{1}{2} \sum^{n}_{i,j=1} \omega_{ij} (x_i-x_j)^2 \geq 0 
$$

Thus $$L$$ is positive semi-definite.

#### 3) Because $$L$$ is Positive Semi-definite, All Eigenvalues Are Nonnegative ####

To see this we simply choose $$v_i \in E_\lambda (L)$$ and substitute into the quadratic form of $$L$$ to find:

$$
\mathbf{x}'L\mathbf{x} = v_i'(Lv_i) = v_i' \lambda_i v_i = \lambda_i v_i' v_i = \lambda |v_i|^2 \geq 0
$$

Note that in the above, $Lv_i = \lambda_i v_i$ by definition, and because vector norm is nonnegative, we have that for any eigenvalue:

$$
\lambda_i \geq 0
$$

As desired.

#### 4) The Smallest Eigenvalue of $$L$$ is $$0$$, and the Corresponding Eigenvector Is the $$\mathbf{1}$$ vector (full if $$L$$ is connected, partial if $$L$$ is disconnected) ####

Note that the rows of the Laplacian add up to $$0$$ by definition (once again, because the degree value is equal to the sum of adjacencies), so $$0$$ must be an eigenvalue. It is easiest to see why this is the case with an example. Multiply the following matrix $$L$$ by the $$\mathbf{1}$$ vector:

$$
\begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} = 0 \cdot \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}
$$

Thus $$0$$ is an eigenvalue, and the corresponding eigenvector must be $\mathbf{1}$.

### Proving the Main Claims ###

Recall our previously stated Main Claims from the section "Why the Algorithm Works":

1. We are given $$L$$, the Laplacian matrix of the similarity graph. Then, the multiplicity of eigenvalue $$0$$ equals the number of connected components in the graph (let us call this number $$k$$).

2. The eigenvectors corresponding to this eigenvalue are disjoint, and each represents a connected component of the similarity graph.

Armed with our above proved properties, let us first try to prove the main claim for a graph that is just a single connected component. 

#### Main Claim for 1 Connected Component ####

Note that the entire graph is connected. Assume that $$v$$ is an eigenvector of the graph's Laplacian matrix $$L$$, with corresponding eigenvalue $$0$$ (we saw from property 4 that every Laplacian matrix must have at least one eigenvector that corresponds to eigenvalue $$0$$). Then we have:

$$
\mathbf{x}'L\mathbf{x} = v'Lv = v'\lambda v= \lambda v'v = 0 v'v = 0
$$

So we now have:

$$
0 = v'Lv = \sum^{n}_{i,j=1} \omega_{ij} (v_i-v_j)^2
$$

This implies that $$v_i = v_j\:\forall i,j$$ because the sum on the right must be $$0$$. Moreover, $$v_i = v_j \forall i,j$$ provides the unique solution for our eigenvector, which is simply the $$\mathbf{1}$$ vector, as all of its components are equal. Thus, the multiplicity of our $$0$$ eigenvalue is exactly 1, as expected because we have only 1 connected component (the entire graph). In addition, our eigenvector is the entire $\mathbf{1}$ vector, meaning that every vertex in our graph is part of the component (also as expected). 

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/1comp.png){: style="max-width:600px; height: auto;"}

#### Main Claim for k Connected Components ####

In a graph with k connected components, the Laplacian will look like a block matrix, where each block corresponds to a connected component:

$$
L = \begin{pmatrix}L_1 & & & \\ & L_2 & & \\ & & \ddots & \\ & & & L_k \end{pmatrix}
$$

The previous claim for 1 connected component applies to each of the blocks individually, and as a result we get that the multiplicity of our eigenvalue 0 is $k$, where each of the $k$ eigenvectors indicates a connected component, as desired. Recall that we already saw such a block matrix in an earlier example:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/Lex2.png){: style="max-width:900px; height: auto;"}

### Loosely Connected Clusters ###

So in practice, your similarity graph won't usually be perfectly separated into connected components. There will be some noise, some additional edges, that sparesely connect densely connected components. How does spectral clustering deal with such instances? Well, in practice, you typically set your $k$ to be roughly the multiplicity of your 0 eigenvalue (usually this multiplicity includes eigenvectors of eigenvalues that are almost 0 as well). It turns out if you take the first $k$ eigenvectors out of your list of eigenvectors sorted from smallest to largest eigenvalue, these $k$ eigenvectors usually contain enough information to separate your data into $k$ clusters. To obtain intuition behind why this works, let's take a look at an example. 

In the following graph we have two "Loosely" connected clusters:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/loosegraph.png){: style="max-width:600px; height: auto;"}

Because we are looking for $$2$$ clusters, we will set $$k=2$$ and look at the 2 eigenvectors of the Laplacian corresponding to the smallest eigenvalues:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/loosegraph2.png){: style="max-width:600px; height: auto;"}

Notice that in this case, although the eigenvector corresponding to the eigenvalue $$0$$ did not give us any useful information for separating the clusters, the eigenvector corresponding to the second smallest eigenvalues did - with the difference in sign, it clearly indicated what vertices were part of what cluster. 

Now to continute and bring this full circle back to the initial algorithm, what we do is we stack the $$k$$ eigenvectors corresponding to the smallest eigenvalues as columns in a new matrix, interpret each row of this matrix as points in $$k$$-dimensional space, and perform k-means clustering on these points. From the example above, we would be clustering the points:

$$
(1,-0.561553)\:\:(1,-1)\:\:(1,-1)\:\:(1,0.561553)\:\:(1,1)\:\:(1,1)
$$

Notice that because of the sign differences the K-means clustering algorithm will be able to cluster these points into the correct clusters easily. To make it even more clear, let us see how this works with one of our old examples:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/finalexample.png){: style="max-width:900px; height: auto;"}

So in this case the two connected components were separated very cleanly into two perfectly separate clusters indicated by the blue and red dots on the resulting graph.

In general, the intuition behind why Spectral clustering works is that it uses the spectral properties of the Laplacian matrix to project data from a potentially very complex space represented by the similarity graph, to a more simple space, where locally connected regions of the graph are likely to end up closer together and are likely to form more convex shapes due to the inherent separation provided by the eigenvectors of our Laplacian. Because in this projected region convex clusters are more likely, it is common practice to use K-means after the projection (as we mentioned previously, K-means works well with convex data). We will avoid going into further theoretical details here, or this tutorial will drag on for too long, but if you are interested please Google "Spectral Graph Theory", "Fiedler Vectors", or "Graph Partitioning Spectral Methods" to find out more. 

## Spectral Clustering Performance Examples ##

All in all, spectral clustering has been shown to give much better results that K-means for non-convex clustering from active work experience. Here is an example where spectral clustering gives us a reasonable clustering on a hard data set:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/spectral1.png){: style="max-width:600px; height: auto;"}

Here are some more reasonably impressive results of spectral clustering on hard data:

{: style="text-align:center"}
![Ideal Clustering]({{ site.baseurl }}/images/blogpost1/spectral2.png){: style="max-width:600px; height: auto;"}

## Further Extensions ##

For the sake of brevity, I had to ommit a lot of material regarding spectral clustering. As opposed to just constructing a similarity graph via an adjacency matrix, there are a whole host of other ways to construct the similarity graph mentioned by Ulrike von Luxburg in the below reference to his paper on Spectral Clustering. In addition, a different distance metric could be use in the construction of the similarity graph (for example, Gaussian distance). In practice the issues of noise between clusters can also be reduced by applying minimum graph cuts to the similarity graph and also by using different normalized versions of the Laplacian matrix (in this tutorial/discussion regarding intuition we only used the unnormalized Laplacian matrix). Please refere to the reference links I have listed below or to Google to sufficiently satiate your curiosity for spectral clustering and spectral methods of graph partitioning. 

## Links to References ##

Ulrike von Luxburg's excellent paper - [Spectral Clustering Main Paper](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf)

A few more helpful links:

[A helpful blog](http://blog.shriphani.com/2015/04/06/the-smallest-eigenvalues-of-a-graph-laplacian/)

[Another cool paper](http://musingsfromming.blogspot.com/2012/07/spectral-clustering-intuition.html)

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

