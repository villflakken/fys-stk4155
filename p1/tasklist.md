# List of tasks

## Part (a)

The fundamental conduct of code and linear algebra.

### 1: Make data

- [ ] Generate dataset.
  - [ ] With stochastic noise.

### 2: Analysis

- [ ] Perform a "standard least square regression analysis,
      using polynomials in $x$ and $y$ up to fifth order".
  - May use code from hw1 & hw2.
- [ ] Find "confidence intervals of the parameters (estimators) $\bm{\beta}$" by...
  - [ ] computing their **variances**,
  - [ ] Evaluate the **MSE**,
  - [ ] and the **R2 score function**.
- [ ] And then **scale the data** (e.g. by subtracting the mean value)
- See hw2 for examples.
- Use  SciKit-Learn for splitting "training" and "test" data, use function `train_test_split`.
- Also suggested:
    **normalization functionality of SKL**,
    as demonstrated in **hw2**, **exercise 2**.
  
## Part (b)

Studying bias-variance trade-off by implementing **bootstrap** resampling technique, with OLS code and in context of "continuous predictions" (such as regression).

### 1: _Before_ conducting bias-variance trade-off on test data

Make a figure similar to Fig. 2.11 of Hastie, Tibshirani, and Friedman to:

- [ ] Display the test and training MSEs.
- [ ] Use test MSE to indicate possible regions of low/high bias and variance.
  - (Try to get a smooth curve!)

### 2: _Now_, the bias-variance trade-off analysis

Consider a dataset $\mathcal{L}$, consisting of data $\bm{X}_\mathcal{L}=\{(y_j, \boldsymbol{x}_j), j=0\ldots n-1\}$

- [ ] Create noisy data.
  - Assume true data from noisy model:
    $\\y = f(x) + \varepsilon,\quad\varepsilon\ is\ \mathcal{N}(\mu=0, \sigma^2)$
- [ ] Define f in terms of the **bias parameter** $\bm{\beta}$.
  - (from OLS): assume to define approximation of the function
    $f(x) = \tilde{y} = \bm{X}\bm{\beta}$.
- [ ] Find parameters for $\bm{\beta}$ by optimizing the mean **squared error**, via the **Cost Function**:
$$
C(\bm{X},\bm{\beta})
  = \frac{1}{n}\sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2
  = \mathbb{E}\left[(\bm{y}-\bm{\tilde{y}})^2\right].
$$
  - *(?)* With $\mathbb{E}\left[(\bm{y}-\bm{\tilde{y}})^2\right]$, show that "**the expectation value is the sample value**".
  - [ ] Show how to rewrite this into
$$
\mathbf{E}\left[(\bm{y}-\bm{\tilde{y}})^2\right] =
    \frac{1}{n}\sum_i(f_i-\mathbb{E}\left[\bm{\tilde{y}}\right])^2
  + \frac{1}{n}\sum_i(\tilde{y}_i-\mathbb{E}\left[\bm{\tilde{y}}\right])^2
  + \sigma^2.
$$
- [ ] Perform a **bias-variance analysis** of the Franke function, by studiyng the **MSE** values as a function of the **complexity** of used model.
- [ ] Discuss **bias and variance trade-off** as function of your model **complexity** and the **number of data points**, 
  - [ ] and possibly also on **training and test data** using the **bootstrap** resampling method.
- Note:
    When calculating bias, in all applications one doesn't know the function values $f_i$,
    so these would hence be replaced with actual data points $y_i$.

## Part (c)

Cross-validation as resampling techniques (adding more complexity).

### 1: Before we begin

- [ ] Scale the data.

### 2: Implement

- [ ] Implement **$k$-fold cross-validation algorithm** (by own code, this time).
  - [ ] Evaluate the **MSE** function resulting from **test** folds.
    - [ ] Compare **own code**'s results, with results that utilize **SKL**.
  - [ ] Compare MSE from **cross-validation** code, with MSE that came from **bootstrap**.
  - Try 5-10 folds.
    - [ ] Compare own **cross-validation code**, with the one provided by **SKL**.

## Part (d)

Ridge Regression on the Franke function with resampling.

### 1: Write your own code

- [ ] ...for the **Ridge** method, either by using **matrix inversion**,
      or the **singular value decomposition** - as done in the previous exercise, or howework 2.
  - [see also chapter 3.4 of Hastie et al., equations (3.43) and (3.44)].
- [ ] Perform the same **bootstrap analysis** as in the part (b) (for the same polynomials)
  - [ ] and the **cross-validation** part in part (c), but now for **different values** of λ.
- [ ] Compare and analyze your results with those obtained in parts (a)-(c):
  - Study the dependence on λ.
- [ ] Study also the **bias-variance trade-off** as function of various values of the parameter λ.
- [ ] For the bias-variance trade-off, **use the bootstrap resampling method**. Comment your results.

## Part (e)

Lasso Regression on the Franke function with resampling. This part is essentially a repeat of the previous two ones, but now with

### 1: Lasso regression

Write either your own code (difficult and optional) or, in this case,
you can also use the functionalities of Scikit-Learn (recommended).

- [ ] Give a critical discussion of the three methods and a judgement of which model fits the data best.
- [ ] Perform here as well an analysis of the **bias-variance trade-off**,
      using the **bootstrap resampling technique**
      and an analysis of the **mean squared error** using **cross-validation**.

## Part (f)

Introducing real data and preparing the data analysis.

With our codes functioning and having been tested properly on a simpler function, we are now ready to look at real data.

We will essentially repeat in part (g) what was done in parts (a)-(e). However, we need first to download the data and prepare properly the inputs to our codes. We are going to download digital terrain data from the website [https://earthexplorer.usgs.gov/](https://earthexplorer.usgs.gov/).

In order to obtain data for a specific region, you need to register as a user (free) at this website and then decide upon which area you want to fetch the digital terrain data from. In order to be able to read the data properly, you need to specify that the format should be SRTM Arc-Second Global and download the data as a GeoTIF file. The files are then stored in /tif/ format which can be imported into a Python program using `scipy.misc.imread`

Here is a simple part of a Python code which reads and plots the data from such files:

```python
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrain1 = imread(’SRTM_data_Norway_1.tif’)
# Show the terrain
plt.figure()
plt.title(’Terrain over Norway 1’)
plt.imshow(terrain1, cmap=’gray’)
plt.xlabel(’X’)
plt.ylabel(’Y’)
plt.show()
```

If you should have problems in downloading the digital terrain data, we provide two examples under the data folder of project 1. One is from a region close to Stavanger in Norway and the other Møsvatn Austfjell, again in Norway. Feel free to produce your own terrain data.

Alternatively, if you would like to use another data set, feel free to do so. This could be data close to your reseach area or simply a data set you found interesting. See for example kaggle.com for examples.

## Part (g)

OLS, Ridge and Lasso regression with resampling.

Our final part deals with the parameterization of your digital terrain data (or your own
data).

We will:

- [ ]  apply **all three methods** for **linear regression**,
  - [ ]  (OLS)
  - [ ]  (Ridge)
  - [ ]  (Lasso)
- [ ] as well as the same type (or higher order) of **polynomial approximation** and **cross-validation as resampling technique** to evaluate which model fits the data best.

At the end, you should present:

- [ ] a critical evaluation of your results, and
- [ ] discuss the applicability of these regression methods to the type of data presented here (either the terrain data we propose or other data sets).
