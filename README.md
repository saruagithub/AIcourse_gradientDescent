<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
##### environment: pycharm, python3.6
AIcourse_gradientDescent
homework in Ai basic course

##### homework1
here are 1000 datas (x,y)

hypothesis = theta1 + theta2 * x

loss_function = (hypothesis - y)^2

finish the SGD & BachGD algorithm, adding some tips to improve the optimization process


* * *

##### homework2

train 'a3data1' to get alpha, then apply alpha to different sigma Gaussian & IMQ interpolation in 'a3data1t'.

draw LOOCV error graph with sigma changed in 'a3data1'

θ = (α,σ), Gaussian & IMQ interpolation by small batch gradient method

###### RBF interpolation:

* Gaussian function

```math
\phi(x,\sigma) = e^{-\sigma^2||x||_2^2}
```

* interpolation condition

```math
\begin{pmatrix}
\phi_{11} & \phi_{12} &\cdots & \phi_{1n} &\\
\phi_{21} & \phi_{22} &\cdots & \phi_{2n} &\\
\vdots & \vdots &\ddots & \vdots&\\
\phi_{n1} & \phi_{n2} &\cdots & \phi_{nn} &
\end{pmatrix}

\times

\begin{pmatrix}
\alpha_1 &\\
\alpha_2 &\\
\vdots & \\
\alpha_n &
\end{pmatrix}
=
\begin{pmatrix}
y_1 &\\
y_2 &\\
\vdots&\\
y_n &
\end{pmatrix}
```
