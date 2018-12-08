##### environment: pycharm, python3.6
AIcourse_gradientDescent
homework in Ai basic course

##### homework1
here are 1000 datas (x,y)
hypothesis = theta1 + theta2 * x
loss_function = (hypothesis - y)^2
finish the SGD & BachGD algorithm, adding some tips to improve the optimization process

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
\phi_11 & \phi_12 &\cdots & \phi_1n &\\
\phi_21 & \phi_22 &\cdots & \phi_2n &\\
\vdots & \vdots &\ddots & \vdots&\\
\phi_n1 & \phi_n2 &\cdots & \phi_nn &
\end{pmatrix}

\times

\begin{pmatrix}
\alpha_1 &\\
\alpha_2 &\\
\cdots & \\
\alpha_n &
\end{pmatrix}\
=
\begin{pmatrix}
y_1 &\\
y_2 &\\
\cdots&\\
y_n &
\end{pmatrix}
```
