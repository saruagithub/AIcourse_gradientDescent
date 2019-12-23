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

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(x,\sigma)&space;=&space;e^{-\sigma^2||x||_2^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi(x,\sigma)&space;=&space;e^{-\sigma^2||x||_2^2}" title="\phi(x,\sigma) = e^{-\sigma^2||x||_2^2}" /></a>

* interpolation condition

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;\phi_{11}&space;&&space;\phi_{12}&space;&\cdots&space;&&space;\phi_{1n}&space;&\\&space;\phi_{21}&space;&&space;\phi_{22}&space;&\cdots&space;&&space;\phi_{2n}&space;&\\&space;\vdots&space;&&space;\vdots&space;&\ddots&space;&&space;\vdots&\\&space;\phi_{n1}&space;&&space;\phi_{n2}&space;&\cdots&space;&&space;\phi_{nn}&space;&&space;\end{pmatrix}&space;\times&space;\begin{pmatrix}&space;\alpha_1&space;&\\&space;\alpha_2&space;&\\&space;\vdots&space;&&space;\\&space;\alpha_n&space;&&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;y_1&space;&\\&space;y_2&space;&\\&space;\vdots&\\&space;y_n&space;&&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\begin{pmatrix}&space;\phi_{11}&space;&&space;\phi_{12}&space;&\cdots&space;&&space;\phi_{1n}&space;&\\&space;\phi_{21}&space;&&space;\phi_{22}&space;&\cdots&space;&&space;\phi_{2n}&space;&\\&space;\vdots&space;&&space;\vdots&space;&\ddots&space;&&space;\vdots&\\&space;\phi_{n1}&space;&&space;\phi_{n2}&space;&\cdots&space;&&space;\phi_{nn}&space;&&space;\end{pmatrix}&space;\times&space;\begin{pmatrix}&space;\alpha_1&space;&\\&space;\alpha_2&space;&\\&space;\vdots&space;&&space;\\&space;\alpha_n&space;&&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;y_1&space;&\\&space;y_2&space;&\\&space;\vdots&\\&space;y_n&space;&&space;\end{pmatrix}" title="\begin{pmatrix} \phi_{11} & \phi_{12} &\cdots & \phi_{1n} &\\ \phi_{21} & \phi_{22} &\cdots & \phi_{2n} &\\ \vdots & \vdots &\ddots & \vdots&\\ \phi_{n1} & \phi_{n2} &\cdots & \phi_{nn} & \end{pmatrix} \times \begin{pmatrix} \alpha_1 &\\ \alpha_2 &\\ \vdots & \\ \alpha_n & \end{pmatrix} = \begin{pmatrix} y_1 &\\ y_2 &\\ \vdots&\\ y_n & \end{pmatrix}" /></a>
