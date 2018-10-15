----- .bg-black
@unsplash(0IvpEBGw8aw) 

.wrap 
 @h1 .text-data **Deep Optimizers**

 @div .wrap @div .span
  @button(href="https://github.com/thoppe/DeepOptimizerViz") .alignleft .ghost
   ::github:: github.com/thoppe/DeepOptimizerViz
   
  @button(href="https://www.youtube.com/watch?v=Z-CiRcrJiKo") .alignleft .ghost
   ::youtube:: Watch it live
   
  @button(href="https://twitter.com/metasemantic") .ghost .alignleft
   ::twitter:: @metasemantic 

---- .bg-black 

.wrap
	@h2 What is an optimizer?
	@img(images/GD.png width=400px) 
	
	@h4 
	
		There are lots of them! Gradient Descent, Adam, RMS Prop, Momentum, Adagrad...

	    Used to solve problems in deep learning. In general, can solve any problem with a gradient.

---- .bg-white
 
.wrap
	@h1 .text-landing **What other problems?**
	<br><br><br>
	@h2 $x^2 + 1 = 0$
	@h1 _Solve for x_, get two roots _i, -i_
	<br><br><br>
	@h2 $2.3 x^3 -0.4x^2 + 0.1x + 2.8 = 0$
	@h1 _Solve for x_, get three roots...
	
----- .bg-black

.aligncenter @h1 Shape of convergence depends on the optimizer!

.aligncenter
	@img(src=figures/early_figures/GradientDescent_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/ADAM_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/RMSProp_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/Momentum_quadratic_roots.png width=275px)
	<br> Gradient Descent, ADAM, RMS Prop, Momentum

.aligncenter	
	@img(src=figures/early_figures/ProximalAdagrad_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/ProximalGradientDescent_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/Adagrad_quadratic_roots.png width=275px)
	@img(src=figures/early_figures/FTRL_quadratic_roots.png width=275px)
	<br> Proximal AdaGrad, Proximal Gradient Descent, Adagrad, FTRL

---- .slide-bottom .bg-black 

.aligncenter 
	Add some style. Used `cv2` and image processing tricks<br>
	@img(src=figures/early_figures/ADAM_stylized.png width=600px)

---- .bg-black 

.aligncenter 
 @img(src="figures/ADAM.jpg" width=400px) 
 @img(src="figures/GradientDescent.jpg" width=400px)
 @img(src="figures/RMSProp.jpg" width=400px)
 
 <br> <br> <br>
 .text-landing @button(href="https://www.youtube.com/watch?v=Z-CiRcrJiKo") .ghost ::youtube:: Deep Optimizers
