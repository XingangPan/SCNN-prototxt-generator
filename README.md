## Spatial CNN Prototxt Generator

This script is for generating [Spatial CNN](https://github.com/XingangPan/SCNN) caffe prototxt file.

### How to use
- Run run.sh to generate Spatial CNN module, modify parameters to yours.
'--height', '--width', and '--channel' are number of rows, columns, and channels of the last feature maps respectively.
- Copy the text in the generated SCNN.prototxt, and paste it to the right position in your original model prototxt file (the one that you want to add Spatial CNN).
- Don't forget to modify the 'bottom' of the next layer to "SCNN".
- Example: 
Original prototxt file: examples/resnet101.prototxt  
Generated SCNN text: examples/SCNN.prototxt  
New prototxt file with SCNN: examples/resnet101_SCNN.prototxt  
(For this example, the prototxt is aligned with the caffe version at https://github.com/hszhao/PSPNet)  
You can visualize the network architecture at http://ethereon.github.io/netscope/#/editor.  

### About Spatial CNN (SCNN)
![SCNN](examples/SCNN.png)
- Spatial CNN enables explicit and effective spatial information propagation between neurons in the same layer of a CNN. 
- For more details, please refer to [our paper](https://arxiv.org/abs/1712.06080)

### Practical Concerns
- In practice I initialize SCNN layers with variance sqrt(5) times smaller than the MSRA initialization for numarical stability concern.
- It would be helpful to initialize all other layers with a pretrained model. Directly training from random initialization may cause SCNN to diverge.
- The best way to utilize SCNN is to insert it right after the last feature maps (top hidden layer, before the classification layer).
- The number of channels (num_c) of the last feature maps should not be too large since the memory consumption of SCNN is sensitive to num_c.  
In practive I find 128 to be an appropriate choice for num_c, 512 might be a little bit large. You can try larger num_c if you have enough GPU memory.
- To reduce num_c in ResNet50/101, directly change num_output from 512 to 128 in "Conv5_4" layer may decrease the performance. Aternatively, you can add an extra "Conv5_5" layer to map the 512 feature maps to 128 with 1x1 convolution, as did in 'examples/resnet101.prototxt'.
