#!/usr/bin/env python
"""
Generate prototxt of the Spatial CNN for Caffe.
Paper: https://arxiv.org/pdf/1712.06080.pdf
"""
import argparse
import sys
import math

def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate Spatial CNN Prototxt')
    parser.add_argument('--height', dest='h', default=90, type=int,
                        help='number of rows in feature maps ')
    parser.add_argument('--width', dest='w', default=90, type=int,
                        help='number of colomns in feature maps ')
    parser.add_argument('--kernel_width', dest='kw', default=9, type=int,
                        help='kernel width of Spatial CNN')
    parser.add_argument('--channel', default=128, type=int,
                        help='number of channels in Spatial CNN')
    parser.add_argument('--bottom', default='conv5_4', type=str,
                        help='name of bottom blob')
    parser.add_argument('--output', default='SCNN.prototxt', type=str,
                        help='Output Spatial CNN prototxt file')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def generate_conv_layer_no_bias(name, bottom, top, num_output, kernel_h, kernel_w, pad_h, pad_w, std=0.01):
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param { lr_mult: 1 decay_mult: 1 }
  convolution_param {
    num_output: %d
    kernel_h: %d
    kernel_w: %d
    pad_h: %d
    pad_w: %d
    stride: 1
    bias_term: false
    weight_filler { type: "gaussian" std: %.5f }
  }
}\n'''%(name, bottom, top, num_output, kernel_h, kernel_w, pad_h, pad_w, std)
    return conv_layer_str

def generate_activation_layer(name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}\n'''%(name, bottom, bottom, act_type)
    return act_layer_str

def generate_eltwise_layer(name, bottom0, bottom1, top):
    eltwise_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  type: "Eltwise"
  eltwise_param {
    operation: SUM
  }
}\n'''%(name, bottom0, bottom1, top)
    return eltwise_layer_str

def generate_slice_layer(name, bottom, top, axis, num):
    slice_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  type: "Slice"\n'''%(name, bottom)
    for i in range(num):
        top_i = top + '_' + str(i+1)
        slice_layer_str += '''  top: "%s"\n'''%(top_i)
    slice_layer_str += '''  slice_param {
    axis: %d\n'''%(axis)
    for i in range(num - 1):
        slice_layer_str += '''    slice_point: %s\n'''%(str(i+1))
    slice_layer_str += '''  }
}\n'''
    return slice_layer_str

def generate_concat_layer(name, bottom, top, last, axis, num):
    concat_layer_str = '''layer {
  name: "%s"
  type: "Concat"\n'''%(name)
    for i in range(num - 1):
        bottom_i = bottom + '_' + str(i+1)
        concat_layer_str += ''' bottom: "%s"\n'''%(bottom_i)
    bottom_last = last + '_' + str(num)
    concat_layer_str += ''' bottom: "%s"\n'''%(bottom_last)
    concat_layer_str += '''  top: "%s"
  concat_param {
    axis: %d
  }
}\n'''%(top, axis)
    return concat_layer_str

def generate_SCNN(args):
    network_str = generate_slice_layer('Slice1', args.bottom, 'slice1', 2, args.h)
    std = math.sqrt(2. / (args.kw * args.channel * 5))
    # Build SCNN downward
    for i in range(1, args.h):
        name = 'SCNN_D_' + str(i)
        if i == 1:
            bottom = 'slice1_1'
        else:
            bottom = name
        top = 'SCNN_D_' + str(i) + '/message'
        target = 'slice1_' + str(i + 1)
        summation = 'SCNN_D_' + str(i + 1)
        network_str += generate_conv_layer_no_bias(name, bottom, top, args.channel, 1, args.kw, 0, (args.kw-1)/2, std)
        network_str += generate_activation_layer(name+'/relu', top)
        network_str += generate_eltwise_layer(name+'/sum', top, target, summation)

    # Build SCNN upward
    for i in range(args.h, 1, -1):
        name = 'SCNN_U_' + str(i)
        top = name+ '/message'
        if i == args.h:
            bottom = 'SCNN_D_' + str(i)
        else:
            bottom = 'SCNN_U_' + str(i)
        if i == 2:
            target = 'slice1_1'
        else:
            target = 'SCNN_D_' + str(i - 1)
        summation = 'SCNN_U_' + str(i - 1)
        network_str += generate_conv_layer_no_bias(name, bottom, top, args.channel, 1, args.kw, 0, (args.kw-1)/2, std)
        network_str += generate_activation_layer(name+'/relu', top)
        network_str += generate_eltwise_layer(name+'/sum', top, target, summation)

    network_str += generate_concat_layer('Concat1', 'SCNN_U', 'SCNN_U', 'SCNN_D', 2, args.h)
    network_str += generate_slice_layer('Slice2', 'SCNN_U', 'slice2', 3, args.w)

    # Build SCNN rightward
    for i in range(1, args.w):
        name = 'SCNN_R_' + str(i)
        if i == 1:
            bottom = 'slice2_1'
        else:
            bottom = name
        top = 'SCNN_R_' + str(i) + '/message'
        target = 'slice2_' + str(i + 1)
        summation = 'SCNN_R_' + str(i + 1)
        network_str += generate_conv_layer_no_bias(name, bottom, top, args.channel, args.kw, 1, (args.kw-1)/2, 0, std)
        network_str += generate_activation_layer(name+'/relu', top)
        network_str += generate_eltwise_layer(name+'/sum', top, target, summation)

    # Build SCNN leftward
    for i in range(args.w, 1, -1):
        name = 'SCNN_L_' + str(i)
        top = name + '/message'
        if i == args.w:
            bottom = 'SCNN_R_' + str(i)
        else:
            bottom = 'SCNN_L_' + str(i)
        if i == 2:
            target = 'slice2_1'
        else:
            target = 'SCNN_R_' + str(i - 1)
        summation = 'SCNN_L_' + str(i - 1)
        network_str += generate_conv_layer_no_bias(name, bottom, top, args.channel, args.kw, 1, (args.kw-1)/2, 0, std)
        network_str += generate_activation_layer(name+'/relu', top)
        network_str += generate_eltwise_layer(name+'/sum', top, target, summation)

    network_str += generate_concat_layer('Concat2', 'SCNN_L', 'SCNN', 'SCNN_R', 3, args.w)
    return network_str

def main():
    args = parse_args()
    scnn_pt = generate_SCNN(args)

    fp = open(args.output, 'w')
    fp.write(scnn_pt)
    fp.close()

if __name__ == '__main__':
    main()
