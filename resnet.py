import tensorflow as tf


class conv2d_fixed_padding(tf.keras.layers.Layer):
    """
        Conv2d with fixed padding

        Args:
            inputs : A tensor of size [batch, height_in, width_in, channels]
    """"
    def __init__(self, filters, kernel_size, strides):
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling)
        self.strides = strides

    def call(self, inputs):
        if self.strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)
        return self.conv2d(inputs)
    
    def fixed_padding(inputs, kernel_size):
        """
            Fixed zero padding to be used with conv2d or max_pool2d.

            Args:
                inputs : A tensor of size [batch, height_in, width_in, channels]
                kernel_size: The kernel to be used in the conv2d or max_pool2d operation
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end],
                                                        [pad_beg, pad_end], [0, 0]])


class building_block(tf.keras.Model):
    """
        Resnet v1/v2 building block without bottleneck.
        2 x [Convolution x Batch normalization x Relu]

        Args:
            inputs: A tensor of size [batch, height_in, width_in, channels]
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
            version: Either "v1" or "v2"
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON, filters, training, projection_shortcut, strides, version="v1"):
        super(building_block_v1, self).__init__()
        self.BATCH_NORM_DECAY = BATCH_NORM_DECAY
        self.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON
        self.training = training
        self.shortcut = None
        self.version = version
        if projection_shortcut is not None:
            self.shortcut = projection_shortcut
            self.batch_norm_shortcut = self.batch_norm()

        self.conv2d_1 = conv2d_fixed_padding(filters, 3, strides)
        self.batch_norm_1 = self.batch_norm()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv2d_2 = conv2d_fixed_padding(filters, 3, 1)
        self.batch_norm_2 = self.batch_norm()
        self.relu_2 = tf.keras.layers.ReLU()
    
    def call(self, inputs):
        if self.version == "v1":
            return self.version1(inputs)
        elif self.version == "v2":
            return self.version2(inputs)
        else:
            NotImplementedError()
    
    def version1(self, inputs):
        if self.shortcut is None:
            self.shortcut = inputs
        outputs = self.conv2d_1(inputs)
        outputs = self.batch_norm_1(outputs)
        outputs = self.relu_1(outputs)
        outputs = self.conv2d_2(outputs)
        outputs = self.batch_norm_2(outputs)
        outputs += self.shortcut
        outputs = self.relu_2(outputs)
        return outputs
    
    def version2(self, inputs):
        if self.shortcut is None:
            self.shortcut = inputs
        outputs = self.batch_norm_1(outputs)
        outputs = self.relu_1(outputs)
        outputs = self.conv2d_1(inputs)
        outputs = self.batch_norm_2(outputs)
        outputs = self.relu_2(outputs)
        outputs = self.conv2d_2(outputs)
        outputs += self.shortcut
        return outputs

    def batch_norm(self):
        return tf.keras.layers.BatchNormalization(axis=3, momentum=self.BATCH_NORM_DECAY,
                                                  epsilon=self.BATCH_NORM_EPSILON, center=True,
                                                  scale=True, training=self.training, fused=True)


class bottleneck_block(tf.keras.Model):
    """
        Resnet v1/v2 bottleneck block.
        Similar to _building_block_v1(), except using the "bottleneck" blocks described in:
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
            inputs: A tensor of size [batch, height_in, width_in, channels]
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately downsample the input.
            version: Either "v1" or "v2"
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON, filters, training, projection_shortcut, strides, version="v1"):
        super(building_block_v1, self).__init__()
        self.BATCH_NORM_DECAY = BATCH_NORM_DECAY
        self.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON
        self.training = training
        self.shortcut = None
        self.version = version
        if projection_shortcut is not None:
            self.shortcut = projection_shortcut
            self.batch_norm_shortcut = self.batch_norm()

        self.conv2d_1 = conv2d_fixed_padding(filters, 1, 1)
        self.batch_norm_1 = self.batch_norm()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv2d_2 = conv2d_fixed_padding(filters, 3, strides)
        self.batch_norm_2 = self.batch_norm()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv2d_3 = conv2d_fixed_padding(4 * filters, 1, 1)
        self.batch_norm_3 = self.batch_norm()
        self.relu_3 = tf.keras.layers.ReLU()
    
    def call(self, inputs):
        if self.version == "v1":
            return self.version1(inputs)
        elif self.version == "v2":
            return self.version2(inputs)
        else:
            NotImplementedError()
    
    def version1(self, inputs):
        if self.shortcut is None:
            self.shortcut = inputs
        outputs = self.conv2d_1(inputs)
        outputs = self.batch_norm_1(outputs)
        outputs = self.relu_1(outputs)
        outputs = self.conv2d_2(outputs)
        outputs = self.batch_norm_2(outputs)
        outputs = self.relu_2(outputs)
        outputs = self.conv2d_3(outputs)
        outputs = self.batch_norm_3(outputs)
        outputs += self.shortcut
        outputs = self.relu_3(outputs)
        return outputs
    
    def version2(self, inputs):
        if self.shortcut is None:
            self.shortcut = inputs
        outputs = self.batch_norm_1(outputs)
        outputs = self.relu_1(outputs)
        outputs = self.conv2d_1(inputs)
        outputs = self.batch_norm_2(outputs)
        outputs = self.relu_2(outputs)
        outputs = self.conv2d_2(outputs)
        outputs = self.batch_norm_3(outputs)
        outputs = self.relu_3(outputs)
        outputs = self.conv2d_3(outputs)
        outputs += self.shortcut
        return outputs

    def batch_norm(self):
        return tf.keras.layers.BatchNormalization(axis=3, momentum=self.BATCH_NORM_DECAY,
                                                  epsilon=self.BATCH_NORM_EPSILON, center=True,
                                                  scale=True, training=self.training, fused=True)


class Resnet(tf.keras.Model):
    """
        This class contains the whole Resnet.
        Residual networks ('v1' ResNets) were originally proposed in:
        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
        The full preactivation 'v2' ResNet variant was introduced by:
        [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
        The key difference of the full preactivation 'v2' variant compared to the
        'v1' variant in [1] is the use of batch normalization before every weight layer
        rather than after.

        Args:
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON):
        super(Resnet, self).__init__(name='')
        self.BATCH_NORM_DECAY = BATCH_NORM_DECAY
        self.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON
    

