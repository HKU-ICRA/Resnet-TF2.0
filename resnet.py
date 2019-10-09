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
            batch_norm: The batch normalization function
            version: Either "v1" or "v2"
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON, filters, training, projection_shortcut, strides, batch_norm, version="v1"):
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
            batch_norm: The batch normalization function
            version: Either "v1" or "v2"
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON, filters, training, projection_shortcut, strides, batch_norm, version="v1"):
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


class block_layer(tf.keras.Model):
    """
        Creates one layer of blocks for the ResNet model.

        Args:
            inputs: A tensor of size [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the first convolution of the layer.
            bottleneck: Is the block created a bottleneck block.
            block_fn: The block to use within the model, either `building_block` or `bottleneck_block`.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.
            batch_norm: The batch normalization function

        Returns:
            The output tensor of the block layer.
    """
    def __init__(self, filters, bottleneck, block_fn, blocks, strides, training, name, batch_norm):
        self.filters_out = filters * 4 if bottleneck else filters
        self.blocks = blocks
        self.strides = strides
        self.training = training
        self.name = name
        self.batch_norm = batch_norm
        # Only the first block per block_layer uses projection_shortcut and strides
        self.block = block_fn(filters, training, projection_shortcut, strides, batch_norm=self.batch_norm)
        self.all_blocks = []
        for _ in range(1, blocks):
            self.all_blocks.append(block_fn(filters, training, None, 1, batch_norm=self.batch_norm))
    
    def call(self, inputs):
        outputs = self.block(inputs)
        for current_block in self.blocks:
            outputs = current_block(outputs)
        return tf.identity(outputs, name=self.name)
    
    def projection_shortcut(self):
        return conv2d_fixed_padding(filters=self.filters_out, kernel_size=1, strides=self.strides)


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
            resnet_size: A single integer for the size of the ResNet model.
            bottleneck: Use regular blocks or bottleneck blocks.
            num_classes: The number of classes used as labels.
            num_filters: The number of filters to use for the first block layerof the model.
                         This number is then doubled for each subsequent block
                         layer.
            kernel_size: The kernel size to use for convolution.
            conv_stride: stride size for the initial convolutional layer
            first_pool_size: Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer. Not used if first_pool_size is None.
            block_sizes: A list containing n values, where n is the number of sets of block layers desired.
                         Each value should be the number of blocks in the i-th set.
            block_strides: List of integers representing the desired stride size for each of the sets of block layers.
                           Should be same length as block_sizes.
            resnet_version: Integer representing which version of the ResNet network to use. See README for details. Valid values: ["v1", "v2"]
            dtype: The TensorFlow dtype to use for calculations. If not specified tf.float32 is used.
        Raises:
            ValueError: if invalid version is selected.
    """
    def __init__(self, BATCH_NORM_DECAY, BATCH_NORM_EPSILON, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size, conv_stride, first_pool_size, first_pool_stride, block_sizes, block_strides, training
               resnet_version="v1", dtype=tf.float32):
        super(Resnet, self).__init__(name='')
        self.BATCH_NORM_DECAY = BATCH_NORM_DECAY
        self.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON
        self.resnet_size = resnet_size

        self.resnet_version = resnet_version
        if resnet_version not in ("v1", "v2"):
            raise ValueError('Resnet version should be v1 or v2.')
        self.bottleneck = bottleneck
        if bottleneck:
            self.block_fn = bottleneck_block
        else:
            self.block_fn = building_block

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.training = training
        self.dtype = dtype
        self.pre_activation = resnet_version == "v2"
    
    def call(self, inputs):
        """
            Add operations to classify a batch of input images.

            Args:
                inputs: A Tensor representing a batch of input images.

            Returns:
                A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        with tf.name_scope("resnet_model"):
            initial_conv = conv2d_fixed_padding(self.num_filters, self.kernel_size, self.conv_stride)
            inputs = initial_conv(inputs)
            inputs = tf.identity(inputs, 'initial_conv')

            # We do not include batch normalization or activation functions in V2
            # for the initial conv1 because the first ResNet unit will perform these
            # for both the shortcut and non-shortcut paths as part of the first
            # block's projection. Cf. Appendix of [2].

            if self.resnet_version == "v1":
                initial_batch_norm = batch_norm()
                initial_relu = tf.keras.layers.ReLU()
                inputs = initial_batch_norm(inputs)
                inputs = initial_relu(inputs)
            
            if self.first_pool_size:
                initial_maxpool2d = tf.keras.layers.MaxPool2D(pool_size=self.first_pool_size, strides=self.first_pool_stride, padding='SAME')
                inputs = initial_maxpool2d(inputs)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                block_layer = block_layer(filters=num_filters, bottleneck=self.bottleneck, block_fn=self.block_fn,
                                          blocks=num_blocks, strides=self.block_strides[i], training=training,
                                          name='block_layer{}'.format(i + 1), =self.data_format,
                                          batch_norm=self.batch_norm)
                inputs = block_layer(inputs)
            
            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                preact_batch_norm = batch_norm()
                preact_relu = tf.keras.layers.ReLU()
                inputs = preact_batch_norm(inputs)
                inputs = preact_relu(inputs)

        return inputs
    
    def batch_norm(self):
        return tf.keras.layers.BatchNormalization(axis=3, momentum=self.BATCH_NORM_DECAY,
                                                  epsilon=self.BATCH_NORM_EPSILON, center=True,
                                                  scale=True, training=self.training, fused=True)
