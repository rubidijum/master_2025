from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling1D, 
                                     Input, 
                                     GlobalAveragePooling1D, 
                                     Dense, 
                                     Reshape, 
                                     Multiply, 
                                     Dropout, 
                                     Conv1D, 
                                     BatchNormalization, 
                                     MaxPooling1D, 
                                     Flatten, 
                                     Add, 
                                     Activation, 
                                     Concatenate, 
                                     GlobalMaxPooling1D, 
                                     Lambda)

# Define the VGG inspired CNN model with the additional Squeeze-and-Excitation (SE) layer

def se_block(input_tensor, ratio=16):
  channels = input_tensor.shape[-1]
  se = GlobalAveragePooling1D()(input_tensor)
  se = Dense(channels // ratio, activation='relu')(se)
  se = Dense(channels, activation='sigmoid')(se)
  se = Reshape((1, channels))(se)
  return Multiply()([input_tensor, se])

def cnn_squeeze_excite(filters, se_ratio, dropout_rate, classes=256,input_dim=700, batch_normalize=False):
    # From VGG16 design
    input_shape = (input_dim,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(filters[0], 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    if batch_normalize:
        x = BatchNormalization()(x)
    # Block 2
    x = Conv1D(filters[1], 11, activation='relu', padding='same', name='block2_conv1')(x)
    if batch_normalize:
        x = BatchNormalization()(x)
    # Squeeze excitation block 1
    x = se_block(x, ratio=se_ratio)
    x = MaxPooling1D(2)(x)

    # Block 3
    x = Conv1D(filters[2], 11, activation='relu', padding='same', name='block3_conv1')(x)
    if batch_normalize:
        x = BatchNormalization()(x)
    # Block 4
    x = Conv1D(filters[3], 11, activation='relu', padding='same', name='block4_conv1')(x)
    if batch_normalize:
        x = BatchNormalization()(x)
  # Block 5
    x = Conv1D(filters[4], 11, activation='relu', padding='same', name='block5_conv1')(x)
    if batch_normalize:
        x = BatchNormalization()(x)
    # Squeeze excitation block 1
    x = se_block(x, ratio=se_ratio)
    x = MaxPooling1D(2)(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_squeeze_excitation')

    return model

# Define the VGG inspired CNN model with the additional Convolutional Block Module Attention (CBAM) layer

def cbam_block(input_tensor, ratio=16, name_prefix="cbam"):
  channel = input_tensor.shape[-1]

  avg_pool = GlobalAveragePooling1D()(input_tensor)
  max_pool = GlobalMaxPooling1D()(input_tensor)

  shared_dense_1 = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
  shared_dense_2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)

  mlp_avg = shared_dense_2(shared_dense_1(avg_pool))
  mlp_max = shared_dense_2(shared_dense_1(max_pool))

  channel_attention = Activation('sigmoid')(Add()([mlp_avg, mlp_max]))
  channel_attention = Reshape((1, channel))(channel_attention)

  x = Multiply()([input_tensor, channel_attention])

  avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
  max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
  # avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
  # max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
  # avg_pool = Lambda(lambda z: K.mean(z, axis=-1, keepdims=True))(x)
  # max_pool = Lambda(lambda z: K.max(z, axis=-1, keepdims=True))(x)
  concat = Concatenate(axis=-1)([avg_pool, max_pool])

  spatial_attention = Conv1D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

  cbam_out = Multiply()([x, spatial_attention])
  return cbam_out

def vgg_cnn_cbam(filters, cbam_ratio, dropout_rate, optimizer, classes=256,input_dim=701, batch_normalize=False):
	# From VGG16 design
	input_shape = (input_dim,1)
	img_input = Input(shape=input_shape)
	# Block 1
	x = Conv1D(filters[0], 11, activation='relu', padding='same', name='block1_conv1')(img_input)
	if batch_normalize:
		x = BatchNormalization()(x)
	x = AveragePooling1D(2, strides=2, name='block1_pool1')(x)
	# Block 2
	x = Conv1D(filters[1], 11, activation='relu', padding='same', name='block2_conv1')(x)
	if batch_normalize:
		x = BatchNormalization()(x)
	# Squeeze excitation block 1
	x = cbam_block(x, ratio=cbam_ratio)
	x = AveragePooling1D(2, strides=2, name='block1_pool2')(x)

	# Block 3
	x = Conv1D(filters[2], 11, activation='relu', padding='same', name='block3_conv1')(x)
	if batch_normalize:
		x = BatchNormalization()(x)
	x = AveragePooling1D(2, strides=2, name='block1_pool3')(x)
	# Block 4
	x = Conv1D(filters[3], 11, activation='relu', padding='same', name='block4_conv1')(x)
	if batch_normalize:
		x = BatchNormalization()(x)
	x = AveragePooling1D(2, strides=2, name='block1_pool4')(x)
  # Block 5
	x = Conv1D(filters[4], 11, activation='relu', padding='same', name='block5_conv1')(x)
	if batch_normalize:
		x = BatchNormalization()(x)
 	# Squeeze excitation block 1
	x = cbam_block(x, ratio=cbam_ratio)
	x = AveragePooling1D(2, strides=2, name='block1_pool5')(x)

	# Classification block
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dropout(dropout_rate)(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dropout(dropout_rate)(x)
	x = Dense(classes, activation='softmax', name='predictions')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='cbam_model')

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model