import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling1D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Layer,
    Multiply,
    Reshape
)

# --- Squeeze-and-Excitation Model Components ---

def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, channels))(se)
    return Multiply()([input_tensor, se])

def cnn_squeeze_excite(filters, se_ratio, dropout_rate, classes=256, input_dim=701, batch_normalize=False):
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 1 & 2
    x = Conv1D(filters[0], 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    if batch_normalize: x = BatchNormalization()(x)
    x = Conv1D(filters[1], 11, activation='relu', padding='same', name='block2_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = se_block(x, ratio=se_ratio)
    x = AveragePooling1D(2, strides=2)(x)
    # Block 3, 4, & 5
    x = Conv1D(filters[2], 11, activation='relu', padding='same', name='block3_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = Conv1D(filters[3], 11, activation='relu', padding='same', name='block4_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = Conv1D(filters[4], 11, activation='relu', padding='same', name='block5_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = se_block(x, ratio=se_ratio)
    x = AveragePooling1D(2, strides=2)(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=x, name='cnn_squeeze_excitation')
    return model

# --- CBAM Model Components ---

class ChannelwisePool(Layer):
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return Concatenate(axis=-1)([avg_pool, max_pool])
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)

def cbam_block(input_tensor, ratio=16):
    channel = input_tensor.shape[-1]
    # Channel Attention
    avg_pool_ch = GlobalAveragePooling1D()(input_tensor)
    max_pool_ch = GlobalMaxPooling1D()(input_tensor)
    shared_dense_1 = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)
    mlp_avg = shared_dense_2(shared_dense_1(avg_pool_ch))
    mlp_max = shared_dense_2(shared_dense_1(max_pool_ch))
    channel_attention = Activation('sigmoid')(Add()([mlp_avg, mlp_max]))
    channel_attention = Reshape((1, channel))(channel_attention)
    x = Multiply()([input_tensor, channel_attention])
    # Spatial Attention (robust version)
    concat = ChannelwisePool()(x)
    spatial_attention = Conv1D(filters=1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)
    cbam_out = Multiply()([x, spatial_attention])
    return cbam_out

def vgg_cnn_cbam(filters, cbam_ratio, dropout_rate, optimizer, classes=256, input_dim=701, batch_normalize=False):
    input_shape = (input_dim, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(filters[0], 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    if batch_normalize: x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool1')(x)
    # Block 2
    x = Conv1D(filters[1], 11, activation='relu', padding='same', name='block2_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = cbam_block(x, ratio=cbam_ratio)
    x = AveragePooling1D(2, strides=2, name='block1_pool2')(x)
    # Block 3
    x = Conv1D(filters[2], 11, activation='relu', padding='same', name='block3_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool3')(x)
    # Block 4
    x = Conv1D(filters[3], 11, activation='relu', padding='same', name='block4_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool4')(x)
    # Block 5
    x = Conv1D(filters[4], 11, activation='relu', padding='same', name='block5_conv1')(x)
    if batch_normalize: x = BatchNormalization()(x)
    x = cbam_block(x, ratio=cbam_ratio)
    x = AveragePooling1D(2, strides=2, name='block1_pool5')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=x, name='cbam_model')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class GaussianAttention(tf.keras.layers.Layer):
    def __init__(self, seq_len, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_heads = n_heads

    def build(self, input_shape):
      # μ initialization
      init_mu = tf.keras.initializers.RandomUniform(minval=100.0, maxval=self.seq_len)
      self.mu = self.add_weight(
          shape=(self.n_heads,),
          initializer=init_mu,
          trainable=True,
          name="mu"
      )

      # σ initialization (wider spread)
      init_sigma = tf.keras.initializers.Constant(value=30.0)
      self.sigma = self.add_weight(
          shape=(self.n_heads,),
          initializer=init_sigma,
          trainable=True,
          name="sigma"
      )

      # Static projection to attention heads
      self.proj = tf.keras.layers.Dense(self.n_heads)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        T = self.seq_len
        C = tf.shape(inputs)[2]

        # Time indices
        time = tf.range(0, T, dtype=tf.float32)
        time = tf.reshape(time, (1, T, 1))  # (1, T, 1)

        mu = tf.reshape(self.mu, (1, 1, self.n_heads))
        sigma = tf.reshape(self.sigma, (1, 1, self.n_heads))

        weights = tf.exp(-0.5 * ((time - mu) / (sigma + 1e-6))**2)
        weights /= tf.reduce_sum(weights, axis=1, keepdims=True)

        # Project input to H heads
        proj = self.proj(inputs)  # (B, T, H)

        # Attention-weighted sum
        attended = tf.reduce_sum(proj * weights, axis=1)  # (B, H)
        return attended
        
from tensorflow.keras.layers import Input, Conv1D, ReLU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_estranet(seq_len=700, num_classes=256, n_heads=8, conv_filters=32):
    # 4 from EstraClone.ipynb
    inputs = Input(shape=(seq_len, 1))

    # 1st Conv layer
    x = Conv1D(32, kernel_size=11, padding='same')(inputs)
    x = ReLU()(x)

    # 2nd Conv layer
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = ReLU()(x)

    # 3rd Conv layer
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = ReLU()(x)

    # 2. Gaussian attention block
    attn = GaussianAttention(seq_len=seq_len, n_heads=n_heads)(x)

    x = attn  # output from GaussianAttention

    x = Dense(128)(attn)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(256)(x)

    model = Model(inputs, outputs, name="EstraNet")
    return model
    
def GaussianTransformerBlock(x_in, seq_len, n_heads, ffn_dim=128, dropout_rate=0.3, name_prefix="gblock"):
    attn_out = GaussianAttention(seq_len=seq_len, n_heads=n_heads, name=f"{name_prefix}_attn")(x_in)

    if x_in.shape[-1] != attn_out.shape[-1]:
        x_proj = Dense(attn_out.shape[-1])(x_in)
    else:
        x_proj = x_in

    x = Add()([x_proj, attn_out])
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    ffn = Dense(ffn_dim)(x)
    ffn = ReLU()(ffn)
    ffn = Dense(attn_out.shape[-1])(ffn)

    x = Add()([x, ffn])
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    return x
    
## Stack Gaussi attentions

from tensorflow.keras.layers import Input, Conv1D, ReLU, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_estranet_stackable(seq_len=700, num_classes=256, n_heads=8, conv_filters=32, gaussip_attn_layers=1):
    inputs = Input(shape=(seq_len, 1))

    # 1st Conv layer
    x = Conv1D(32, kernel_size=11, padding='same')(inputs)
    x = ReLU()(x)

    # 2nd Conv layer
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = ReLU()(x)

    # 3rd Conv layer
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = ReLU()(x)

    # 2. Gaussian attention block
    for i in range(gaussip_attn_layers):
      x = GaussianTransformerBlock(x, seq_len=seq_len, n_heads=n_heads, ffn_dim=128, name_prefix=f"GaussiTrfmrBlock{i}")

    x = GlobalAveragePooling1D()(x)

    # x = Dense(128)(attn)
    # x = ReLU()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(256)(x)

    model = Model(inputs, outputs, name="EstraNet")
    return model
