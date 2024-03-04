import math
import tensorflow as tf
from data_utils.data import Featurizer

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


__all__ = [
    'dice',
    'dice_loss',
    'ovl',
    'ovl_loss',
    'CATransUnet',
]


def dice(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    keras layers.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smoothing_factor)
            / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))


def dice_loss(y_true, y_pred):
    """Keras loss function for Dice coefficient (loss(t, y) = -dice(t, y))"""
    return 1 - dice(y_true, y_pred)


def ovl(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with keras layers"""
    concat = K.concatenate((y_true, y_pred))
    return ((K.sum(K.min(concat, axis=-1)) + smoothing_factor)
            / (K.sum(K.max(concat, axis=-1)) + smoothing_factor))


def ovl_loss(y_true, y_pred):
    """Keras loss function for overlap coefficient (loss(t, y) = -ovl(t, y))"""
    return 1 - ovl(y_true, y_pred)


class SingleConv3DBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, activation=None, use_bias=True):
        super(SingleConv3DBlock, self).__init__()
        self.kernel = kernel_size
        self.block = tf.keras.layers.Conv3D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            padding='same',
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_regularizer=l2(1e-5))

    def call(self, inputs, **kwargs):
        return self.block(inputs)


class Conv3DBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3, 3), use_bias=True):
        super(Conv3DBlock, self).__init__()
        self.a = tf.keras.Sequential([
            SingleConv3DBlock(filters, kernel_size=kernel_size, use_bias=use_bias),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def call(self, inputs, **kwargs):
        return self.a(inputs)


class IdentityBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3, 3, 3)):
        super(IdentityBlock, self).__init__()
        self.a = Conv3DBlock(filters, kernel_size)
        self.b = tf.keras.Sequential([
            SingleConv3DBlock(filters, kernel_size=kernel_size),
            tf.keras.layers.BatchNormalization(),
        ])

    def call(self, inputs, **kwargs):
        x = self.a(inputs)
        x = self.b(x)
        out = Add()([x, inputs])
        out = Activation('relu')(out)
        return out


class SelfAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, embed_dim, dropout, vis=False):
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(self.all_head_size)
        self.key = tf.keras.layers.Dense(self.all_head_size)
        self.value = tf.keras.layers.Dense(self.all_head_size)

        self.out = tf.keras.layers.Dense(embed_dim)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.proj_dropout = tf.keras.layers.Dropout(dropout)

        self.softmax = tf.keras.layers.Softmax()

        self.vis = vis

    def transpose_for_scores(self, x):
        new_x_shape = list(x.get_shape()[:-1] + (self.num_attention_heads, self.attention_head_size))
        new_x_shape[0] = -1
        y = tf.reshape(x, new_x_shape)
        return tf.transpose(y, perm=[0, 2, 1, 3])

    def call(self, hidden_states, **kwargs):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = query_layer @ tf.transpose(key_layer, perm=[0, 1, 3, 2])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = attention_probs @ value_layer
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        new_context_layer_shape = list(context_layer.shape[:-2] + (self.all_head_size,))
        new_context_layer_shape[0] = -1
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class Mlp(tf.keras.layers.Layer):

    def __init__(self, output_features, drop=0.):
        super(Mlp, self).__init__()
        self.a = tf.keras.layers.Dense(units=output_features, activation=tf.nn.relu)
        self.b = tf.keras.layers.Dropout(drop)

    def call(self, inputs, **kwargs):
        x = self.a(inputs)
        return self.b(x)


class PositionwiseFeedForward(tf.keras.layers.Layer):

    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.a = tf.keras.layers.Dense(units=d_ff)
        self.b = tf.keras.layers.Dense(units=d_model)
        self.c = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, **kwargs):
        return self.b(self.c(tf.nn.relu(self.a(inputs))))


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, cube_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_of_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.positionalEmbedding = tf.keras.layers.Embedding(self.num_of_patches, embed_dim)
        self.patches = None
        self.lyer = tf.keras.layers.Conv3D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size,
                                           padding='valid')

    def call(self, inputs, **kwargs):
        patches = self.lyer(inputs)
        patches = tf.reshape(patches, (-1, self.num_of_patches, patches.shape[-1]))
        positions = tf.range(0, self.num_of_patches, 1)[tf.newaxis, ...]
        positionalEmbedding = self.positionalEmbedding(positions)
        patches = patches + positionalEmbedding

        return patches, positionalEmbedding


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout):
        super(TransformerLayer, self).__init__()

        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mlp = PositionwiseFeedForward(embed_dim, 1024)
        self.attn = SelfAttention(num_heads, embed_dim, dropout, vis=True)

    def call(self, x, training=True):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h

        return x


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, c_dim, s_factor, num_heads, dropout):
        super(ChannelAttention, self).__init__()
        self.g_avg = GlobalAveragePooling3D()
        self.g_max = GlobalMaxPooling3D()
        self.conv_s = Conv3DBlock(c_dim // s_factor, kernel_size=1, use_bias=False)
        self.squeeze = TransformerLayer(c_dim // s_factor, num_heads, dropout)
        self.conv_e = Conv3DBlock(c_dim, kernel_size=1, use_bias=False)
        self.excitation = TransformerLayer(c_dim, num_heads, dropout)

    def call(self, x, training=True):

        g_a = self.g_avg(x)
        g_m = self.g_max(x)
        ma = Add()([g_m, g_a])
        ma = Reshape((1, 1, 1, ma.shape[-1]))(ma)
        ma = self.conv_s(ma)
        ma = Reshape((1, ma.shape[-1]))(ma)
        s = self.squeeze(ma)
        s = Activation('relu')(s)
        s = Reshape((1, 1, 1, s.shape[-1]))(s)
        s = self.conv_e(s)
        s = Reshape((1, s.shape[-1]))(s)
        e = self.excitation(s)
        e = Activation('sigmoid')(e)
        e = Reshape((1, 1, 1, e.shape[-1]))(e)
        ch_att = Multiply()([x, e])
        out = Add()([x, ch_att])
        out = Activation('relu')(out)

        return out


class UnetEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3), pool_size=[2, 2, 3, 3]):
        super(UnetEncoder, self).__init__()
        self.conv0 = tf.keras.Sequential([
            Conv3DBlock(filters, kernel_size),
            Conv3DBlock(filters, kernel_size)]
        )
        self.pool0 = MaxPooling3D(pool_size=pool_size[0])

        self.conv1 = tf.keras.Sequential([
            Conv3DBlock(filters * 2, kernel_size),
            Conv3DBlock(filters * 2, kernel_size)]
        )
        self.pool1 = MaxPooling3D(pool_size=pool_size[1])

        self.conv2 = tf.keras.Sequential([
            Conv3DBlock(filters * 4, kernel_size),
            Conv3DBlock(filters * 4, kernel_size)]
        )
        self.pool2 = MaxPooling3D(pool_size=pool_size[2])

        self.conv3 = tf.keras.Sequential([
            Conv3DBlock(filters * 8, kernel_size),
            Conv3DBlock(filters * 8, kernel_size)]
        )
        self.pool3 = MaxPooling3D(pool_size=pool_size[3])

    def call(self, inputs, training=True):
        c0 = self.conv0(inputs)
        p0 = self.pool0(c0)

        c1 = self.conv1(p0)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        return c0, c1, c2, c3, p3


class CATransUnet(Model):

    DEFAULT_SIZE = 36

    def __init__(self, max_dist=35, featurizer=Featurizer(save_molecule_codes=False), scale=0.5,
                 box_size=36, input_channels=18, output_channels=1, l2_lambda=1e-3, **kwargs):
        """Creates a new network."""

        self.featurizer = featurizer
        self.scale = scale
        self.max_dist = max_dist

        n_filters = 24
        embed_dim = n_filters * 8
        num_heads = 8
        num_layers = 2
        s_factor = 6

        unet_encoder = UnetEncoder(n_filters)
        transformer_layers = [TransformerLayer(embed_dim, num_heads, 0.1) for _ in range(num_layers)]

        inputs = Input((box_size, box_size, box_size, input_channels), name='input')  # 36 * 36 * 36 * 18

        c0, c1, c2, c3, p3 = unet_encoder(inputs)
        embeddings = tf.reshape(p3, (-1, 1, p3.shape[-1]))
        for i in range(num_layers):
            embeddings = transformer_layers[i](embeddings)
        center = tf.reshape(embeddings, (-1, 1, 1, 1, embeddings.shape[-1]))
        center = Conv3DBlock(n_filters * 16, 3)(center)

        c3 = ChannelAttention(n_filters * 8, s_factor, num_heads, 0.1)(c3)
        up1 = concatenate([UpSampling3D(size=3)(center), c3], axis=4)
        conv1 = Conv3DBlock(n_filters * 8, 3)(up1)
        conv1 = Conv3DBlock(n_filters * 8, 3)(conv1)

        c2 = ChannelAttention(n_filters * 4, s_factor, num_heads, 0.1)(c2)
        up2 = concatenate([UpSampling3D(size=3)(conv1), c2], axis=4)
        conv2 = Conv3DBlock(n_filters * 4, 3)(up2)
        conv2 = Conv3DBlock(n_filters * 4, 3)(conv2)

        # RBF block is implement with IdentityBlock
        c1 = IdentityBlock(n_filters * 2, 6)(c1)
        up3 = concatenate([UpSampling3D(size=2)(conv2), c1], axis=4)
        conv3 = Conv3DBlock(n_filters * 2, 3)(up3)
        conv3 = Conv3DBlock(n_filters * 2, 3)(conv3)

        # RBF block is implement with IdentityBlock
        c0 = IdentityBlock(n_filters, 6)(c0)
        up4 = concatenate([UpSampling3D(size=2)(conv3), c0], axis=4)
        conv4 = Conv3DBlock(n_filters, 3)(up4)
        conv4 = Conv3DBlock(n_filters, 3)(conv4)

        outputs = Convolution3D(
            filters=output_channels,
            kernel_size=1,
            activation='sigmoid',
            kernel_regularizer=l2(l2_lambda),
            name='pocket'
        )(conv4)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    @staticmethod
    def __total_shape(tensor_list):
        if len(tensor_list) == 1:
            total_shape = tuple(tensor_list[0].shape.as_list())
        else:
            total_shape = (*tensor_list[0].shape.as_list()[:-1],
                           sum(t.shape.as_list()[-1] for t in tensor_list))
        return total_shape

    def save_keras(self, path):
        class_name = self.__class__.__name__
        self.__class__.__name__ = 'Model'
        self.save(path, include_optimizer=False)
        self.__class__.__name__ = class_name

    @staticmethod
    def load_model(path, **attrs):
        """Load model"""
        from tensorflow.keras.models import load_model as keras_load
        custom_objects = {name: val for name, val in globals().items()
                          if name in __all__}
        model = keras_load(path, custom_objects=custom_objects)

        for attr, value in attrs.items():
            setattr(model, attr, value)
        return model
