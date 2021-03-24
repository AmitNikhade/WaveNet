


import tensorflow
from tensorflow.keras.layers import *
from tensorflow.python.keras.activations import sigmoid, tanh
from tensorflow.keras import Model




class residuals(tensorflow.keras.layers.Layer):
    def __init__(self,num_filters, kernel_size, dilation_rate):
        super(residuals, self).__init__()
        # self.input_size = input_size
        self.num_filters = num_filters
        self.filter_size = kernel_size
        self.dilation_rate = dilation_rate
        self.tanh_ = Conv1D(filters=self.num_filters, kernel_size=self.filter_size, dilation_rate=self.dilation_rate,
                        padding='causal', activation='tanh')
        self.sigmoid_ = Conv1D(filters=self.num_filters, kernel_size=self.filter_size
                               , dilation_rate=self.dilation_rate, padding='causal', activation='sigmoid')
        self.mul = Multiply()
        self.skip = Conv1D(1, 1, activation='relu', padding="same")
        self.add = Add()
        self.ap = AveragePooling1D(10)
        self.bn = BatchNormalization()
        self.sd  = SpatialDropout1D(0.05)

    def call(self,x):
 
        self.t = self.tanh_(x)
        self.s = self.sigmoid_(x)
        self.m = self.mul([self.t, self.s])
        skp = self.skip(self.m)
        add = self.add([skp, x])
        ap = self.ap(add)
        bn = self.bn(ap)
        sd = self.sd(bn)
        return sd, skp


class skip_connect():
    def __init__(self, input_size):
        self.net1 = Add()
        self.net2 = Activation('relu')
        self.net3 = Conv1D(1, 1, activation='relu')
        self.net4 = Conv1D(1, 1)
        self.net5 = Flatten()
        self.net6 = Dense(input_size, activation='softmax')
    def call(self,x , skips):
        n1 = self.net1(skips)
        n2 = self.net2(n1)
        n3 = self.net3(n2)
        n4 = self.net4(n3)
        n5 = self.net5(n4)
        n6 = self.net6(n5)
        return Model(x, n6)


class WaveNet(tensorflow.keras.layers.Layer):
    def __init__(self, input_size, num_layers, kernel_size, dilation_rate, num_filters):
        super(WaveNet, self).__init__()
          
        self.input_size = input_size  
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_filters = num_filters
        self.x = Input(shape=(input_size, 1))
        
    def model(self):
        output,_ = residuals(self.num_filters, self.kernel_size, self.dilation_rate).call(self.x)

        skips = []
        for layers in range(self.num_layers):
            _,skip = residuals(self.num_filters, self.kernel_size, self.dilation_rate).call(output)
            skips.append(skip)
        return skip_connect(self.input_size).call(self.x, skips)
    