
import numpy as np
from keras.models import Model,load_model
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout,GaussianNoise, Input,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import  Conv2DTranspose,UpSampling2D,concatenate,add
from keras.optimizers import SGD
import keras.backend as K
from losses import *

#this is for the losses
import keras.backend as K



#this is for the end of the losses

K.set_image_data_format("channels_last")

 #u-net model
class Unet_model(object):
    
    def __init__(self,img_shape,load_model_weights=None):
        self.img_shape=img_shape
        self.load_model_weights=load_model_weights
        self.model =self.compile_unet()

    def dice(self,y_true, y_pred):
    #computes the dice score on two tensors

        sum_p=K.sum(y_pred,axis=0)
        sum_r=K.sum(y_true,axis=0)
        sum_pr=K.sum(y_true * y_pred,axis=0)
        dice_numerator =2*sum_pr
        dice_denominator =sum_r+sum_p
        dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
        return dice_score


    def dice_whole_metric(self,y_true, y_pred):
    #computes the dice for the whole tumor

        y_true_f = K.reshape(y_true,shape=(-1,4))
        y_pred_f = K.reshape(y_pred,shape=(-1,4))
        y_whole=K.sum(y_true_f[:,1:],axis=1)
        p_whole=K.sum(y_pred_f[:,1:],axis=1)
        dice_whole=dice(y_whole,p_whole)
        return dice_whole

    def dice_en_metric(self,y_true, y_pred):
    #computes the dice for the enhancing region

        y_true_f = K.reshape(y_true,shape=(-1,4))
        y_pred_f = K.reshape(y_pred,shape=(-1,4))
        y_enh=y_true_f[:,-1]
        p_enh=y_pred_f[:,-1]
        dice_en=dice(y_enh,p_enh)
        return dice_en

    def dice_core_metric(self,y_true, y_pred):
    ##computes the dice for the core region

        y_true_f = K.reshape(y_true,shape=(-1,4))
        y_pred_f = K.reshape(y_pred,shape=(-1,4))
        
        #workaround for tf
        #y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
        #p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)
        
        y_core=K.sum(y_true_f[:,[1,3]],axis=1)
        p_core=K.sum(y_pred_f[:,[1,3]],axis=1)
        dice_core=dice(y_core,p_core)
        return dice_core



    def weighted_log_loss(self, y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # weights are assigned in this order : normal,necrotic,edema,enhancing 
        weights=np.array([1,5,2,4])
        weights = K.variable(weights)
        loss = y_true * K.log(y_pred) * weights
        loss = K.mean(-K.sum(loss, -1))
        return loss

    def gen_dice_loss(self, y_true, y_pred):
        '''
        computes the sum of two losses : generalised dice loss and weighted cross entropy
        '''

        #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
        y_true_f = K.reshape(y_true,shape=(-1,4))
        y_pred_f = K.reshape(y_pred,shape=(-1,4))
        sum_p=K.sum(y_pred_f,axis=-2)
        sum_r=K.sum(y_true_f,axis=-2)
        sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
        weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
        generalised_dice_numerator =2*K.sum(weights*sum_pr)
        generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
        generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
        GDL=1-generalised_dice_score
        del sum_p,sum_r,sum_pr,weights

        return GDL+weighted_log_loss(y_true,y_pred)
        
    
    def compile_unet(self):
        """
        compile the U-net model
        """
        i = Input(shape=self.img_shape)
        #add gaussian noise to the first layer to combat overfitting
        i_=GaussianNoise(0.01)(i)

        i_ = Conv2D(64, 2, padding='same',data_format = 'channels_last')(i_)
        out=self.unet(inputs=i_)
        model = Model(input=i, output=out)

        sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
        #1 - model.compile(loss=gen_dice_loss, optimizer=sgd, metrics=[dice_whole_metric,dice_core_metric,dice_en_metric])
        #2 - model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[dice_whole_metric,dice_core_metric,dice_en_metric]) 
        #model.compile(optimizer='rmsprop',loss='mse') - this works and returns no issue
        #metricsFM =[self.dice_whole_metric,self.dice_core_metric,self.dice_en_metric] -> throws can only concatenate list (not "int") to list
        #metricsFM = metricsFM =[dice_whole_metric,dice_core_metric,dice_en_metric]
        #this is the only metric that works - others give error for concatanation
        model.compile(loss=gen_dice_loss, optimizer=sgd, metrics=[self.dice_whole_metric])

        #load weights if set for prediction
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model


    def unet(self,inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2. ,activation='relu', dropout=0.0, batchnorm=True, upconv=True,format_='channels_last'):
        """
        the actual u-net architecture
        """
        o = self.level_block(inputs,start_ch, depth, inc_rate,activation, dropout, batchnorm, upconv,format_)
        o = BatchNormalization()(o) 
        #o =  Activation('relu')(o)
        o=PReLU(shared_axes=[1, 2])(o)
        o = Conv2D(nb_classes, 1, padding='same',data_format = format_)(o)
        o = Activation('softmax')(o)
        return o



    def level_block(self,m, dim, depth, inc, acti, do, bn, up,format_="channels_last"):
        if depth > 0:
            n = self.res_block_enc(m,0.0,dim,acti, bn,format_)
            #using strided 2D conv for donwsampling
            m = Conv2D(int(inc*dim), 2,strides=2, padding='same',data_format = format_)(n)
            m = self.level_block(m,int(inc*dim), depth-1, inc, acti, do, bn, up )
            if up:
                m = UpSampling2D(size=(2, 2),data_format = format_)(m)
                m = Conv2D(dim, 2, padding='same',data_format = format_)(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2,padding='same',data_format = format_)(m)
            n=concatenate([n,m])
            #the decoding path
            m = self.res_block_dec(n, 0.0,dim, acti, bn, format_)
        else:
            m = self.res_block_enc(m, 0.0,dim, acti, bn, format_)
        return m

  
   
    def res_block_enc(self,m, drpout,dim,acti, bn,format_="channels_last"):
        
        """
        the encoding unit which a residual block
        """
        n = BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format = format_)(n)
                
        n = BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)

        n=add([m,n]) 
        
        return  n 



    def res_block_dec(self,m, drpout,dim,acti, bn,format_="channels_last"):

        """
        the decoding unit which a residual block
        """
         
        n = BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format = format_)(n)

        n = BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n=PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same',data_format =format_ )(n)
        
        Save = Conv2D(dim, 1, padding='same',data_format = format_,use_bias=False)(m) 
        n=add([Save,n]) 
        
        return  n   



    
