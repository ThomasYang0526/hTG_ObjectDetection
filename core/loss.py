import tensorflow as tf
from tensorflow.keras import backend as K
from configuration import Config


class FocalLoss:
    def __call__(self, y_true, y_pred, *args, **kwargs):
        # return FocalLoss.__ce__loss(y_true, y_pred)
        return FocalLoss.__neg_loss(y_true, y_pred)
        # return FocalLoss.mse(y_true, y_pred)

    @staticmethod
    def mse(hm_true, hm_pred):
        pos_mask = tf.cast(tf.greater_equal(hm_true, 0.5), dtype=tf.float32)
        neg_mask = tf.cast(tf.less(hm_true, 0.5), dtype=tf.float32)
        neg_weights = tf.pow(1. - hm_true, 4)
        # neg_weights = tf.where(K.equal(neg_weights, 1.), 0.00000001, neg_weights)
   
        pos_loss = ((hm_true - hm_pred)**2) * tf.pow(1. - hm_pred, 2) * pos_mask
        neg_loss = ((hm_true - hm_pred)**2) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask

        # num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss) * 2. * 0.01
        neg_loss = tf.reduce_sum(neg_loss) * 1. * 0.01
        print('pos_loss: ', -pos_loss.numpy(), 'neg_loss: ', -neg_loss.numpy())
        
        loss = pos_loss + neg_loss
        loss = tf.reduce_mean(loss)
        
        return loss    
    
    @staticmethod
    def __neg_loss(hm_true, hm_pred):
        pos_mask = tf.cast(tf.equal(hm_true, 1.), dtype=tf.float32)
        neg_mask = tf.cast(tf.less(hm_true, 1.), dtype=tf.float32)
        neg_weights = tf.pow(1. - hm_true, 4)
        # neg_weights = tf.where(K.equal(neg_weights, 1.), 0.00000001, neg_weights)
   
        pos_loss = tf.math.log(tf.clip_by_value(hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(1. - hm_pred, 2) * pos_mask
        neg_loss = tf.math.log(tf.clip_by_value(1. - hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss) * 1.
        neg_loss = tf.reduce_sum(neg_loss) * 1.
        # print('pos_loss: ', -pos_loss.numpy(), 'neg_loss: ', -neg_loss.numpy())
        
        loss = 0
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        
        return loss        

    @staticmethod
    def __ce__loss(hm_true, hm_pred):
        loss = hm_true * -tf.math.log(tf.clip_by_value(hm_pred, 1e-8, 1. - 1e-8)) + \
               (1 - hm_true) * -tf.math.log(1 - tf.clip_by_value(hm_pred, 1e-8, 1. - 1e-8))
        loss = tf.reduce_mean(loss)
        return loss

class RegL1Loss:
    def __call__(self, y_true, y_pred, mask, index, *args, **kwargs):
        y_pred = RegL1Loss.gather_feat(y_pred, index)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), tf.constant([1, 1, 2], dtype=tf.int32))
        loss = tf.math.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        reg_loss = loss / (tf.math.reduce_sum(mask) + 1e-4)
        return reg_loss

    @staticmethod
    def gather_feat(feat, idx):
        feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[-1]))
        idx = tf.cast(idx, dtype=tf.int32)
        feat = tf.gather(params=feat, indices=idx, batch_dims=1)
        return feat

class RegL1Loss_joint_loc:
    def __call__(self, y_true, y_pred, mask, index, *args, **kwargs):
        y_pred = RegL1Loss_joint_loc.gather_feat(y_pred, index)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), tf.constant([1, 1, 2*Config.num_joints], dtype=tf.int32))
        loss = tf.math.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        # print(y_pred[0])
        reg_loss = loss / (tf.math.reduce_sum(mask) + 1e-4)
        return reg_loss

    @staticmethod
    def gather_feat(feat, idx):
        feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[-1]))
        idx = tf.cast(idx, dtype=tf.int32)
        feat = tf.gather(params=feat, indices=idx, batch_dims=1)
        return feat

class CombinedLoss:
    def __init__(self):
        self.heatmap_loss_object = FocalLoss()
        self.reg_loss_object = RegL1Loss()
        self.wh_loss_object = RegL1Loss()
        
        # self.joint_loss_object = FocalLoss()
        # self.joint_loc_loss_object = RegL1Loss_joint_loc()

    def __call__(self, y_pred, heatmap_true, reg_true, wh_true, reg_mask, indices, *args, **kwargs):
        # print('*****************', y_pred.shape)
        heatmap, reg, wh = tf.split(value=y_pred, num_or_size_splits=[Config.num_classes, 2, 2], axis=-1)
        heatmap_loss = self.heatmap_loss_object(y_true=heatmap_true, y_pred=heatmap)
        off_loss = self.reg_loss_object(y_true=reg_true, y_pred=reg, mask=reg_mask, index=indices)
        wh_loss  = self.wh_loss_object(y_true=wh_true, y_pred=wh, mask=reg_mask, index=indices)
        
        # joint_loss = self.joint_loss_object(y_true=joint_true[...,5:], y_pred=joint[...,5:])
        # joint_loc_loss = self.joint_loc_loss_object(y_true=joint_loc_true, y_pred=joint_loc, mask=joint_reg_mask, index=joint_indices)
        
        total_loss = (Config.hm_weight * heatmap_loss + 
                      Config.off_weight * off_loss + 
                      Config.wh_weight * wh_loss)
        return total_loss, heatmap_loss, wh_loss, off_loss
    
    
