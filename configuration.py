import os

def get_txt_combine(txt_file_path, txt_item, training = None):
    train_line = []
    for file in txt_item:
        with open(txt_file_path + file + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                train_line.append(line)
                
    train_flag={False:'data_val.txt', True:'data_train.txt'}
    with open(txt_file_path + train_flag[training], 'w') as fw:     
        for i in train_line:
            fw.write(i + '\n')                     

txt_file_path = './txt_file/pose_detection/'

class Config:
    epochs = 50
    batch_size = 12
    learning_rate_decay_epochs = 10
    learning_rate_decay_step = 5000

    # save model
    save_frequency = 1
    save_model_dir = "saved_model/"
    load_weights_before_training = False
    load_weights_from_epoch = 0

    # test image
    test_single_image_dir = "./test_pictures/street.jpg"
    test_images_during_training = True
    val_images_during_training_step_save_frequency = 400
    test_images_during_training_step_save_frequency = 100
    training_results_save_dir = "./test_pictures/"
    
    test_images_dir_list = [os.path.join('./test_pictures/test_sample_joint', f) 
                            for f in os.listdir('./test_pictures/test_sample_joint')]
    # test_images_dir_list = ["/home/thomas_yang/ML/hTG_ObjectDetection/data/datasets/Viveland/video_1/00000009.jpg"]

    # network architecture
    backbone_name = "mobilenetv2"
    # can be selected from: resnet_18, resnet_34, resnet_50, resnet_101, resnet_152, D0~D7, mobilenetv2

    # efficientdet: 8, others: 4 
    if 'D' in backbone_name:        
        downsampling_ratio = 8
    else:
        downsampling_ratio = 4

    image_size = {"resnet_18": (384, 384), "resnet_34": (384, 384), "resnet_50": (384, 384),
                  "resnet_101": (384, 384), "resnet_152": (384, 384),
                  "D0": (512, 512), "D1": (640, 640), "D2": (768, 768),
                  "D3": (896, 896), "D4": (1024, 1024), "D5": (1280, 1280),
                  "D6": (1408, 1408), "D7": (1536, 1536),
                  "mobilenetv2": (416, 416)}
    image_channels = 3

    # dataset
    num_classes = 1
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"
    pascal_voc_classes = {"person": 0}
    
    num_joints = 17
    num_joints_loc = 34

    # train txt file    
    train_txt_item = ['vive_land_autolabel_vote_1_.txt',
                      'vive_land_autolabel_vote_2_.txt',
                      'vive_land_autolabel_vote_3_.txt',
                      'vive_land_autolabel_vote_7_.txt',
                      'crowpose_autolabel_vote.txt',
                      'mpii_autolabel_vote.txt',
                      'mhp_autolabel_vote.txt']      
    txt_file_dir = [txt_file_path + i for i in train_txt_item]
    
    # val txt file
    val_txt_item = ['validation_vive_land_manual_16.txt']
    val_txt_file_dir = [txt_file_path + i for i in val_txt_item]
        
    max_boxes_per_image = 50
    max_joints_per_image = 50

    # efficientdet
    width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
    depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
    dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}
    # bifpn channels
    w_bifpn = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
    # bifpn layers
    d_bifpn = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}
    
    # model detection-head
    """ 
    coco : 0-nose    1-Leye    2-Reye    3-Lear    4Rear    5-Lsho  
           6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri  11-Lhip
           12-Rhip   13-Lkne   14-Rkne   15-Lank   16-Rank
    """
    heads = {"heatmap": num_classes, "wh": 2, "reg": 2, "joint": 17, "joint_loc": 34}
    
    head_conv = {"no_conv_layer": 0, "resnets": 64, "dla": 256,
                 "D0": w_bifpn["D0"], "D1": w_bifpn["D1"], "D2": w_bifpn["D2"], "D3": w_bifpn["D3"],
                 "D4": w_bifpn["D4"], "D5": w_bifpn["D5"], "D6": w_bifpn["D6"], "D7": w_bifpn["D7"],
                 "mobilenetv2": 128}


    # loss weight
    hm_weight = 1.0 #1.0
    wh_weight = 0.2 #0.2
    off_weight = 1.0#1.0
    joint_weight = 1.0 #1.5
    joint_loc_weight = 1.0

    score_threshold = 0.3
    joint_threshold = 0.1
    
    skeleton = [[0, 1], [1, 3], [5, 7], [7,  9], [11, 13], [13, 15],
                [0, 2], [2, 4], [6, 8], [8, 10], [12, 14], [14, 16],
                [5, 6], [6, 12], [12, 11], [11, 5]]


    @classmethod
    def get_image_size(cls):
        return cls.image_size[cls.backbone_name]

    @classmethod
    def get_width_coefficient(cls, backbone_name):
        return cls.width_coefficient[backbone_name]

    @classmethod
    def get_depth_coefficient(cls, backbone_name):
        return cls.depth_coefficient[backbone_name]

    @classmethod
    def get_dropout_rate(cls, backbone_name):
        return cls.dropout_rate[backbone_name]

    @classmethod
    def get_w_bifpn(cls, backbone_name):
        return cls.w_bifpn[backbone_name]

    @classmethod
    def get_d_bifpn(cls, backbone_name):
        return cls.d_bifpn[backbone_name]
