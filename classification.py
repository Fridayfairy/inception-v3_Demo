import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path,uid_lookup_path)
        
    def load(self,label_lookup_path,uid_lookup_path):
        #imagenet_2012_challenge_label_map_proto.pbtxt  
        #target_class to target_class_string: "n01440764"
        class_to_uid = {}
        class_to_uid_lines = tf.gfile.GFile(label_lookup_path).readlines()        
        for line in class_to_uid_lines:
            if line.startswith('  target_class:'):
                classnum = int(line.split(':')[-1])
            if line.startswith('  target_class_string:'):
                uid = line.split(':')[-1][2:-2]
                class_to_uid[classnum] = uid
        #imagenet_synset_to_human_label_map.txt
        #n00004475 to organism, being        
        uid_to_obj = {}
        uid_to_obj_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        for line in uid_to_obj_lines:
            line.strip('\n')
            temp = line.split('\t')
            uid = temp[0]
            obj = temp[1]
            uid_to_obj[uid] = obj
            
        class_to_obj = {}
        for class_temp, uid_temp in class_to_uid.items():
            obj_temp = uid_to_obj[uid_temp]
            class_to_obj[class_temp] = obj_temp
        
        return class_to_obj
    
    def lookup_obj_by_class(self,classnum):
        if classnum not in self.node_lookup:
            return ""
        return self.node_lookup[classnum]
    
    
with tf.gfile.GFile('inception_model/classify_image_graph_def.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
    
with tf.Session() as sess:
    #给Graph取个名字
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    for root,dirs,files in os.walk('images/'):
        for file in files:
            image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            prediction = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            #print("压缩前，prediction格式是",prediction)
            prediction = np.squeeze(prediction)
            #print("压缩后，prediction格式是",prediction)
            image_path = os.path.join(root,file)
            print(image_path)
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            top_k = prediction.argsort()[-5:][::-1]
            #print("top_k are: ", top_k)
            node_lookup = NodeLookup()
            for classnum in top_k:
                print(classnum)
                obj_string = node_lookup.lookup_obj_by_class(classnum)
                scores = prediction[classnum]
                print('%s (scores = %.5f)' %(obj_string,scores))
                
            print()













